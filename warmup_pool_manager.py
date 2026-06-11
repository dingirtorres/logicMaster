# warmup_pool_manager.py
#
# Gestor de warmup paralelo de indicadores.
#
# Responsabilidad única:
#   Procesar el histórico completo de N tokens en paralelo,
#   producir el tensor de indicadores de cada uno, y entregarlo
#   al MainController para inyección en el MarketBufferHandler.
#
# Lo que HACE:
#   - Detecta P-cores en arquitectura Intel híbrida (Alder Lake+).
#   - Calibra el costo real de RAM por instancia de LogicMaster.
#   - Calcula el número seguro de workers concurrentes.
#   - Spawnea un proceso por token con CPU affinity a P-cores.
#   - Envía tensores grandes via Pipe.send_bytes() (numpy tobytes, sin pickle).
#   - Acepta tokens dinámicamente mientras otros workers corren.
#   - Emite señales Qt de progreso y completion por token.
#
# Lo que NO HACE:
#   - No toca el MarketBufferHandler directamente.
#   - No inicia el WebSocket.
#   - No gestiona el estado incremental del CalculusManager.
#
# Protocolo Pipe worker → manager:
#   1. conn.send(meta_dict)        ← pequeño dict con metadata, pickle rápido
#   2. conn.send_bytes(tensor_raw) ← numpy 2D (n_candles × n_ind) tobytes, sin pickle
#
# Integración con MainController:
#   sig_primary_ready(token_id, preheating_min) → Phase B + CMD_WARMUP al CM
#   sig_tensor_ready(token_id, preheating_min)  → inyección buffer + CMD_WARMUP al CM
#   MainController llama pool.pop_tensor(token_id) para obtener los datos.

from __future__ import annotations

import gc
import importlib.util
import json
import logging
import multiprocessing
import os
import queue
import re
import sys
import threading
import time
import weakref
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
from PyQt6.QtCore import QThread, pyqtSignal


log = logging.getLogger(__name__)


# ==============================================================================
# STUB para procesador_ref
# ==============================================================================

class _LMProcessorStub:
    """
    Adaptador mínimo que satisface la interfaz procesador_ref del LogicMaster.

    LogicMaster necesita solo dos atributos de su procesador_ref:
      - lm_config_dir:      directorio que contiene logicMaster.json del token.
      - buffer_size_config: tamaño de buffer (int) o None → LM usa su default.
    """

    def __init__(self, lm_config_dir: str, buffer_size_config: Optional[int] = None):
        self.lm_config_dir    = lm_config_dir
        self.buffer_size_config = buffer_size_config


# ==============================================================================
# UTILIDADES DE HARDWARE
# ==============================================================================

def _detect_performance_cores() -> List[int]:
    """
    Detecta los índices de P-cores en arquitectura Intel híbrida (Alder Lake+).

    Lee /sys/devices/system/cpu/cpuN/topology/core_type en Linux.
    Retorna una lista de índices (ej. [0, 1, 2, 3, 16, 17, 18, 19]).
    Fallback: todos los núcleos físicos si la información no está disponible.
    """
    perf: List[int] = []
    try:
        cpu_base = "/sys/devices/system/cpu"
        for entry in sorted(os.listdir(cpu_base)):
            if not re.match(r"^cpu\d+$", entry):
                continue
            type_path = os.path.join(cpu_base, entry, "topology", "core_type")
            try:
                with open(type_path) as f:
                    if "Performance" in f.read():
                        perf.append(int(entry[3:]))
            except (OSError, ValueError):
                pass
    except Exception:
        pass

    fallback = list(range(psutil.cpu_count(logical=False) or 4))
    result = perf if perf else fallback
    log.info("[WarmupPool] P-cores detectados: %s", result)
    return result


def _calculate_safe_workers(worker_cost_mb: float, p_cores: List[int]) -> int:
    """
    Número seguro de workers concurrentes dado el costo RAM medido por el canario.

    Limitado por:
      - RAM disponible × 0.80 / (costo × 1.20 de headroom)
      - Cantidad de P-cores disponibles
    """
    if worker_cost_mb <= 0:
        worker_cost_mb = 300.0

    avail_mb   = (psutil.virtual_memory().available / (1024 ** 2)) * 0.80
    by_ram     = max(1, int(avail_mb / (worker_cost_mb * 1.20)))
    by_cpu     = max(1, len(p_cores))
    n_workers  = min(by_ram, by_cpu)

    log.info(
        "[WarmupPool] RAM disponible: %.0f MB | costo/worker: %.0f MB | "
        "por RAM: %d | por CPU: %d → workers: %d",
        avail_mb, worker_cost_mb, by_ram, by_cpu, n_workers,
    )
    return n_workers


# ==============================================================================
# FUNCIONES STANDALONE (corren en procesos hijo)
# ==============================================================================

def _calibrate_worker(
    candles: List[Dict],
    lm_module_path: str,
    lm_config_dir: str,
    result_queue: multiprocessing.Queue,
) -> None:
    """
    Proceso canario: mide el costo real de RAM de una instancia de LogicMaster.
    Procesa solo preheating + 200 velas para ser rápido.
    Envía el resultado a result_queue como {"status", "ram_mb"}.
    """
    try:
        proc      = psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / (1024 ** 2)

        stub = _LMProcessorStub(lm_config_dir=lm_config_dir)

        module_name = os.path.splitext(os.path.basename(lm_module_path))[0]
        spec   = importlib.util.spec_from_file_location(module_name, lm_module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        lm = getattr(module, "LogicMaster")(procesador_ref=stub)

        for candle in candles:
            lm.process_candle(candle)

        mem_after = proc.memory_info().rss / (1024 ** 2)
        delta = max(mem_after - mem_before, 80.0)
        result_queue.put({"status": "OK", "ram_mb": delta})

    except Exception as e:
        result_queue.put({"status": "ERROR", "ram_mb": 300.0, "error": str(e)})


def _warmup_worker(
    token_id: str,
    candles: List[Dict],
    lm_module_path: str,
    lm_config_dir: str,
    result_conn: multiprocessing.Connection,
    log_queue: multiprocessing.Queue,
    p_cores: List[int],
    worker_index: int,
) -> None:
    """
    Worker de warmup: procesa TODAS las velas del token y envía el tensor
    al WarmupPoolManager via Pipe.

    Protocolo de envío:
      1. result_conn.send(meta_dict)        ← metadata, pickle rápido (<1KB)
      2. result_conn.send_bytes(raw_bytes)  ← tensor numpy tobytes, sin pickle
    """
    # Pinear a P-core asignado
    if p_cores:
        try:
            core = p_cores[worker_index % len(p_cores)]
            os.sched_setaffinity(0, {core})
        except Exception:
            pass

    # Logger IPC: envía mensajes a log_queue para que el manager los emita
    class _IPCHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                if "__PROGRESS__" in msg:
                    log_queue.put_nowait(("PROGRESS", token_id, msg))
                else:
                    log_queue.put_nowait(("LOG", token_id, msg))
            except Exception:
                pass

    logger = logging.getLogger(f"WarmupWorker.{token_id}")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    h = _IPCHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)

    def _report_progress(current: int, total: int) -> None:
        try:
            log_queue.put_nowait(("PROGRESS", token_id, current, total))
        except Exception:
            pass

    try:
        # Leer preheating del config del perfil
        config_path = os.path.join(lm_config_dir, "logicMaster.json")
        preheating_min = 200
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            preheating_min = int(cfg.get("preheating", 200))
        except Exception:
            pass

        # Instanciar LogicMaster con el stub (sin CalculusManager real)
        stub = _LMProcessorStub(lm_config_dir=lm_config_dir)

        module_name = os.path.splitext(os.path.basename(lm_module_path))[0]
        spec   = importlib.util.spec_from_file_location(module_name, lm_module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        lm = getattr(module, "LogicMaster")(procesador_ref=stub)

        total        = len(candles)
        report_every = max(500, total // 20)  # ~20 reportes por token
        all_caches: List[Dict] = []
        all_timestamps: List[int] = []

        for i, candle in enumerate(candles):
            lm.process_candle(candle)

            cache = dict(lm.indicator_cache)
            ts    = int(candle.get("timestamp", 0))

            if cache:
                all_caches.append(cache)
                all_timestamps.append(ts)

            if (i + 1) % report_every == 0 or (i + 1) == total:
                _report_progress(i + 1, total)

        # Construir tensor 2D: shape (n_candles, n_indicators)
        if not all_caches:
            raise ValueError("Ningún cache generado — el LogicMaster no produjo indicadores.")

        output_keys = list(all_caches[0].keys())
        n_candles   = len(all_caches)
        n_ind       = len(output_keys)

        # Un array por clave, luego stack en columnas (más rápido que bucle anidado)
        cols = []
        for key in output_keys:
            col = np.array(
                [float(c[key]) if c.get(key) is not None else np.nan
                 for c in all_caches],
                dtype=np.float64,
            )
            cols.append(col)

        packed = np.column_stack(cols)           # shape: (n_candles, n_ind)
        ts_arr = np.array(all_timestamps, dtype=np.int64)

        # ── Envío 1: metadata (pequeño dict, pickle es rápido) ──
        result_conn.send({
            "token_id":      token_id,
            "status":        "OK",
            "preheating_min": preheating_min,
            "output_keys":   output_keys,
            "n_candles":     n_candles,
            "n_indicators":  n_ind,
        })

        # ── Envío 2: tensor como bytes crudos (sin pickle) ──
        result_conn.send_bytes(packed.tobytes())

        # ── Envío 3: timestamps como bytes crudos (sin pickle) ──
        result_conn.send_bytes(ts_arr.tobytes())

    except Exception as e:
        log.error("[WarmupWorker] Error en %s: %s", token_id, e)
        try:
            result_conn.send({
                "token_id":      token_id,
                "status":        "ERROR",
                "error":         str(e),
                "preheating_min": 200,
                "output_keys":   [],
                "n_candles":     0,
                "n_indicators":  0,
            })
            result_conn.send_bytes(b"")
            result_conn.send_bytes(b"")
        except Exception:
            pass

    finally:
        try:
            result_conn.close()
        except Exception:
            pass
        gc.collect()


# ==============================================================================
# MANAGER PRINCIPAL
# ==============================================================================

class WarmupPoolManager(QThread):
    """
    Gestor de warmup paralelo de indicadores.

    Ciclo de vida:
      1. MainController llama submit_token(primary, candles, ...) antes de start().
      2. start() arranca el QThread → run() comienza.
      3. run() calibra con el primer token, calcula max_workers, spawnea workers.
      4. _load_secondary_tokens llama submit_token() conforme llegan datos.
      5. Workers terminan → sig_primary_ready / sig_tensor_ready.
      6. MainController llama pop_tensor(token_id) para obtener los datos.
      7. Cuando todos terminan → sig_pool_complete.
    """

    # token primario listo → MainController dispara Phase B + CMD_WARMUP(preheating)
    sig_primary_ready = pyqtSignal(str, int)   # token_id, preheating_min

    # token secundario listo → MainController inyecta buffer + CMD_WARMUP(preheating)
    sig_tensor_ready  = pyqtSignal(str, int)   # token_id, preheating_min

    # progreso de cálculo por token
    sig_progress      = pyqtSignal(str, int, int)  # token_id, current, total

    # error en un token
    sig_error         = pyqtSignal(str, str)   # token_id, error_msg

    # todos los tokens completados o con error
    sig_pool_complete = pyqtSignal()

    def __init__(
        self,
        primary_token: str,
        total_tokens: int,
        parent=None,
    ):
        super().__init__(parent)
        self._primary_token = primary_token.upper()
        self._total_tokens  = total_tokens
        self._running       = True

        # Cola thread-safe: MainController → WarmupPoolManager
        # Acepta tuplas (token_id, candles, lm_module_path, lm_config_dir)
        # o None como señal de parada.
        self._submit_q: queue.Queue = queue.Queue()

        # Tensores listos para que el MainController los consuma via pop_tensor()
        self._tensors: Dict[str, Dict[str, np.ndarray]] = {}
        self._tensor_lock = threading.Lock()

        # Hardware
        self._p_cores    = _detect_performance_cores()
        self._max_workers = 1         # se recalibra en run()
        self._calibrated  = False

        # Workers activos: token_id → (Process, parent_conn)
        self._active: Dict[str, Tuple[multiprocessing.Process,
                                      multiprocessing.Connection]] = {}

        # Queue IPC para logs y progreso de workers
        self._log_queue: multiprocessing.Queue = multiprocessing.Queue()

        self._done_count   = 0
        self._worker_index = 0

    # ── API pública ────────────────────────────────────────────────────────────

    def submit_token(
        self,
        token_id: str,
        candles: List[Dict],
        lm_module_path: str,
        lm_config_dir: str,
    ) -> None:
        """
        Encola un token para warmup.
        Thread-safe — se puede llamar desde el hilo Qt en cualquier momento.
        """
        self._submit_q.put((token_id.upper(), candles, lm_module_path, lm_config_dir))
        log.info("[WarmupPool] Token encolado: %s (%d velas)", token_id, len(candles))

    def pop_tensor(
        self, token_id: str
    ) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        """
        Devuelve y elimina (tensor_dict, timestamps_array) para un token.
        Llamar desde el slot conectado a sig_primary_ready / sig_tensor_ready.
        Retorna None si el tensor ya fue consumido o nunca llegó.
        """
        with self._tensor_lock:
            return self._tensors.pop(token_id.upper(), None)

    def stop(self) -> None:
        """Parada segura: termina workers activos y desbloquea el loop."""
        self._running = False
        self._submit_q.put(None)
        for token_id, (proc, conn) in list(self._active.items()):
            try:
                proc.terminate()
                conn.close()
            except Exception:
                pass

    # ── Loop principal (QThread.run) ───────────────────────────────────────────

    def run(self) -> None:
        """
        Loop de gestión del pool:
          1. Acepta tokens de _submit_q.
          2. Calibra con el primero (bloquea ~10-30s).
          3. Spawnea workers hasta max_workers simultáneos.
          4. Recolecta resultados y emite señales.
          5. Acepta tokens adicionales dinámicamente.
          6. Sale cuando done_count == total_tokens o se pide parada.
        """
        pending: List[Tuple] = []   # tokens esperando slot libre

        while self._running:

            # ── Drenar mensajes de progreso de workers ──
            self._drain_log_queue()

            # ── Aceptar nuevos tokens enviados desde el hilo Qt ──
            while True:
                try:
                    item = self._submit_q.get_nowait()
                except queue.Empty:
                    break

                if item is None:   # señal de parada
                    self._running = False
                    break

                pending.append(item)

                # Calibrar con el primer token (primario)
                if not self._calibrated:
                    token_id, candles, lm_module, lm_cfg_dir = item
                    self._max_workers = self._calibrate(
                        candles, lm_module, lm_cfg_dir
                    )
                    self._calibrated = True
                    log.info(
                        "[WarmupPool] max_workers = %d", self._max_workers
                    )

            # ── Spawnear workers si hay slots libres y calibración hecha ──
            while (
                pending
                and self._calibrated
                and len(self._active) < self._max_workers
            ):
                args = pending.pop(0)
                self._spawn(*args)

            # ── Recolectar workers que terminaron ──
            for token_id in list(self._active.keys()):
                proc, conn = self._active[token_id]
                if conn.poll(0):
                    self._collect(token_id, proc, conn)

            # ── Detectar workers muertos sin enviar datos ──
            for token_id in list(self._active.keys()):
                proc, conn = self._active[token_id]
                if not proc.is_alive() and not conn.poll(0):
                    log.error(
                        "[WarmupPool] Worker muerto sin enviar: %s", token_id
                    )
                    self.sig_error.emit(token_id, "Worker terminó sin enviar tensor")
                    self._cleanup(token_id)
                    self._on_done(token_id)

            # ── Condición de salida ──
            if (
                self._done_count >= self._total_tokens
                and self._submit_q.empty()
                and not self._active
                and not pending
            ):
                break

            time.sleep(0.05)

        self.sig_pool_complete.emit()
        log.info("[WarmupPool] Pool completo. Procesados: %d", self._done_count)

    # ── Internos ───────────────────────────────────────────────────────────────

    def _calibrate(
        self,
        candles: List[Dict],
        lm_module_path: str,
        lm_config_dir: str,
    ) -> int:
        """
        Corre el proceso canario y retorna el número seguro de workers.
        Bloquea el QThread hasta que el canario termina (~10-30 segundos).
        """
        # Solo preheating+200 velas para una calibración rápida
        try:
            config_path = os.path.join(lm_config_dir, "logicMaster.json")
            with open(config_path) as f:
                cfg = json.load(f)
            preheating = int(cfg.get("preheating", 200))
        except Exception:
            preheating = 200

        calib_candles = candles[:preheating + 200]

        result_q: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_calibrate_worker,
            args=(calib_candles, lm_module_path, lm_config_dir, result_q),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=120)   # máximo 2 minutos

        ram_mb = 300.0
        try:
            res    = result_q.get_nowait()
            ram_mb = res.get("ram_mb", 300.0)
        except Exception:
            log.warning("[WarmupPool] Canario no respondió — usando 300 MB/worker.")

        return _calculate_safe_workers(ram_mb, self._p_cores)

    def _spawn(
        self,
        token_id: str,
        candles: List[Dict],
        lm_module_path: str,
        lm_config_dir: str,
    ) -> None:
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)

        proc = multiprocessing.Process(
            target=_warmup_worker,
            args=(
                token_id,
                candles,
                lm_module_path,
                lm_config_dir,
                child_conn,
                self._log_queue,
                self._p_cores,
                self._worker_index,
            ),
            daemon=True,
        )
        proc.start()
        child_conn.close()  # cerrar extremo hijo en el proceso padre

        self._active[token_id] = (proc, parent_conn)
        self._worker_index += 1

        log.info(
            "[WarmupPool] Worker spawneado: %s (idx=%d, p_core=%s)",
            token_id,
            self._worker_index - 1,
            self._p_cores[(self._worker_index - 1) % len(self._p_cores)]
            if self._p_cores else "N/A",
        )

    def _collect(
        self,
        token_id: str,
        proc: multiprocessing.Process,
        conn: multiprocessing.Connection,
    ) -> None:
        """
        Lee metadata + tensor + timestamps del Pipe y almacena el resultado.
        Se llama cuando conn.poll(0) retorna True (hay datos disponibles).
        """
        try:
            meta: Dict = conn.recv()

            if meta.get("status") != "OK":
                self.sig_error.emit(token_id, meta.get("error", "Error desconocido"))
                self._cleanup(token_id)
                self._on_done(token_id)
                return

            n_candles  = meta["n_candles"]
            n_ind      = meta["n_indicators"]
            output_keys = meta["output_keys"]

            # Tensor: bytes → numpy 2D → dict {output_key: array}
            raw_tensor = conn.recv_bytes()
            packed = np.frombuffer(raw_tensor, dtype=np.float64).reshape(n_candles, n_ind).copy()
            tensor = {output_keys[i]: packed[:, i] for i in range(n_ind)}

            # Timestamps: bytes → numpy 1D int64
            raw_ts     = conn.recv_bytes()
            timestamps = np.frombuffer(raw_ts, dtype=np.int64).copy()

            # Almacenar para que MainController consuma via pop_tensor()
            with self._tensor_lock:
                self._tensors[token_id] = (tensor, timestamps)

            self._cleanup(token_id)
            self._on_done(token_id)

            preheating_min = meta["preheating_min"]

            if token_id == self._primary_token:
                log.info("[WarmupPool] Primario listo: %s", token_id)
                self.sig_primary_ready.emit(token_id, preheating_min)
            else:
                log.info("[WarmupPool] Secundario listo: %s", token_id)
                self.sig_tensor_ready.emit(token_id, preheating_min)

        except Exception as e:
            log.error("[WarmupPool] Error recolectando %s: %s", token_id, e)
            self.sig_error.emit(token_id, str(e))
            self._cleanup(token_id)
            self._on_done(token_id)

    def _cleanup(self, token_id: str) -> None:
        entry = self._active.pop(token_id, None)
        if not entry:
            return
        proc, conn = entry
        try:
            conn.close()
        except Exception:
            pass
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        except Exception:
            pass

    def _on_done(self, token_id: str) -> None:
        self._done_count += 1

    def _drain_log_queue(self) -> None:
        while True:
            try:
                item = self._log_queue.get_nowait()
            except Exception:
                break
            if not item:
                break
            msg_type = item[0]
            token_id = item[1]
            if msg_type == "PROGRESS" and len(item) == 4:
                _, _, current, total = item
                self.sig_progress.emit(token_id, current, total)
