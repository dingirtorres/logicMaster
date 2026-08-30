# calculus_manager.py
#
# Motor de cálculo de indicadores técnicos de alta fidelidad.
#
# Responsabilidad única: ingesta, cómputo incremental y empaquetado
# de resultados de indicadores técnicos agrupados por base_name.
#
# Lo que HACE:
#   - Gestiona una colección de contextos de cálculo por token_id.
#   - Cada contexto contiene una instancia de LogicMaster y un tensor
#     NumPy de longitud fija para la producción de datos del Canvas.
#   - Ingesta masiva (warmup) con throttling configurable.
#   - Ingesta incremental tick-a-tick con ACK por Pipe bidireccional.
#   - Empaquetado de resultados por base_name para el PresentationManager.
#   - Reporte de PIDs de procesos hijos al MainController.
#   - Recepción de comandos administrativos por Queue de control.
#
# Lo que NO HACE:
#   - No decide colores, estilos ni presentación visual.
#   - No accede a ninguna API externa.
#   - No gestiona timers Qt ni interactúa con la UI.
#   - No modifica el MarketBufferHandler.
#
# Comunicación con MainController:
#   - Pipe bidireccional: flujo de datos (ticks entrada, resultados salida)
#   - Queue de control: comandos administrativos (WARMUP, RESET, SHUTDOWN, etc.)
#
# Arquitectura:
#   Corre como multiprocessing.Process independiente.
#   El MainController lee el pipe con QTimer (no bloqueante).
#   El CalculusManager bloquea en recv() en su propio proceso.

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import time
import numpy as np

from collections import deque
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "[CalculusManager] %(asctime)s - %(levelname)s - %(message)s"


# ---------------------------------------------------------------------------
# Comandos del canal de control (Queue)
# ---------------------------------------------------------------------------
CMD_WARMUP       = "WARMUP"
CMD_RESET        = "RESET"
CMD_ADD_TOKEN    = "ADD_TOKEN"
CMD_REMOVE_TOKEN = "REMOVE_TOKEN"
CMD_SHUTDOWN     = "SHUTDOWN"
CMD_RECONFIG     = "RECONFIG"

# ---------------------------------------------------------------------------
# Tipos de mensaje por Pipe (datos)
# ---------------------------------------------------------------------------
MSG_TICK         = "TICK"
MSG_ACK          = "ACK"
MSG_WARMUP_ACK   = "WARMUP_ACK"
MSG_EVENT        = "EVENT"
MSG_PID_REPORT   = "PID_REPORT"
MSG_ERROR        = "ERROR"
MSG_STATUS       = "STATUS"
MSG_SYNC         = "SYNC"


# ---------------------------------------------------------------------------
# Contexto de cálculo por token
# ---------------------------------------------------------------------------
class _CalculusContext:
    """
    Estado de cálculo aislado para un token_id.

    Contiene:
      - Instancia de LogicMaster (motor analítico).
      - Tensor NumPy de longitud fija (rolling buffer de indicadores).
      - Mapa de output_keys a columnas del tensor.
      - Índice de escritura circular.
      - Flag de warmup completado.
    """

    def __init__(
        self,
        token_id: str,
        logic_master_instance,
        tensor_length: int = 3000,
    ):
        self.token_id = token_id
        self.logic_master = logic_master_instance
        self.tensor_length = tensor_length

        # El mapa de columnas se construye dinámicamente al primer tick
        # porque los output_keys dependen de la configuración de indicadores
        # cargada por LogicMaster desde su JSON.
        self._column_map: Dict[str, int] = {}
        self._tensor: Optional[np.ndarray] = None
        self._write_idx: int = 0
        self._total_written: int = 0
        self._is_warmed_up: bool = False

        # Timestamps asociados a cada fila del tensor (para alineación)
        self._timestamps: Optional[np.ndarray] = None

        # Mapa output_key → base_name, construido desde LogicMaster.
        # Esta es la fuente de verdad para el agrupamiento por base_name.
        self._key_to_base: Dict[str, str] = {}
        self._build_base_name_map()

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def _build_base_name_map(self) -> None:
        """
        Construye el mapa output_key → base_name desde los indicadores
        cargados por LogicMaster.

        Fuente de verdad: cada instancia en active_indicators tiene:
          - .output_keys (list[str]): claves que genera (siempre presente)
          - .base_name (str): nombre base del grupo (solo en multi-output)

        Para single-output (SMA, EMA, RSI): output_keys tiene 1 elemento,
        no hay .base_name → la clave es su propio grupo.

        Para multi-output (BB, MACD, STOCH, CIN_LOG, GEO_LOG):
        output_keys tiene N elementos y .base_name indica el grupo.
        """
        if not self.logic_master or not hasattr(self.logic_master, 'active_indicators'):
            return

        for _inst_key, indicator in self.logic_master.active_indicators.items():
            output_keys = getattr(indicator, 'output_keys', [])
            base_name = getattr(indicator, 'base_name', None)

            if base_name and len(output_keys) > 1:
                # Multi-output: todas las claves mapean al base_name
                for ok in output_keys:
                    self._key_to_base[ok] = base_name
            else:
                # Single-output: la clave es su propio grupo
                for ok in output_keys:
                    self._key_to_base[ok] = ok

    def _ensure_tensor(self, indicator_cache: Dict[str, Any]) -> None:
        """
        Inicializa el tensor y el mapa de columnas al recibir
        el primer indicator_cache con datos reales.
        Se ejecuta una sola vez por contexto.
        """
        if self._tensor is not None:
            return

        # Construir mapa de columnas a partir de las claves del cache
        keys = sorted(indicator_cache.keys())
        self._column_map = {key: idx for idx, key in enumerate(keys)}
        num_cols = len(keys)

        # Tensor de producción: NaN por defecto (sin dato)
        self._tensor = np.full(
            (self.tensor_length, num_cols),
            np.nan,
            dtype=np.float64,
        )
        self._timestamps = np.zeros(self.tensor_length, dtype=np.int64)

    def append_tick(
        self,
        timestamp: int,
        indicator_cache: Dict[str, Any],
    ) -> None:
        """
        Apila un snapshot de indicadores en el tensor circular.
        Si aparecen claves nuevas que no estaban en el tensor original,
        se expande dinámicamente el mapa y el tensor.
        """
        self._ensure_tensor(indicator_cache)

        # Detectar claves nuevas y expandir si es necesario
        new_keys = [k for k in indicator_cache if k not in self._column_map]
        if new_keys:
            self._expand_tensor(new_keys)

        # Escribir en la posición circular
        idx = self._write_idx % self.tensor_length
        row = np.full(len(self._column_map), np.nan, dtype=np.float64)

        for key, value in indicator_cache.items():
            col = self._column_map.get(key)
            if col is not None and value is not None:
                try:
                    row[col] = float(value)
                except (ValueError, TypeError):
                    pass  # Mantiene NaN

        self._tensor[idx] = row
        self._timestamps[idx] = timestamp
        self._write_idx += 1
        self._total_written += 1

    def _expand_tensor(self, new_keys: List[str]) -> None:
        """
        Expande el tensor con columnas nuevas (inicializadas en NaN).
        Esto ocurre si LogicMaster carga indicadores dinámicamente
        después del primer tick.
        """
        old_cols = self._tensor.shape[1]
        for key in new_keys:
            self._column_map[key] = len(self._column_map)

        new_cols = len(self._column_map) - old_cols
        if new_cols > 0:
            expansion = np.full(
                (self.tensor_length, new_cols),
                np.nan,
                dtype=np.float64,
            )
            self._tensor = np.hstack([self._tensor, expansion])

    def get_latest_cache(self) -> Dict[str, Any]:
        """
        Devuelve el último snapshot del tensor como diccionario.
        Es la interfaz de salida hacia el Pipe.
        """
        if self._tensor is None or self._total_written == 0:
            return {}

        idx = (self._write_idx - 1) % self.tensor_length
        row = self._tensor[idx]
        result = {}
        for key, col in self._column_map.items():
            val = row[col]
            if np.isnan(val):
                result[key] = None
            else:
                result[key] = float(val)
        return result

    def get_tensor_slice(self, n_rows: int = 0) -> Dict[str, Any]:
        """
        Devuelve exactamente tensor_length filas para alineacion 1:1 con
        el buffer de velas del canvas.

        Cuando _total_written < tensor_length (precalentamiento del LM),
        las primeras (tensor_length - _total_written) posiciones se rellenan
        con NaN / timestamp=0. Esas posiciones corresponden a las velas de
        precalentamiento — el canvas las pinta sin indicador, que es correcto.

        Cuando el tensor ya dio la vuelta completa, mismo comportamiento
        que antes: devuelve las ultimas tensor_length filas reales.
        """
        if self._tensor is None or self._total_written == 0:
            return {"timestamps": [], "data": {}}

        written = min(self._total_written, self.tensor_length)

        if n_rows <= 0 or n_rows > self.tensor_length:
            n_rows = self.tensor_length

        # Caso 1: tensor completo o dio la vuelta
        if written >= n_rows:
            indices = []
            for i in range(n_rows):
                idx = (self._write_idx - n_rows + i) % self.tensor_length
                indices.append(idx)
            timestamps = self._timestamps[indices].tolist()
            data = {}
            for key, col in self._column_map.items():
                data[key] = self._tensor[indices, col].tolist()
            return {"timestamps": timestamps, "data": data}

        # Caso 2: tensor parcial — rellenar inicio con NaN
        pad = n_rows - written
        real_indices = []
        for i in range(written):
            idx = (self._write_idx - written + i) % self.tensor_length
            real_indices.append(idx)

        ts_real = self._timestamps[real_indices].tolist()
        timestamps = [0] * pad + ts_real

        data = {}
        nan_pad = [float("nan")] * pad
        for key, col in self._column_map.items():
            real_vals = self._tensor[real_indices, col].tolist()
            data[key] = nan_pad + real_vals

        return {"timestamps": timestamps, "data": data}

    def reset(self) -> None:
        """Limpia el tensor y los estados internos sin destruir LogicMaster."""
        if self._tensor is not None:
            self._tensor[:] = np.nan
            self._timestamps[:] = 0
        self._write_idx = 0
        self._total_written = 0
        self._is_warmed_up = False

    def close(self) -> None:
        """Libera recursos del contexto."""
        if self.logic_master and hasattr(self.logic_master, 'close'):
            try:
                self.logic_master.close()
            except Exception:
                pass
        self.logic_master = None
        self._tensor = None
        self._timestamps = None
        self._column_map.clear()


# ---------------------------------------------------------------------------
# CalculusManager (Proceso independiente)
# ---------------------------------------------------------------------------
class CalculusManager(multiprocessing.Process):
    """
    Motor de cálculo de indicadores técnicos.

    Corre como proceso independiente. Se comunica con el MainController
    mediante:
      - Pipe bidireccional: flujo de datos (ticks → resultados + ACK)
      - Queue de control: comandos administrativos

    Gestiona una colección de contextos por token_id. En V1, un solo
    token activo con hooks preparados para multi-token.

    Parámetros del constructor:
      pipe_conn         : lado del Pipe para este proceso
      control_queue     : Queue de comandos (WARMUP, RESET, SHUTDOWN, etc.)
      stop_event        : Event para señal de parada global
      config            : dict con parámetros de operación
      logic_module_path : ruta al módulo de LogicMaster
      logic_class_name  : nombre de la clase dentro del módulo

    Config esperada:
      tensor_length       : int  (default 3000) - filas del tensor
      pausa_ms            : int  (default 10)   - pausa entre ráfagas de warmup
      pausa_cada_n_velas  : int  (default 500)  - velas por ráfaga de warmup
      buffer_size_config  : int  (default 2000) - tamaño de buffer para LogicMaster
      log_level           : str  (default "INFO")
    """

    def __init__(
        self,
        pipe_conn: multiprocessing.connection.Connection,
        control_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
        config: Dict[str, Any] = None,
        logic_module_path: str = None,
        logic_class_name: str = "LogicMaster",
    ):
        super().__init__(daemon=True, name="CalculusManager")

        self._pipe = pipe_conn
        self._control_queue = control_queue
        self._stop_event = stop_event

        # Configuración con defaults
        self._config = {
            "tensor_length":      3000,
            "pausa_ms":           10,
            "pausa_cada_n_velas": 500,
            "buffer_size_config": 2000,
            "log_level":          "INFO",
        }
        if config:
            self._config.update(config)

        self._logic_module_path = logic_module_path
        self._logic_class_name  = logic_class_name

        # Exponer buffer_size_config para que LogicMaster lo lea
        # via procesador_ref.buffer_size_config (contrato de realTime_94)
        self.buffer_size_config = self._config["buffer_size_config"]

        # Mapa {TOKEN: ruta_módulo} leído desde calculus_manager.json.
        # Permite que cada token use su propio módulo LogicMaster.
        # Si el archivo no existe o un token no tiene entrada,
        # se usa _logic_module_path como fallback.
        self._token_module_paths: Dict[str, str] = self._load_token_module_map()

        # Colección de contextos por token_id
        self._contexts: Dict[str, _CalculusContext] = {}
        self._active_token: Optional[str] = None

        # Logger (se configura en run() porque estamos en otro proceso)
        self._logger = None

    # ------------------------------------------------------------------
    # Configuración de rutas por token
    # ------------------------------------------------------------------

    def _load_token_module_map(self) -> Dict[str, str]:
        """
        Lee calculus_manager.json desde el directorio de trabajo.
        Retorna {TOKEN_UPPER: ruta_módulo}.
        Si el archivo no existe retorna dict vacío — se usará
        _logic_module_path como fallback para todos los tokens.
        """
        import json as _json
        cfg_path = os.path.join(os.getcwd(), "calculus_manager.json")
        if not os.path.exists(cfg_path):
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if isinstance(data, dict):
                return {k.upper(): v for k, v in data.items() if v}
        except Exception as e:
            # No bloquear el arranque si el archivo está corrupto
            print(f"[CalculusManager] Advertencia: no se pudo leer "
                  f"calculus_manager.json: {e}")
        return {}

    # ------------------------------------------------------------------
    # Carga dinámica de módulos (patrón de realTime_94)
    # ------------------------------------------------------------------

    def _load_logic_class(self):
        """
        Carga la clase LogicMaster desde el path configurado.
        Usa el mismo patrón de importación dinámica que realTime_94
        y procesador_4_2.
        """
        import importlib.util

        if not self._logic_module_path or not self._logic_class_name:
            raise ValueError(
                "logic_module_path y logic_class_name son requeridos "
                "para cargar el motor analítico."
            )

        module_name = os.path.splitext(
            os.path.basename(self._logic_module_path)
        )[0]
        spec = importlib.util.spec_from_file_location(
            module_name, self._logic_module_path
        )
        if spec is None:
            raise ImportError(
                f"No se pudo crear la especificación para "
                f"{self._logic_module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, self._logic_class_name)

    def _create_logic_master(self, token_id: str = "") -> object:
        """
        Instancia un LogicMaster pasando self como procesador_ref.

        Resolución del módulo (en orden de prioridad):
          1. Ruta en _token_module_paths[token_id] — configurada por
             calculus_manager.json (escrito por CalculusConfigDialog).
          2. _logic_module_path global — fallback para tokens sin
             configuración específica.

        Al cargar el módulo desde la carpeta del perfil del token,
        __file__ dentro de LogicMaster apunta a esa carpeta, por lo que
        _load_config() leerá el logicMaster.json del perfil correcto —
        exactamente el mismo patrón que usa realTime_94.
        """
        token_path = self._token_module_paths.get(token_id.upper(), "")
        module_path = token_path if token_path else self._logic_module_path

        if not module_path:
            raise ValueError(
                f"No hay ruta de módulo configurada para el token '{token_id}'. "
                "Configuralo en calculus_manager.json o pasá logic_module_path "
                "al instanciar el CalculusManager."
            )

        # Guardar el path activo temporalmente para _load_logic_class
        original_path = self._logic_module_path
        self._logic_module_path = module_path
        try:
            LogicMasterClass = self._load_logic_class()
            return LogicMasterClass(procesador_ref=self)
        finally:
            self._logic_module_path = original_path

    # ------------------------------------------------------------------
    # Gestión de contextos
    # ------------------------------------------------------------------

    def _add_context(self, token_id: str) -> _CalculusContext:
        """
        Crea un nuevo contexto de cálculo para un token.

        El módulo LogicMaster a usar se resuelve en _create_logic_master
        consultando _token_module_paths — no se necesita pasar nada extra.
        """
        if token_id in self._contexts:
            self._logger.warning(
                "Contexto para %s ya existe. Se reutiliza.", token_id
            )
            return self._contexts[token_id]

        logic_master = self._create_logic_master(token_id)
        ctx = _CalculusContext(
            token_id=token_id,
            logic_master_instance=logic_master,
            tensor_length=self._config["tensor_length"],
        )
        self._contexts[token_id] = ctx
        self._logger.info("Contexto creado para token: %s", token_id)
        return ctx

    def _remove_context(self, token_id: str) -> None:
        """Destruye un contexto y libera recursos."""
        ctx = self._contexts.pop(token_id, None)
        if ctx:
            ctx.close()
            self._logger.info("Contexto destruido para token: %s", token_id)

        if self._active_token == token_id:
            self._active_token = None

    def _reset_context(self, token_id: str) -> None:
        """Resetea el tensor y el estado del LogicMaster para un token."""
        ctx = self._contexts.get(token_id)
        if not ctx:
            self._logger.warning(
                "Reset solicitado para token inexistente: %s", token_id
            )
            return

        # Destruir y recrear LogicMaster para limpiar estados internos
        if ctx.logic_master and hasattr(ctx.logic_master, 'close'):
            ctx.logic_master.close()

        ctx.logic_master = self._create_logic_master()
        ctx.reset()
        self._logger.info("Contexto reseteado para token: %s", token_id)

    def _get_active_context(self) -> Optional[_CalculusContext]:
        """Devuelve el contexto del token activo o None."""
        if not self._active_token:
            return None
        return self._contexts.get(self._active_token)

    # ------------------------------------------------------------------
    # Warmup (ingesta masiva con throttling)
    # ------------------------------------------------------------------

    def _warmup(self, token_id: str, candles: List[Dict], skip_series: bool = False) -> None:
        """
        Inyecta el histórico completo en LogicMaster con throttling.

        skip_series=True: envía WARMUP_ACK sin indicators_series ni timestamps.
        Usar cuando el WarmupPool ya inyectó la serie completa en el buffer —
        el CM solo necesita construir su estado interno (mini-warmup).
        """
        ctx = self._contexts.get(token_id)
        if not ctx:
            self._logger.error(
                "Warmup: contexto no encontrado para %s", token_id
            )
            self._pipe.send({
                "type": MSG_ERROR,
                "error": f"Contexto no encontrado para {token_id}",
            })
            return

        total = len(candles)
        pausa_ms = self._config["pausa_ms"]
        pausa_cada_n = self._config["pausa_cada_n_velas"]

        self._logger.info(
            "Warmup iniciado: %d velas para %s "
            "(ráfaga: %d velas, pausa: %dms)",
            total, token_id, pausa_cada_n, pausa_ms,
        )

        # Notificar inicio
        self._pipe.send({
            "type": MSG_STATUS,
            "status": "warmup_started",
            "token_id": token_id,
            "total_candles": total,
        })

        for idx, candle in enumerate(candles):
            # Verificar stop global
            if self._stop_event.is_set():
                self._logger.warning("Warmup interrumpido por stop_event.")
                return

            # Inyectar vela en LogicMaster
            ctx.logic_master.process_candle(candle)

            # Recoger indicator_cache y apilar en tensor
            cache = dict(ctx.logic_master.indicator_cache)
            ts = candle.get("timestamp", 0)
            if cache:
                ctx.append_tick(ts, cache)

            # Recoger eventos (alertas, aperturas, cierres)
            if ctx.logic_master.has_new_events():
                events = ctx.logic_master.get_new_events()
                for evt in events:
                    self._pipe.send({
                        "type": MSG_EVENT,
                        "token_id": token_id,
                        "event": evt,
                    })

            # Throttling: pausa cada N velas
            if pausa_cada_n > 0 and (idx + 1) % pausa_cada_n == 0:
                if pausa_ms > 0:
                    time.sleep(pausa_ms / 1000.0)

                # Reporte de progreso
                self._pipe.send({
                    "type": MSG_STATUS,
                    "status": "warmup_progress",
                    "token_id": token_id,
                    "processed": idx + 1,
                    "total": total,
                })

        ctx._is_warmed_up = True

        # ACK de warmup completado.
        # skip_series=True (mini-warmup vía WarmupPool): ACK slim sin series —
        # el tensor histórico ya está en el buffer, el CM solo construyó estado.
        # skip_series=False (warmup clásico): ACK completo con tensor slice.
        latest = ctx.get_latest_cache()
        packed = self._pack_by_base_name(latest, ctx._key_to_base)

        if skip_series:
            self._pipe.send({
                "type":             MSG_WARMUP_ACK,
                "token_id":         token_id,
                "total_processed":  total,
                "indicators":       packed,       # para inspect_and_compile_matrix
                "indicators_flat":  latest,       # snapshot actual
                "indicators_series": {},          # vacío — tensor ya en buffer
                "timestamps":       [],
                "is_warmed_up":     True,
                "is_mini":          True,         # flag para _on_warmup_ack
            })
        else:
            tensor_data = ctx.get_tensor_slice()
            self._pipe.send({
                "type":             MSG_WARMUP_ACK,
                "token_id":         token_id,
                "total_processed":  total,
                "indicators":       packed,
                "indicators_flat":  latest,
                "indicators_series": tensor_data["data"],
                "timestamps":       tensor_data["timestamps"],
                "is_warmed_up":     True,
            })

        self._logger.info(
            "Warmup completado: %d velas para %s (mini=%s).",
            total, token_id, skip_series,
        )

    # ------------------------------------------------------------------
    # Procesamiento tick-a-tick
    # ------------------------------------------------------------------

    def _process_tick(
        self,
        token_id: str,
        candle: Dict,
    ) -> Dict[str, Any]:
        """
        Procesa un tick incremental:
          1. Inyecta en LogicMaster.process_candle()
          2. Lee indicator_cache
          3. Apila en tensor
          4. Recoge eventos
          5. Empaqueta por base_name
          6. Devuelve el diccionario ACK

        Retorna el mensaje ACK completo para enviar por Pipe.
        """
        ctx = self._contexts.get(token_id)
        if not ctx:
            return {
                "type": MSG_ERROR,
                "error": f"Contexto no encontrado para {token_id}",
            }

        # 1. Inyectar
        ctx.logic_master.process_candle(candle)

        # 2. Leer cache
        cache = dict(ctx.logic_master.indicator_cache)
        ts = candle.get("timestamp", 0)

        # 3. Apilar en tensor
        if cache:
            ctx.append_tick(ts, cache)

        # 4. Recoger eventos
        events = []
        if ctx.logic_master.has_new_events():
            events = ctx.logic_master.get_new_events()

        # 5. Empaquetar
        packed = self._pack_by_base_name(cache, ctx._key_to_base)

        # 6. ACK
        return {
            "type": MSG_ACK,
            "token_id": token_id,
            "timestamp": ts,
            "indicators": packed,               # agrupado por base_name (legacy/PresentationManager)
            "indicators_flat": cache,            # plano: output_key → scalar (para deque buffer)
            "events": events,
            "is_warmed_up": ctx.is_warmed_up,
        }

    # ------------------------------------------------------------------
    # Empaquetado por base_name
    # ------------------------------------------------------------------

    
