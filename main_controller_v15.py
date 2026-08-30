# main_controller_v15.py
#
# Orquestador del pipeline de análisis técnico.
#
# Responsabilidades:
#   - Instanciar y conectar los módulos de datos.
#   - Carga histórica inicial (klines + OI + opciones).
#   - Polling periódico de opciones via QTimer — llama a GexCalculator
#     y pasa el resultado al buffer.
#   - Drenaje de la queue del WebSocket y despacho al buffer.
#   - Debouncing del refresco GEX encapsulado en _should_refresh_gex().
#   - Consultar PresentationManager y pasar datos + parámetros visuales
#     al canvas — sin tomar decisiones de presentación propias.
#   - Delegar gestión de rangos Y, zoom y ciclo de vida visual al
#     UIController (mediador pasivo).
#
# Lo que NO hace:
#   - No calcula GEX — eso es GexCalculator.
#   - No persiste estado de opciones — eso es MarketBufferHandler.
#   - No decide colores, estilos ni posiciones — eso es PresentationManager.
#   - No gestiona rangos Y ni zoom — eso es UIController.
#
# Arquitectura de I/O:
#   UN solo _rest_loop asyncio vive en thread daemon durante toda la sesión.
#   UN solo BybitHttpClient y UNA sola ClientSession aiohttp ligados a ese
#   loop. Carga inicial y polling comparten esa sesión. stop() cierra la
#   sesión y el loop de forma ordenada sin deadlocks ni sesiones huérfanas.

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import json
import signal
import threading
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

# ---------------------------------------------------------------------------
# Path resolver — módulos de ingesta local viven en ingesta_local/
# ---------------------------------------------------------------------------
import pathlib as _pathlib
import sys as _sys
_INGESTA_LOCAL = str(_pathlib.Path(__file__).resolve().parent / "ingesta_local")
if _INGESTA_LOCAL not in _sys.path:
    _sys.path.insert(0, _INGESTA_LOCAL)

from bybit_http_client     import BybitHttpClient
from perp_kline_client     import PerpKlineClient
from perp_oi_client        import PerpOIClient
from options_client        import OptionsClient
from gex_calculator        import GexCalculator
from market_buffer_handler import MarketBufferHandler
from ws_ticker_client      import WebSocketTickerClient
from orderbook_client      import OrderbookClient
from presentation_manager import PresentationManager, LiquidityVisualizationManager
from ui_controller         import UIController
from rest_scheduler        import RestScheduler

# --- Motor de cálculo de indicadores técnicos ---
from calculus_manager import (
    create_calculus_engine,
    CMD_WARMUP, CMD_RESET, CMD_SHUTDOWN,
    MSG_TICK, MSG_ACK, MSG_WARMUP_ACK,
    MSG_EVENT, MSG_PID_REPORT, MSG_ERROR, MSG_STATUS,
    MSG_SYNC,
)

# --- Pipeline nuevo de presentación de indicadores ---
from indicator_mapper import IndicatorMapper
from indicator_presentation import IndicatorPresentationManager

# --- Legacy: mantenido para red de seguridad en _render_indicator_buffer ---
from indicator_presentation import (
    load_indicators_config,
    build_indicator_render_packets,
)

import multiprocessing

# --- Pool de warmup paralelo de indicadores ---
from warmup_pool_manager import WarmupPoolManager
import numpy as np
from collections import deque

# --- Pipeline de ingesta local ---
from local_preloader  import LocalPreloader
from persistence_manager import PersistenceManager

log = logging.getLogger("main_controller")

# ---------------------------------------------------------------------------
# Configuración por defecto
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "candle_limit":                  2000,
    "oi_interval":                   "5min",
    "gex_refresh_interval_ms":       2000,
    "options_poll_interval_ms":      10000,
    "ws_queue_drain_interval_ms":    100,
    "orderbook_poll_interval_ms":    5000,   # polling REST orderbook cada 5s
    "orderbook_depth_limit":         1000,     # niveles por lado (bid/ask)
    "buffer_size":                   3000,
    "price_debounce_pct":            0.001,  # 0.1% — umbral mínimo de cambio
                                             # de precio para disparar refresco GEX
}





# ---------------------------------------------------------------------------
# MainController
# ---------------------------------------------------------------------------

class MainController(QObject):
    """
    Orquestador principal del pipeline de análisis técnico.

    Ciclo de vida de I/O:
      - UN solo BybitHttpClient vive dentro del _rest_loop durante toda
        la sesión.
      - UN solo ClientSession aiohttp ligado a ese loop.
      - Carga inicial + polling usan la misma sesión.
      - stop() cierra todo de forma ordenada sin futures colgados.

    Gestión visual:
      - UIController actúa como mediador entre este orquestador y el
        Canvas de liquidez. Gestiona rangos Y (1.5% padding), zoom
        (soberanía del usuario), y ciclo de vida de widgets.
      - Este orquestador NO emite comandos de escala (setYRange/setXRange)
        ni gestiona la visibilidad de paneles. Solo pasa datos.

    Instanciación binaria:
      - FULL (has_options=True): LiquidityContainer con tríada completa.
      - LIGHT (has_options=False): LiquidityDistributionCanvas solitario.
      - El caller instancia el widget correcto y lo pasa al constructor.
    """

    # Señales hacia la UI
    sig_status        = pyqtSignal(str)       # mensajes de estado
    sig_loading_done  = pyqtSignal()          # carga inicial completada
    sig_error         = pyqtSignal(str)       # errores críticos
    candle_ready      = pyqtSignal(dict)      # vela lista para observadores externos
    sig_token_ready   = pyqtSignal(str)       # token completamente listo (buffer + indicadores + WS)
    sig_token_progress = pyqtSignal(str, int) # token en cálculo: (token_id, pct 0-100)

    def __init__(
        self,
        canvas,            # PyQtGraphCanvas
        liquidity_canvas,  # LiquidityContainer (FULL) o LiquidityDistributionCanvas (LIGHT)
        perp_oi_widget,    # PerpOIWidget
        symbol:    str,
        base_coin: str,
        timeframe: str,
        config: Dict[str, Any] = None,
        has_options: bool = True,
        oscillator_window = None,  # OscillatorWindow — inyectado desde GexAnalyzer
        parent=None,
    ):
        super().__init__(parent)

        self._canvas           = canvas
        self._liquidity_canvas = liquidity_canvas
        self._perp_oi_widget   = perp_oi_widget
        self._symbol           = symbol.upper()
        self._base_coin        = base_coin.upper()
        self._timeframe        = str(timeframe)
        self._oscillator_window = oscillator_window  # OscillatorWindow | None

        self._cfg = dict(_DEFAULT_CONFIG)
        if config:
            self._cfg.update(config)

        # --- Módulos de datos ---
        # UN SOLO BybitHttpClient vive durante toda la sesión dentro del
        # _rest_loop dedicado. Todos los módulos que hacen I/O (kline, oi,
        # options, orderbook) lo comparten. Sin handshakes extra, sin sesiones huérfanas.
        self._http:      Optional[BybitHttpClient]     = None
        self._kline:     Optional[PerpKlineClient]     = None
        self._oi:        Optional[PerpOIClient]        = None
        self._options:   Optional[OptionsClient]       = None
        self._orderbook: Optional[OrderbookClient]     = None
        self._gex:       Optional[GexCalculator]       = None
        self._buffer:    Optional[MarketBufferHandler] = None

        # --- Loop REST dedicado ---
        # Vive en thread daemon durante toda la sesión. Aloja la única
        # ClientSession aiohttp del proceso. Carga inicial y polling
        # se encolan contra este loop via run_coroutine_threadsafe.
        self._rest_loop:   Optional[asyncio.AbstractEventLoop] = None
        self._rest_thread: Optional[threading.Thread]          = None

        # WebSocket
        self._ws_queue:           Optional[multiprocessing.Queue] = None
        self._ws_client:          Optional[WebSocketTickerClient] = None
        self._ws_subscribe_queue: Optional[multiprocessing.Queue] = None  # suscripciones dinámicas
        self._ws_ready:           bool                            = False  # WS activo y conectado

        # Tokens completamente listos (buffer + indicadores + WS suscripto).
        # Separado de _ready_tokens (solo carga de velas) para que la UI
        # refleje el estado real de disponibilidad operativa.
        self._fully_ready_tokens: set = set()

        # Pool de warmup paralelo — None si se usa el flujo clásico (CM directo)
        self._warmup_pool:        Optional[WarmupPoolManager] = None
        self._using_warmup_pool:  bool                        = False
        # True si _load_history_async inyectó tensor desde .iend antes del pool.
        # _start_warmup_pool lo usa para enviar skip_series=True al CM.
        self._sidecar_preloaded:  bool                        = False
        # Velas REST appendeadas al .ender durante el gap fill del token primario.
        # _on_pool_primary_ready lo usa para appendear las filas faltantes al .iend.
        self._primary_gap_filled: int                         = 0
        # True cuando el sidecar cubre parte del buffer (no todo).
        # _start_warmup_pool usa CM clásico en vez de WarmupPool.
        self._partial_sidecar:   bool                        = False

        # Timers Qt
        self._timer_ws:        Optional[QTimer] = None  # drenaje de queue WS
        self._timer_options:   Optional[QTimer] = None  # polling REST opciones
        self._timer_gex:       Optional[QTimer] = None  # refresco visual GEX
        self._timer_rotation:  Optional[QTimer] = None  # rotación de estructurales
        self._timer_orderbook: Optional[QTimer] = None  # polling REST orderbook

        # Presentación
        self._presentation = PresentationManager()

        # Estado interno
        self._last_price:      float = 0.0
        self._last_gex_price:  float = 0.0
        self._is_running:      bool  = False
        self._last_gex_series: Dict  = {}
        self._last_gex_keys:   set   = set()   # keys del último render_gex — detecta expirados
        self._rotation_index:  int   = 0
        self._has_options:     bool  = has_options  # False para tokens LIGHT (sin opciones)

        # --- UIController: mediador de rangos Y, zoom y ciclo de vida ---
        # El MainController pasa datos al canvas; el UIController gestiona
        # rangos, zoom y destrucción/recreación de widgets.
        self._ui_controller = UIController(
            parent_pid=os.getpid(),
            parent=self,
        )
        self._ui_controller.attach_canvas(
            self._liquidity_canvas,
            has_options=self._has_options,
        )

        # Cache de orderbook — niveles de profundidad para el canvas 5.2
        # Formato: list[tuple[float, float]] → [(price, usdt_notional), ...]
        self._last_orderbook_perp: List = []
        self._last_orderbook_spot: List = []

        # --- Debouncing del refresco GEX ---
        # _force_refresh     : seteado por timer periódico, rotación, polling
        # _options_updated   : seteado por _poll_options_async
        # _price_debounce_pct: umbral mínimo de cambio de precio
        #
        # La lógica de debouncing está encapsulada en _should_refresh_gex()
        # para facilitar su futura extracción a una clase autónoma.
        self._force_refresh:      bool  = False
        self._options_updated:    bool  = False
        self._price_debounce_pct: float = float(self._cfg["price_debounce_pct"])

        # --- Centrado inicial diferido ---
        # Los rangos Y se aplican DESPUÉS del primer update_all exitoso,
        # no en load_history_sync (donde el canvas aún no tiene datos).
        self._initial_ranges_pending: bool = True

        # --- Centinela anti-zombis ---
        self._child_pids: List[int] = []
        atexit.register(self._kill_children)

        # --- Motor de cálculo de indicadores (CalculusManager) ---
        self._calculus_engine:      Optional[Any] = None
        self._calculus_pipe:        Optional[Any] = None
        self._calculus_control_q:   Optional[Any] = None
        self._calculus_stop_event:  Optional[Any] = None
        self._timer_calculus:       Optional[QTimer] = None
        self._calculus_engine_pid:  int = 0
        self._indicators_config:    Dict = {}

        # --- Tokens con buffer completamente cargado ---
        # Se popula cuando _load_secondary_tokens recibe el callback on_result.
        # set_active_token verifica este set antes de hacer fast switch.
        # El token principal se agrega cuando termina _load_history_async.
        self._ready_tokens: set = set()

        # --- Tokens con contexto creado en el CalculusManager ---
        # Evita enviar CMD_WARMUP duplicados para el mismo token.
        self._known_calculus_tokens: set = set()

        # --- Flags de modo LogicMaster ---
        # Modo 1 (calculadora pura): ambos False — solo indicadores, sin señales.
        # Modo 2 (asistente analítico): lm_signal_enabled=True — pinta señales
        #         en canvas pero no sincroniza estado de posición real.
        # Modo 3 (copiloto/automático): lm_copilot_enabled=True — sincroniza
        #         la posición real con el LM. lm_auto_trade_enabled=True
        #         además reenvía señales del LM al pipeline de órdenes.
        # Controlados desde LM_GEX_J via set_lm_mode() o toggles de UI.
        self._lm_signal_enabled:     bool = False
        self._lm_copilot_enabled:    bool = False
        self._lm_auto_trade_enabled: bool = False

        # --- Modos operativos por token ---
        # Cargados desde token_modes.json.
        # Formato: {SYMBOL_UPPER: "calc" | "signal" | "auto"}
        # Se recargan en caliente con reload_token_modes().
        self._token_modes: Dict[str, str] = {}
        self._token_modes_path: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "token_modes.json"
        )
        self._load_token_modes()

        # --- Pipeline nuevo de presentación de indicadores ---
        # Instanciados directamente en __init__ (resilientes ante JSON ausente).
        self._indicator_mapper = IndicatorMapper(map_path="indicators_map.json")
        self._presentation_manager = IndicatorPresentationManager(
            config_path="indicators_config.json",
        )

        # --- Buffer persistente de indicadores ---
        self._indicator_buffer:   Dict = {}
        self._warmup_complete:    bool = False
        self._last_indicator_ts:  int  = 0       # último ack_ts procesado (token activo)
        self._indicator_ts_by_token: Dict[str, int] = {}  # último ack_ts por token (todos)
        self._key_to_base:        Dict = {}       # output_key → base_name

        # --- Estado del timer de refresco gráfico unificado ---
        self._last_rendered_price:     float = 0.0
        self._last_rendered_ohlcv:     Dict  = {}
        self._last_rendered_countdown: str   = ""
        self._last_rendered_ts:        int   = 0
        self._timer_canvas_refresh:    Optional[QTimer] = None

        # --- Cola de ticks pendientes durante reset visual ---
        # Acumula ticks que llegan mientras _is_running=False (ventana
        # de recreación del canvas). Se drena en mark_canvas_ready()
        # filtrando por timestamp para evitar duplicar velas ya pintadas
        # por plot_full_chart. Solo acumula ticks del símbolo activo.
        self._pending_ticks: List = []

        # --- Perfiles analíticos multi-token ---
        # Cargados desde session_config.json al inicializar.
        # Formato: {symbol_upper: profile_dir_abs}
        # Permite que _warmup_calculus_engine pase el profile_dir correcto
        # al CalcusManager para cada token, y que set_active_token
        # cambie el token activo sin recargar datos desde la API.
        self._active_profiles: Dict[str, str] = {}
        self._session_config_path: str = self._cfg.get(
            "session_config_path",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_config.json")
        )
        self._load_session_profiles()

        # --- Pipeline de ingesta local ---
        # _local_paths: {symbol: {"ender": str|None, "sidecar": str|None}}
        # Poblado por _resolve_local_data_paths() en la primera llamada.
        self._local_paths: Dict[str, Dict[str, Optional[str]]] = {}
        # PersistenceManager: appendea .ender + .iend vela a vela.
        # Instanciado en _data_initialize(); tokens registrados cuando
        # el buffer confirma su carga inicial.
        self._persistence_manager: Optional[PersistenceManager] = None

        print(f"[MainController] Inicializado. {self._symbol} TF={self._timeframe}")
        log.info("MainController symbol=%s tf=%s", self._symbol, self._timeframe)

    # ==================================================================
    # Puntos de entrada públicos
    # ==================================================================

    def load_history_sync(self) -> Optional[pd.DataFrame]:
        """
        Carga histórica bloqueante. Se invoca desde el DataWorker (QThread).

        Delega toda la I/O al _rest_loop dedicado via run_coroutine_threadsafe.
        La ClientSession aiohttp creada en _init_rest_clients_async queda
        ligada a ese loop — todas las peticiones HTTP deben correr ahí.

        No es async def porque no hay nada que awaitar desde el caller:
        future.result() bloquea el thread del DataWorker hasta que
        _load_history_async() termina, sin bloquear el Qt main thread.

        Retorna el DataFrame listo para plot_full_chart(), o None si falló.
        No inicializa el canvas ni arranca el WebSocket.
        """
        self.sig_status.emit(
            f"[MainController] Cargando {self._symbol}..."
        )

        if not self._rest_loop or not self._rest_loop.is_running():
            self._data_initialize()
            self._start_rest_loop()

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._load_history_async(),
                self._rest_loop,
            )
            df = future.result(timeout=120)   # carga histórica + opciones

            # Arrancar motor de cálculo de indicadores
            self._init_calculus_engine()

            # Los rangos Y iniciales se aplican de forma diferida:
            # después del primer update_all/orderbook exitoso
            # (ver _visual_refresh_gex y _visual_refresh_orderbook).
            # Esto garantiza que el canvas tenga datos dibujados
            # ANTES de fijar las escalas.

            return df
        except Exception as e:
            msg = f"[MainController] Error en carga síncrona: {e}"
            print(msg)
            log.error(msg, exc_info=True)
            self.sig_error.emit(msg)
            return None

    def mark_canvas_ready(self) -> None:
        """
        Activa el procesamiento de mensajes del WS y drena la cola de
        ticks pendientes acumulados durante la recreación del canvas.

        Llamar después de init_chart() + plot_full_chart().

        Secuencia:
          1. Activar _is_running.
          2. Conectar señal de línea vertical.
          3. Drenar _pending_ticks filtrando por timestamp:
             solo procesa ticks estrictamente posteriores a la última
             vela ya pintada por plot_full_chart (_last_rendered_ts),
             evitando duplicar velas del bloque histórico.
        """
        self._is_running = True
        # Conectar señal de línea vertical → OscillatorWindow
        self._connect_vl_signal()

        # Drenar ticks acumulados durante la ventana de reset visual.
        # El filtro ts >= _last_rendered_ts garantiza que solo se
        # procesen ticks de la vela en formación o de velas nuevas —
        # nunca velas que plot_full_chart ya pintó en el canvas.
        if self._pending_ticks:
            pending = list(self._pending_ticks)
            self._pending_ticks.clear()
            drained = 0
            for (sym, vela) in pending:
                ts = int(vela.get("timestamp", 0))
                if ts >= self._last_rendered_ts:
                    self._on_candle(sym, vela)
                    drained += 1
            if drained:
                print(
                    f"[MainController] mark_canvas_ready: "
                    f"{drained} tick(s) pendiente(s) drenado(s) "
                    f"(last_rendered_ts={self._last_rendered_ts})."
                )

        print("[MainController] Canvas marcado como listo.")

    def get_base_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Snapshot del buffer de velas vivo como DataFrame, para consumidores
        externos (ej. ResampleWindow).

        Devuelve las velas actuales (timestamp/OHLCV) ya con la columna
        'open_time' que el canvas y el resampleo necesitan. None si el
        buffer aún no tiene datos.
        """
        if not self._buffer:
            return None
        candles = self._buffer.get_candles(self._symbol)
        if not candles:
            return None
        return self._data_candles_to_dataframe(candles)

    def refresh_gex_visual(self) -> None:
        """Refresco visual del GEX — expuesto para el QTimer de la PoC."""
        self._visual_refresh_gex()

    def start_websocket(self) -> None:
        """
        Phase A del pipeline de datos en vivo.

        Arranca el CalculusManager (timer de drenaje + warmup del token
        primario). El WebSocket y los timers de mercado (Phase B) se
        inician automáticamente desde _on_warmup_ack cuando el warmup
        del token primario termina, garantizando que no haya ticks
        entrando mientras el CM procesa el buffer histórico.

        PRECONDICIÓN: _rest_loop y los clientes REST ya están vivos
        (arrancados por load_history_sync() previamente).
        """
        self._start_calculus_timer()
        self._start_warmup_pool()
        print(
            "[MainController] Phase A iniciada — "
            "WS arrancará al completarse el warmup."
        )

    def _do_start_websocket(self) -> None:
        """
        Phase B del pipeline de datos en vivo.

        Crea el WebSocket con el símbolo primario únicamente y arranca
        todos los timers de mercado. Se invoca desde _on_warmup_ack
        cuando el warmup del token activo está completo, asegurando
        que el CM ya no está bloqueado en el pipe cuando empiezan a
        llegar ticks.

        Los tokens secundarios se suscriben dinámicamente via
        _subscribe_token_to_ws conforme terminan su propio warmup.
        """
        if self._ws_ready:
            return  # idempotente — evita doble arranque

        # Queue de suscripciones dinámicas compartida con WebSocketTickerClient.
        # Permite agregar nuevos símbolos sin reconectar.
        self._ws_subscribe_queue = multiprocessing.Queue()

        self._ws_queue  = multiprocessing.Queue(maxsize=2000)
        self._ws_client = WebSocketTickerClient(
            symbols=[self._symbol],          # solo el token primario al arrancar
            data_queue=self._ws_queue,
            subscribe_queue=self._ws_subscribe_queue,
            config={"kline_interval": self._timeframe},
        )
        self._ws_client.start()

        if hasattr(self._ws_client, 'pid') and self._ws_client.pid:
            self._register_child_pid(self._ws_client.pid)

        # Timer — drenaje de queue del WS → buffer
        self._timer_ws = QTimer(self)
        self._timer_ws.setInterval(self._cfg["ws_queue_drain_interval_ms"])
        self._timer_ws.timeout.connect(self._drain_ws_queue)
        self._timer_ws.start()

        if self._has_options:
            self._timer_options = QTimer(self)
            self._timer_options.setInterval(self._cfg["options_poll_interval_ms"])
            self._timer_options.timeout.connect(self._poll_options)
            self._timer_options.start()

            self._timer_gex = QTimer(self)
            self._timer_gex.setInterval(self._cfg["gex_refresh_interval_ms"])
            self._timer_gex.timeout.connect(self._periodic_refresh)
            self._timer_gex.start()

            rotation_ms = self._presentation.ROTATION_INTERVAL_MS
            self._timer_rotation = QTimer(self)
            self._timer_rotation.setInterval(rotation_ms)
            self._timer_rotation.timeout.connect(self._advance_rotation)
            self._timer_rotation.start()

        self._timer_orderbook = QTimer(self)
        self._timer_orderbook.setInterval(self._cfg["orderbook_poll_interval_ms"])
        self._timer_orderbook.timeout.connect(self._poll_orderbook)
        self._timer_orderbook.start()

        self._timer_canvas_refresh = QTimer(self)
        self._timer_canvas_refresh.setInterval(1000)
        self._timer_canvas_refresh.timeout.connect(self._trigger_canvas_refresh)
        self._timer_canvas_refresh.start()

        self._ws_ready = True
        self._fully_ready_tokens.add(self._symbol)
        self.sig_token_ready.emit(self._symbol)

        log.info("[MainController] Phase B iniciada — WS activo para %s.", self._symbol)
        print(f"[MainController] WebSocket y timers iniciados para {self._symbol}.")

        # Disparar carga de tokens secundarios ahora que el WS primario
        # está conectado. El retraso de 2s permite que Bybit confirme
        # la suscripción antes de que el CM empiece los warmups secundarios.
        QTimer.singleShot(2000, self._warmup_all_tokens)

    def _subscribe_token_to_ws(self, symbol: str) -> None:
        """
        Agrega un token a la suscripción del WS activo sin reconectar.

        Usa _ws_subscribe_queue para enviar el mensaje al proceso
        WebSocketTickerClient. Si el WS aún no está activo, no hace nada
        (los tokens secundarios solo llegan cuando el WS ya corre).
        """
        if not self._ws_ready or not self._ws_subscribe_queue:
            return
        try:
            tf   = str(self._timeframe)
            args = [f"kline.{tf}.{symbol}", f"tickers.{symbol}"]
            self._ws_subscribe_queue.put_nowait({"op": "subscribe", "args": args})
            log.info("[WS] Suscripción dinámica enviada para %s.", symbol)
            print(f"[MainController] WS: suscripto a {symbol}.")
        except Exception as e:
            log.warning("[WS] Error encolando suscripción para %s: %s", symbol, e)

    def stop(self) -> None:
        """
        Detiene todos los procesos, timers y conexiones de forma ordenada.

        Orden crítico:
          1. Flags: impedir nuevos ciclos de refresco.
          2. Timers Qt: stop (no bloqueante).
          3. WebSocket: cerrar y join con timeout corto.
          4. Shutdown asíncrono del _rest_loop:
             a) Encolar _shutdown_rest_async que cierra la ClientSession.
             b) Esperar ese future con timeout.
             c) call_soon_threadsafe(loop.stop) desde el caller.
             d) join del thread del loop.
        """
        self._is_running = False

        # 1. Detener timers Qt (no bloqueante)
        for timer in (
            self._timer_ws,
            self._timer_options,
            self._timer_gex,
            self._timer_rotation,
            self._timer_orderbook,
            self._timer_calculus,
            self._timer_canvas_refresh,
        ):
            if timer:
                try:
                    timer.stop()
                except Exception as e:
                    log.debug("Error deteniendo timer: %s", e)

        # 2. Cerrar WebSocket
        if self._ws_client:
            try:
                self._ws_client.close()
                self._ws_client.join(timeout=3)
            except Exception as e:
                log.warning("Error cerrando WebSocket: %s", e)

        # 2b. Cerrar subscribe_queue del WS
        if self._ws_subscribe_queue:
            try:
                self._ws_subscribe_queue.cancel_join_thread()
                self._ws_subscribe_queue.close()
            except Exception:
                pass
            self._ws_subscribe_queue = None
        self._ws_ready = False

        # 2c. Detener WarmupPool si está corriendo
        if self._warmup_pool and self._warmup_pool.isRunning():
            try:
                self._warmup_pool.stop()
                self._warmup_pool.wait(2000)   # esperar máx 2s
            except Exception as e:
                log.debug("Error deteniendo WarmupPool: %s", e)
            self._warmup_pool = None

        # 2d. Cerrar handles de persistencia incremental
        if self._persistence_manager:
            try:
                self._persistence_manager.close_all()
            except Exception as e:
                log.warning("Error cerrando PersistenceManager: %s", e)
            self._persistence_manager = None

        # 3. Limpieza del canvas via UIController
        #    destroy_canvas() hace close_cleanup + clear sin programar
        #    el TTL suicida de hard_shutdown. hard_shutdown se reserva
        #    para el Centinela (muerte del proceso padre) y closeEvent.
        if self._ui_controller:
            try:
                self._ui_controller.destroy_canvas()
            except Exception as e:
                log.warning("Error en UIController.destroy_canvas: %s", e)

        # 3b. Limpiar datos del canvas de osciladores sin destruir el widget.
        #     El widget vive en el QSplitter del LeftContainer — solo se
        #     borran las curvas para que el próximo token/reset parta limpio.
        if self._oscillator_window:
            try:
                self._oscillator_window.clear_plots()
            except Exception as e:
                log.warning("Error en OscillatorWindow.clear_plots: %s", e)

        # 4. Shutdown del motor de cálculo de indicadores
        self._shutdown_calculus_engine()

        # 5. Shutdown del loop REST y su sesión HTTP
        self._shutdown_rest_loop()

        # 6. Matar procesos hijos que sigan vivos
        self._kill_children()

        print("[MainController] Detenido.")
        log.info("MainController detenido.")

    # ==================================================================
    # Inicialización y loop REST
    # ==================================================================

    def _data_initialize(self) -> None:
        """
        Instancia los módulos que NO requieren I/O (gex, buffer).
        Los módulos de I/O (http, kline, oi, options) se instancian
        dentro del _rest_loop via _init_rest_clients_async() para que
        la ClientSession aiohttp quede ligada a ese loop.

        GexCalculator solo se instancia en modo FULL (has_options=True).
        """
        if self._has_options:
            self._gex = GexCalculator()

        self._buffer = MarketBufferHandler(
            config={"buffer_size": self._cfg["buffer_size"]},
        )
        self._buffer.on_candle = self._on_candle
        self._buffer.on_ticker = self._on_ticker

        # Activar el filtro de símbolo del buffer. Sin esto, _active_symbol
        # queda en None y process_message acepta velas/tickers de TODOS los
        # símbolos suscritos al WS — mezclando BTC, ADA, SOL en el mismo
        # canvas (vela fantasma, zoom roto). El WS multi-token suscribe a
        # todos los tokens activos, así que el filtro es obligatorio.
        self._buffer.set_symbol(self._symbol)

        # Gestor de persistencia incremental (.ender + .iend).
        # register_token() se llama para cada token después de que
        # _load_token() confirma que los archivos locales existen.
        self._persistence_manager = PersistenceManager()

    def _start_rest_loop(self) -> None:
        """
        Arranca el loop asyncio dedicado a TODO el tráfico HTTP del proceso.

        El loop vive en un thread daemon separado durante toda la sesión.
        La ClientSession aiohttp se crea una sola vez dentro de ese loop
        y se reutiliza: carga inicial, polling de opciones, futuras
        peticiones REST. Sin loops efímeros, sin sesiones huérfanas.

        Escalabilidad: cuando se agreguen múltiples tokens, este loop
        puede iterar secuencialmente sobre todos ellos en cada ciclo
        con la misma sesión abierta.
        """
        self._rest_loop = asyncio.new_event_loop()

        def _run_loop():
            asyncio.set_event_loop(self._rest_loop)
            self._rest_loop.run_forever()

        self._rest_thread = threading.Thread(
            target=_run_loop,
            daemon=True,
            name="rest-loop",
        )
        self._rest_thread.start()

        # Inicializar todos los clientes dentro del loop dedicado para que
        # la ClientSession aiohttp quede ligada a ese loop
        future = asyncio.run_coroutine_threadsafe(
            self._init_rest_clients_async(),
            self._rest_loop,
        )
        try:
            future.result(timeout=10)
        except Exception as e:
            log.error("Error inicializando clientes REST: %s", e)
            print(f"[MainController] Error inicializando loop REST: {e}")

        print("[MainController] Loop REST dedicado iniciado.")

    async def _init_rest_clients_async(self) -> None:
        """
        Instancia el BybitHttpClient y el RestScheduler dentro del
        _rest_loop dedicado. Los clientes de dominio (kline, oi, options,
        orderbook) son ahora gestionados por el scheduler — el controller
        ya no los instancia directamente.

        Se mantienen self._kline, self._oi, self._options, self._orderbook
        para compatibilidad con el polling existente (migración incremental).
        """
        self._http = BybitHttpClient()
        await self._http._ensure_session()

        # Instanciar el scheduler con el http compartido
        self._scheduler = RestScheduler(
            http=self._http,
            timeframe=self._timeframe,
            delay_between_tasks_s=0.3,
        )
        self._scheduler.start(self._rest_loop)

        # Clientes legacy — para el polling hasta que se migre al scheduler
        self._kline = PerpKlineClient(
            self._http, self._symbol, self._timeframe,
        )
        self._oi = PerpOIClient(
            self._http, self._symbol,
            oi_interval=self._cfg.get("oi_interval", "1h"),
        )
        if self._has_options:
            self._options = OptionsClient(self._http, self._base_coin)
        self._orderbook = OrderbookClient(self._http, self._base_coin)

        log.info(
            "Clientes REST y RestScheduler inicializados en loop propio."
        )

    async def _load_history_async(self) -> Optional[pd.DataFrame]:
        """
        Carga el buffer histórico del token principal.

        Delega a _load_token() que intenta primero la ingesta local
        (.ender + gap fill REST) y cae al REST puro si no hay archivo local.

        Si el token tiene tensor_dict (sidecar compatible), lo inyecta
        en el buffer para que el WarmupPool haga solo el mini-warmup.
        """
        self.sig_status.emit(f"Cargando {self._symbol}...")

        # _load_token bloquea — se llama desde un future en el _rest_loop
        # pero internamente usa threading.Event, no asyncio.Event.
        # Corremos en un executor para no bloquear el loop.
        loop = asyncio.get_event_loop()
        token_result = await loop.run_in_executor(
            None, self._load_token, self._symbol
        )

        if not token_result:
            self.sig_error.emit(f"No se pudieron cargar velas para {self._symbol}.")
            return None

        candles     = token_result["candles"]
        tensor_dict = token_result.get("tensor_dict")

        # Guardar el gap fill para que _on_pool_primary_ready pueda
        # appendear las filas faltantes al .iend después del cómputo.
        self._primary_gap_filled = token_result.get("gap_filled", 0)

        # OI snapshot
        oi_now = await self._oi.fetch_oi_snapshot()

        # Opciones — solo en modo FULL
        if self._has_options and self._options and self._gex:
            self.sig_status.emit("Descargando snapshot de opciones...")
            tickers = await self._options.fetch_options_snapshot()
            if tickers:
                series = self._gex.calculate_series(tickers)
                self._last_gex_series = series
                self._buffer.update_options(
                    series,
                    expiry_hour_utc=self._gex._expiry_hour_utc,
                )

        self._buffer.load_initial_candles(self._symbol, candles)
        self._last_price     = float(candles[-1].get("close", 0.0))
        self._last_gex_price = 0.0

        # Si hay tensor del sidecar, inyectarlo en el buffer.
        # Cuando _load_token retorna tensor_dict != None, la serie está
        # completa — gap fill + cálculo + persistencia ya ocurrieron.
        if tensor_dict is not None:
            self._buffer.load_indicator_series(self._symbol, tensor_dict)
            buffer_size = self._cfg.get("buffer_size", 3000)
            self._indicator_buffer = {}
            for output_key, arr in tensor_dict.items():
                d = deque(maxlen=buffer_size)
                d.extend(arr.tolist() if hasattr(arr, "tolist") else arr)
                self._indicator_buffer[output_key] = d
            self._sidecar_preloaded = True
            log.info(
                "[LocalIngesta] Sidecar inyectado: %s — %d keys, %d filas.",
                self._symbol, len(tensor_dict),
                len(next(iter(tensor_dict.values()))),
            )
        else:
            self._sidecar_preloaded = False

        self._ready_tokens.add(self._symbol)

        if oi_now:
            self._buffer._last_ticker.setdefault(self._symbol, {})
            self._buffer._last_ticker[self._symbol]["openInterest"] = oi_now

        df = self._data_candles_to_dataframe(candles)
        self.sig_status.emit("Carga completada.")
        return df

    async def _load_secondary_tokens(self) -> None:
        """
        Carga los buffers de los tokens secundarios usando el mismo
        patrón que el token primario: local primero, REST como fallback.

        Corre dentro del _rest_loop. Itera secuencialmente — garantiza
        rate limit global y evita solapamiento de peticiones REST.

        Para cada token secundario:
          1. Llama _load_token(symbol) en executor (no bloquea el loop).
          2. load_initial_candles() en el buffer.
          3. Si hay tensor_dict (sidecar): load_indicator_series() + mini CM.
          4. Si no: submit_token() al WarmupPool para cálculo completo.
        """
        if not self._buffer:
            return

        secondary = [
            sym for sym in self._active_profiles
            if sym != self._symbol
        ]

        if not secondary:
            log.info("[LocalIngesta] No hay tokens secundarios para cargar.")
            return

        log.info(
            "[LocalIngesta] Cargando %d tokens secundarios: %s",
            len(secondary), secondary,
        )
        print(
            f"[MainController] Carga secundaria: {len(secondary)} tokens — {secondary}"
        )

        loop = asyncio.get_event_loop()

        for symbol in secondary:
            token_result = await loop.run_in_executor(
                None, self._load_token, symbol
            )

            if not token_result:
                log.warning(
                    "[LocalIngesta] Fallo cargando %s — omitiendo.", symbol
                )
                print(f"[MainController] Error cargando {symbol}: sin resultado.")
                continue

            candles     = token_result["candles"]
            tensor_dict = token_result.get("tensor_dict")

            self._buffer.load_initial_candles(symbol, candles)
            self._ready_tokens.add(symbol)

            if tensor_dict is not None:
                # Sidecar disponible: inyectar tensor y mini-warmup al CM
                self._buffer.load_indicator_series(symbol, tensor_dict)
                self._indicator_ts_by_token[symbol] = (
                    int(candles[-1]["timestamp"]) if candles else 0
                )
                if self._calculus_control_q and symbol not in self._known_calculus_tokens:
                    # Obtener preheating_min del perfil del token
                    profile_dir = self._active_profiles.get(symbol, "")
                    preheating_min = 200
                    if profile_dir:
                        lm_json = os.path.join(profile_dir, "logicMaster.json")
                        try:
                            with open(lm_json, "r", encoding="utf-8") as f:
                                lm_cfg = json.load(f)
                            preheating_min = int(lm_cfg.get("preheating_size", 200))
                        except Exception:
                            pass
                    mini = candles[-preheating_min:]
                    self._calculus_control_q.put({
                        "type":        CMD_WARMUP,
                        "token_id":    symbol,
                        "candles":     mini,
                        "skip_series": True,
                    })
                    self._known_calculus_tokens.add(symbol)
                    log.info(
                        "[LocalIngesta] Mini-warmup CM secundario (sidecar): "
                        "%s (%d velas).", symbol, len(mini),
                    )
            elif self._using_warmup_pool and self._warmup_pool:
                # Sin sidecar: WarmupPool calcula tensor completo
                lm_module = self._resolve_lm_module_path(symbol)
                lm_dir    = self._active_profiles.get(symbol, os.getcwd())
                self._warmup_pool.submit_token(
                    symbol, candles, lm_module, lm_dir
                )
                self.sig_token_progress.emit(symbol, 0)
                log.info(
                    "[WarmupPool] Token secundario encolado: %s (%d velas).",
                    symbol, len(candles),
                )
            elif (self._calculus_control_q
                    and symbol not in self._known_calculus_tokens):
                # Flujo clásico (pool no activo)
                self._calculus_control_q.put({
                    "type":     CMD_WARMUP,
                    "token_id": symbol,
                    "candles":  candles,
                })
                self._known_calculus_tokens.add(symbol)
                log.info(
                    "[Calculus] Warmup secundario (clásico): %d velas para %s.",
                    len(candles), symbol,
                )

            print(
                f"[MainController] Buffer cargado: {symbol} "
                f"({len(candles)} velas, "
                f"fuente={token_result['source']}, "
                f"sidecar={'sí' if tensor_dict else 'no'}) — fast switch habilitado"
            )

        log.info("[LocalIngesta] Carga de tokens secundarios completa.")
        print("[MainController] Carga de tokens secundarios completa.")

    def _shutdown_rest_loop(self) -> None:
        """
        Cierre ordenado del _rest_loop en 3 fases.

        Separar la corrutina de cierre (que cierra el ClientSession)
        del stop del loop evita el deadlock que ocurre si el propio
        loop llama loop.stop() antes de que future.result() retorne.
        """
        if not self._rest_loop:
            return

        # Fase 1: cerrar ClientSession dentro del loop
        if self._rest_loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._shutdown_rest_async(),
                    self._rest_loop,
                )
                future.result(timeout=5)
            except Exception as e:
                log.warning("Error cerrando ClientSession: %s", e)

            # Fase 2: pedir al loop que se detenga desde afuera
            try:
                self._rest_loop.call_soon_threadsafe(self._rest_loop.stop)
            except Exception as e:
                log.debug("Error en call_soon_threadsafe(stop): %s", e)

        # Fase 3: esperar al thread del loop
        if self._rest_thread and self._rest_thread.is_alive():
            self._rest_thread.join(timeout=3)
            if self._rest_thread.is_alive():
                log.warning("rest_thread no terminó en 3s.")

    async def _shutdown_rest_async(self) -> None:
        """
        Cierra el RestScheduler y la ClientSession aiohttp desde dentro
        del _rest_loop.
        """
        if self._scheduler:
            try:
                self._scheduler.stop()
            except Exception as e:
                log.warning("Error deteniendo RestScheduler: %s", e)
        if self._http:
            try:
                await self._http.close()
            except Exception as e:
                log.warning("Error en BybitHttpClient.close(): %s", e)

    def _data_candles_to_dataframe(self, candles: List[Dict]) -> pd.DataFrame:
        """
        Convierte list[dict] → pd.DataFrame para plot_full_chart().

        PerpKlineClient normaliza el timestamp de apertura de vela bajo
        la clave "timestamp" (int, ms UTC). El canvas busca la columna
        "open_time", así que aquí se hace el mapeo explícito.
        """
        df = pd.DataFrame(candles)
        ohlcv = df[["open", "high", "low", "close", "volume"]].astype(float)
        # "timestamp" → "open_time" para el canvas (_timestamp_ms)
        ts_col = None
        for key in ("timestamp", "open_time", "startTime"):
            if key in df.columns:
                ts_col = key
                break
        if ts_col is not None:
            ohlcv["open_time"] = pd.to_numeric(df[ts_col], errors="coerce").astype("int64")
        return ohlcv

    def _initialize_y_ranges(self) -> None:
        """
        Inyecta los rangos Y iniciales al UIController tras la carga
        de datos. Se ejecuta una sola vez por sesión/cambio de token.

        Modo FULL: calcula rangos desde strikes del dataset de opciones.
        Modo LIGHT: calcula rangos desde el precio actual (el orderbook
                    aún no ha cargado en este punto del ciclo de vida).

        Después de esta llamada, el eje Y queda bajo soberanía del
        usuario — no hay actualizaciones automáticas de escala.
        """
        if self._has_options and self._buffer and self._gex:
            series = self._buffer.get_options()
            if series:
                oi_perp = self._buffer.get_perp_oi(self._symbol) or 0.0
                dataset = self._gex.build_liquidity_dataset(
                    series,
                    current_price=self._last_price,
                    oi_perp_total=oi_perp,
                )
                self._ui_controller.initialize_ranges(
                    dataset=dataset,
                    current_price=self._last_price,
                )
                return

        # LIGHT o FULL sin opciones disponibles: precio como referencia
        if self._last_price > 0:
            self._ui_controller.initialize_ranges(
                current_price=self._last_price,
            )

    # ==================================================================
    # Loop de drenaje del WebSocket
    # ==================================================================

    def _drain_ws_queue(self) -> None:
        """
        Drena la queue del WS y pasa mensajes al buffer de perpetuos.
        Llamado por _timer_ws cada ws_queue_drain_interval_ms.
        """
        if not self._ws_queue or not self._buffer:
            return
        drained = 0
        while not self._ws_queue.empty() and drained < 50:
            try:
                msg = self._ws_queue.get_nowait()
                self._buffer.process_message(msg)
                drained += 1
            except Exception:
                break

    # ==================================================================
    # Polling REST de opciones
    # ==================================================================

    def _poll_options(self) -> None:
        """
        Llamado por _timer_options cada options_poll_interval_ms.
        Encola la corrutina en el _rest_loop dedicado — no crea threads.
        """
        if not self._rest_loop or not self._rest_loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self._poll_options_async(),
            self._rest_loop,
        )

    async def _poll_options_async(self) -> None:
        """
        Fetch + cálculo + actualización del buffer de opciones.
        Corre dentro del _rest_loop con la sesión persistente.
        """
        if not self._options or not self._gex or not self._buffer:
            return
        try:
            tickers = await self._options.fetch_options_snapshot()
            if not tickers:
                return
            series = self._gex.calculate_series(tickers)
            self._last_gex_series = series
            self._buffer.update_options(
                series,
                expiry_hour_utc=self._gex._expiry_hour_utc,
            )

            # Notificar al controller para forzar refresco visual
            # independiente del movimiento de precio
            self._options_updated = True
            log.debug(
                "Opciones actualizadas: %d contratos en buffer.",
                self._buffer.get_options_count(),
            )

            # Disparar refresco visual inmediato desde el thread Qt.
            # QTimer.singleShot cross-thread: encola el callback en la
            # event queue del thread donde vive el QObject (self, creado
            # en el thread de Qt). Sin esto, el refresco esperaría al
            # próximo tick del _timer_gex (hasta 2s de latencia).
            QTimer.singleShot(0, self._visual_refresh_gex)

        except Exception as e:
            log.warning("Error en polling de opciones: %s", e)

    # ==================================================================
    # Polling REST de orderbook
    # ==================================================================

    def _poll_orderbook(self) -> None:
        """
        Llamado por _timer_orderbook cada orderbook_poll_interval_ms.
        Encola la corrutina en el _rest_loop dedicado — no crea threads.

        Activo SIEMPRE (FULL y LIGHT) para mantener la profundidad de
        volumen activa en el panel derecho del LiquidityContainer.
        """
        if not self._rest_loop or not self._rest_loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self._poll_orderbook_async(),
            self._rest_loop,
        )

    async def _poll_orderbook_async(self) -> None:
        if not self._orderbook:
            return
        try:
            snapshot = await self._orderbook.fetch_snapshot(
                categories=["linear", "spot"],
                limit=self._cfg["orderbook_depth_limit"],
            )
            curves = LiquidityVisualizationManager.build_orderbook_curves(snapshot)
            self._last_orderbook_perp = curves["perp"]
            self._last_orderbook_spot = curves["spot"]
            QTimer.singleShot(0, self._visual_refresh_orderbook)

        except Exception as e:
            log.warning("Error en polling de orderbook: %s", e)

    # ==================================================================
    # Callbacks del buffer → orquestador
    # ==================================================================

    def _on_candle(self, symbol: str, vela: Dict) -> None:
        """
        Compuerta de ingesta de velas del kline WS.

        Dos responsabilidades separadas e independientes:
          1. Despachar el tick al CalculusManager (TODOS los tokens
             conocidos — token activo y secundarios).
          2. Renderizar la vela en el canvas VIA GATEWAY (solo el
             token activo).

        El guard de canvas (symbol != self._symbol) se aplica SOLO
        al bloque de rendering — no al dispatch al CM.
        Antes el guard estaba antes del dispatch, lo que hacía que
        los tokens secundarios nunca recibieran ticks en el CM y
        sus indicadores quedaban congelados en el snapshot del warmup.

        Mientras _is_running=False (ventana de recreación del canvas),
        los ticks del símbolo activo se acumulan en _pending_ticks.
        Los ticks de tokens secundarios igual se despachan al CM.
        """
        sym_upper = symbol.upper() if symbol else ""

        # --- Dispatch al CM para TODOS los tokens con contexto activo ---
        # Condición: _is_running puede ser False (transición de canvas)
        # pero el CM sigue corriendo — los ticks secundarios no deben
        # perderse durante esa ventana.
        if sym_upper in self._known_calculus_tokens:
            self._dispatch_tick_to_calculus(sym_upper, vela)

        # --- Guard de canvas: solo el token activo pasa de aquí ---
        if not self._is_running:
            if sym_upper == self._symbol:
                self._pending_ticks.append((symbol, dict(vela)))
            return
        if self._canvas is None or self._canvas.current_item is None:
            return
        if sym_upper != self._symbol:
            return

        o = float(vela.get("open",   0))
        h = float(vela.get("high",   0))
        l = float(vela.get("low",    0))
        c = float(vela.get("close",  0))
        v = float(vela.get("volume", 0))
        candle_ts = int(vela.get("timestamp", 0))
        confirmed = vela.get("confirmed", False)

        # --- DIAG: imprimir solo en transiciones de vela ---
        if confirmed or candle_ts != self._last_rendered_ts:
            print(
                f"[WS] symbol={symbol} ts={candle_ts} "
                f"rendered_ts={self._last_rendered_ts} "
                f"confirmed={confirmed} o={o:.6f} c={c:.6f}"
            )

        if c > 0:
            self._last_price = c

        # --- Countdown ---
        countdown_str = ""
        try:
            tf_str = str(self._timeframe)
            tf_minutes = 1440 if tf_str.upper() == "D" else int(tf_str)
            tf_ms      = tf_minutes * 60 * 1000
            now_ms     = int(time.time() * 1000)
            remaining_ms = (candle_ts + tf_ms) - now_ms
            if remaining_ms > 0:
                remaining_s  = remaining_ms // 1000
                mins, secs   = divmod(remaining_s, 60)
                countdown_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        except Exception:
            pass

        # --- 1. Render inmediato via gateway ---
        self._gateway_render_candle(
            candle_ts, o, h, l, c, v, countdown_str, _caller="kline",
        )

        # --- Guardar último estado para el countdown timer ---
        self._last_rendered_ohlcv = {
            "o": o, "h": h, "l": l, "c": c, "v": v,
            "timestamp": candle_ts,
        }

        # --- Emitir señal para consumidores externos ---
        self.candle_ready.emit({
            "o": o, "h": h, "l": l, "c": c, "v": v,
            "confirmed": vela.get("confirmed", False),
            "timestamp": candle_ts,
        })

    def _on_ticker(self, symbol: str, ticker: Dict) -> None:
        """El buffer notifica al orquestador cuando llega un ticker."""
        if symbol and symbol.upper() != self._symbol:
            return
        price = ticker.get("lastPrice") or ticker.get("markPrice")
        if price:
            self._last_price = float(price)

    # ==================================================================
    # Visual — actualización de velas
    # ==================================================================

    def _visual_append_candle(self, o, h, l, c, v) -> None:
        try:
            self._canvas.append_new_candle(o, h, l, c, v)
        except Exception as e:
            print(f"[GW] !!ERROR!! append_new_candle: {e}")
            log.warning("Error en append_new_candle: %s", e)

    def _visual_update_candle(self, o, h, l, c, v, countdown_str: str = "") -> None:
        try:
            self._canvas.update_last_candle(o, h, l, c, v, countdown_str=countdown_str)
        except Exception as e:
            print(f"[GW] !!ERROR!! update_last_candle: {e}")
            log.warning("Error en update_last_candle: %s", e)

    # ==================================================================
    # Gateway de renderizado de velas — punto único de escritura
    # ==================================================================

    def _gateway_render_candle(
        self,
        candle_ts: int,
        o: float, h: float, l: float, c: float, v: float,
        countdown_str: str = "",
        _caller: str = "",
    ) -> None:
        """
        Único método autorizado para escribir velas en el canvas.

        Principio de responsabilidad única: NADIE más toca
        append_new_candle ni update_last_candle. Ni el CM, ni el
        ticker, ni ningún timer. Solo este método.

        Decisión basada en _last_rendered_ts (timestamp del open de
        la última vela renderizada en el canvas):

          candle_ts == _last_rendered_ts → update (misma vela)
          candle_ts >  _last_rendered_ts → append (vela nueva)
          candle_ts <  _last_rendered_ts → descartar (dato rezagado)

        Llamado desde _on_candle (kline WS) en cada tick.
        """
        if not self._canvas or self._canvas.current_item is None:
            print(f"[GW] GUARD canvas=None o current_item=None caller={_caller}")
            return

        if candle_ts < self._last_rendered_ts:
            print(
                f"[GW] DISCARD ts={candle_ts} < rendered={self._last_rendered_ts} "
                f"caller={_caller}"
            )
            return

        if candle_ts > self._last_rendered_ts:
            # Vela nueva: desplazar la vela en formación al buffer
            # histórico y abrir una nueva posición en el canvas.
            prev_ts = self._last_rendered_ts
            x_before = self._canvas._x_counter
            self._last_rendered_ts = candle_ts
            self._visual_append_candle(o, h, l, c, v)
            x_after = self._canvas._x_counter
            print(
                f"[GW] APPEND ts={candle_ts} prev_ts={prev_ts} "
                f"x={x_before}→{x_after} "
                f"o={o:.6f} c={c:.6f} caller={_caller}"
            )
        else:
            # Misma vela: actualizar OHLCV in-place.
            self._visual_update_candle(o, h, l, c, v, countdown_str)

    # ==================================================================
    # Refresco del GEX — debouncing + render
    # ==================================================================

    def _should_refresh_gex(self, current_price: float) -> bool:
        """
        Lógica de debouncing del refresco GEX.

        Retorna True si debe ejecutarse el refresco visual, False si saltear.

        Condiciones para refrescar (OR):
          - Force flag seteado (timer periódico, rotación visual,
            snapshot nuevo de opciones).
          - Primer refresco de la sesión (baseline sin inicializar).
          - Cambio de precio relativo mayor o igual al umbral configurado.

        Efectos secundarios al retornar True:
          - Consume los flags (_force_refresh, _options_updated → False).
          - Actualiza el baseline (_last_gex_price = current_price).

        Diseño: toda la lógica y el estado asociado vive en este método
        y en los atributos _force_refresh, _options_updated, _last_gex_price,
        _price_debounce_pct. Para migrarlo a una clase GexRefreshDebouncer
        basta con mover esos cuatro atributos y este método sin cambios.
        """
        # Force flags — prioridad absoluta sobre el umbral de precio
        force = self._force_refresh or self._options_updated
        if force:
            self._force_refresh   = False
            self._options_updated = False
            self._last_gex_price  = current_price
            log.debug("[GexRefresh] Force refresh (flags).")
            return True

        # Primera pasada de la sesión — baseline vacío
        if self._last_gex_price <= 0:
            self._last_gex_price = current_price
            log.debug("[GexRefresh] Primera pasada — baseline inicializado.")
            return True

        # Debounce por variación de precio
        delta_pct = abs(current_price - self._last_gex_price) / self._last_gex_price
        if delta_pct >= self._price_debounce_pct:
            self._last_gex_price = current_price
            log.debug(
                "[GexRefresh] Precio cambió %.4f%% (umbral %.4f%%) — refresco.",
                delta_pct * 100, self._price_debounce_pct * 100,
            )
            return True

        log.debug(
            "[GexRefresh] Salteo — cambio %.4f%% < umbral %.4f%%.",
            delta_pct * 100, self._price_debounce_pct * 100,
        )
        return False

    def _periodic_refresh(self) -> None:
        """
        Callback del _timer_gex (cada gex_refresh_interval_ms).

        Fuerza un refresco aunque el precio no haya cambiado, para que
        el theta decay de los contratos se refleje visualmente con el
        paso del tiempo incluso en mercados planos.

        Invoca ambos paths: perpetuos (siempre) y GEX (solo FULL).
        """
        self._force_refresh = True
        self._visual_refresh_perp()
        self._visual_refresh_gex()

    def _advance_rotation(self) -> None:
        """
        Incrementa el índice de rotación de estructurales.
        Llamado por _timer_rotation cada ROTATION_INTERVAL_MS.
        Fuerza un refresco visual para mostrar el siguiente vencimiento.

        Solo invoca GEX — la rotación es exclusiva de opciones.
        """
        self._rotation_index += 1
        self._force_refresh   = True
        self._visual_refresh_gex()

    def _visual_refresh_perp(self) -> None:
        """
        Actualiza los datos de mercado perpetuo: OI perp + precio.

        Ejecutado SIEMPRE (FULL y LIGHT) desde _on_candle y
        _periodic_refresh. En modo FULL complementa al pipeline
        de opciones. En modo LIGHT es el único path de datos
        hacia el widget de OI y la línea de precio.

        El orderbook se actualiza por _timer_orderbook, no aquí.
        """
        if not self._is_running or not self._buffer:
            return

        price = self._last_price
        if price <= 0:
            price = self._buffer.get_last_price(self._symbol) or 0
        if price <= 0:
            return

        oi_perp = self._buffer.get_perp_oi(self._symbol) or 0.0

        if oi_perp and self._perp_oi_widget:
            self._perp_oi_widget.set_oi(oi_perp)

        # En modo LIGHT, actualizar la línea de precio en cada tick
        # (en FULL, update_all ya lo hace via set_current_price)
        if not self._has_options and self._liquidity_canvas:
            try:
                self._liquidity_canvas.set_current_price(price)
            except Exception:
                pass

    def _visual_refresh_gex(self) -> None:
        """
        Recalcula y renderiza el GEX sobre el último snapshot en buffer.

        Exclusivo del modo FULL (has_options=True).
        En modo LIGHT este método es un no-op completo.

        La lógica de perpetuos/OI perp vive en _visual_refresh_perp(),
        que se ejecuta independientemente del modo.

        Disparado desde:
          - _periodic_refresh (timer de 2s, bypassa debounce)
          - _on_candle (tick WS, aplica debounce)
          - _advance_rotation (rotación visual, bypassa debounce)
          - _poll_options_async (snapshot fresco, bypassa debounce)

        El debouncing está delegado a _should_refresh_gex() — si retorna
        False, la función sale sin gastar ciclos en cálculos ni paint.
        """
        # --- Guarda de modo: GEX es exclusivo de opciones ---
        if not self._has_options:
            return

        # --- Guardias comunes ---
        if not self._is_running or not self._buffer:
            return
        if self._canvas.main_plot is None:
            return
        if not self._gex:
            return

        price = self._last_price
        if price <= 0:
            price = self._buffer.get_last_price(self._symbol) or 0
        if price <= 0:
            return

        # --- Debouncing ---
        if not self._should_refresh_gex(price):
            return

        series = self._buffer.get_options()
        if not series:
            return

        oi_perp = self._buffer.get_perp_oi(self._symbol) or 0.0

        # Top walls separados por categoría
        walls = self._gex.get_top_walls(series, current_price=price)

        # Enriquecer con presentación: TACTICAL derecha, STRUCTURAL izquierda
        walls_styled = self._presentation.build_walls(
            walls,
            rotation_index=self._rotation_index,
        )

        # --- Detectar contratos expirados y limpiar canvas ---
        # Si el set de keys activas cambió respecto al último render
        # (contrato expiró o entró uno nuevo), limpiar todas las líneas
        # GEX del canvas antes de redibujar. Sin esto, las líneas de
        # contratos vencidos quedan superpuestas indefinidamente.
        current_keys = set(walls_styled.keys()) if walls_styled else set()
        if current_keys != self._last_gex_keys:
            self._canvas.reset_gex()
        self._last_gex_keys = current_keys

        if walls_styled:
            self._canvas.render_gex(walls_styled)

        # Gamma Flip general — sobre todos los contratos
        flip_general = self._gex.get_gamma_flip(series, current_price=price)
        if flip_general:
            self._canvas._update_gamma_flip(flip_general)

        # Gamma Flip TACTICAL — solo contratos 0DTE
        series_tactical = {
            k: v for k, v in series.items()
            if v.get("cat") == "TACTICAL"
        }
        if series_tactical:
            flip_tactical = self._gex.get_gamma_flip(
                series_tactical, current_price=price,
            )
            if flip_tactical:
                self._canvas._update_gamma_flip_tactical(flip_tactical)

        # Volatility Trigger
        vt = self._gex.get_volatility_trigger(series, current_price=price)
        if vt:
            self._canvas._update_volatility_trigger(vt)

        # Dataset de liquidez → LiquidityContainer
        dataset = self._gex.build_liquidity_dataset(
            series,
            current_price=price,
            oi_perp_total=oi_perp,
        )
        if dataset:
            self._visual_update_liquidity(dataset, price)

    def _visual_refresh_orderbook(self) -> None:
        """
        Despacha los niveles de orderbook al canvas activo.

        Polimorfismo por modo:
          FULL:  LiquidityContainer.update_orderbook(perp, spot)
          LIGHT: LiquidityDistributionCanvas.update_orderbook_perp/spot
                 + finalize_orderbook_range() + set_current_price()
        """
        if not self._is_running or not self._liquidity_canvas:
            return

        perp_levels = self._last_orderbook_perp
        spot_levels = self._last_orderbook_spot

        if not perp_levels and not spot_levels:
            return

        try:
            if self._has_options:
                # FULL — LiquidityContainer
                self._liquidity_canvas.update_orderbook(perp_levels, spot_levels)
            else:
                # LIGHT — LiquidityDistributionCanvas solitario
                if perp_levels:
                    self._liquidity_canvas.update_orderbook_perp(perp_levels)
                if spot_levels:
                    self._liquidity_canvas.update_orderbook_spot(spot_levels)
                self._liquidity_canvas.finalize_orderbook_range()

                # Actualizar línea de precio actual en modo LIGHT
                price = self._last_price
                if price > 0:
                    self._liquidity_canvas.set_current_price(price)

                # --- Centrado inicial diferido (LIGHT) ---
                # En LIGHT, los rangos se calculan desde el orderbook.
                # Se aplican DESPUÉS del primer orderbook exitoso,
                # cuando el canvas ya tiene niveles dibujados.
                if self._initial_ranges_pending:
                    all_levels = perp_levels + spot_levels
                    self._ui_controller.initialize_ranges(
                        orderbook_levels=all_levels,
                        current_price=price if price > 0 else 0,
                    )
                    self._initial_ranges_pending = False

        except Exception as e:
            log.warning("Error en _visual_refresh_orderbook: %s", e)

    def _visual_update_liquidity(
        self,
        dataset: List[Dict],
        current_price: float,
    ) -> None:
        if not dataset:
            return
        try:
            self._liquidity_canvas.update_all(dataset, current_price)

            # --- Centrado inicial diferido (FULL) ---
            # Aplica los rangos Y DESPUÉS del primer update_all exitoso,
            # cuando el canvas ya tiene datos dibujados y los ejes X están
            # calculados. Se ejecuta una sola vez por sesión/cambio de token.
            if self._initial_ranges_pending and self._has_options:
                self._initialize_y_ranges()
                self._initial_ranges_pending = False

        except Exception as e:
            log.warning("Error en _visual_update_liquidity: %s", e)

    # ==================================================================
    # Cambio de activo en caliente
    # ==================================================================

    def switch_token(
        self,
        new_symbol: str,
        new_base_coin: str,
        has_options: bool,
        new_canvas,
    ) -> None:
        """
        Cambio de activo en caliente.

        Secuencia atómica:
          1. Detener timers de polling (opciones, orderbook, rotación, GEX).
          2. Destruir canvas anterior via UIController.
          3. Actualizar configuración interna del token.
          4. Vincular nuevo canvas via UIController.
          5. Los timers y la carga de datos se reinician externamente
             por el caller (start_websocket + load_history_sync).

        El canvas de velas (self._canvas) NO se toca — ese ciclo
        de vida es responsabilidad del caller.

        El widget new_canvas debe ser instanciado por el caller:
          - LiquidityContainer      si has_options=True  (FULL)
          - LiquidityDistributionCanvas si has_options=False (LIGHT)

        Parámetros:
          new_symbol:    símbolo del perpetuo (ej: "BTCUSDT")
          new_base_coin: moneda base para opciones (ej: "BTC")
          has_options:   True=FULL, False=LIGHT
          new_canvas:    widget ya instanciado por el caller
        """
        # 1. Detener timers de datos (no el WS — ese se reconfigura aparte)
        for timer in (
            self._timer_options,
            self._timer_gex,
            self._timer_rotation,
            self._timer_orderbook,
        ):
            if timer:
                try:
                    timer.stop()
                except Exception as e:
                    log.debug("switch_token: error deteniendo timer: %s", e)

        # 2. Destruir canvas anterior
        self._ui_controller.destroy_canvas()

        # 3. Actualizar config del token
        self._symbol    = new_symbol.upper()
        self._base_coin = new_base_coin.upper()
        self._has_options = has_options

        # Resetear estado de datos
        self._last_price          = 0.0
        self._last_gex_price      = 0.0
        self._last_orderbook_perp = []
        self._last_orderbook_spot = []
        self._last_gex_series     = {}
        self._last_gex_keys       = set()
        self._rotation_index      = 0
        self._force_refresh       = False
        self._options_updated     = False
        self._initial_ranges_pending = True

        # Reset del motor de cálculo de indicadores
        self._reset_calculus_engine()

        # Reset del buffer de indicadores
        self._indicator_buffer.clear()
        self._warmup_complete = False

        # GexCalculator: instanciar o liberar según modo
        if has_options and not self._gex:
            self._gex = GexCalculator()
        elif not has_options:
            self._gex = None

        # OptionsClient: liberar si ya no se necesita
        if not has_options:
            self._options = None

        # 4. Vincular nuevo canvas
        self._liquidity_canvas = new_canvas
        self._ui_controller.attach_canvas(new_canvas, has_options)

        log.info(
            "switch_token: %s → %s (mode=%s)",
            self._symbol, new_symbol, "FULL" if has_options else "LIGHT",
        )
        print(
            f"[MainController] switch_token: {new_symbol} "
            f"mode={'FULL' if has_options else 'LIGHT'}"
        )

    # ==================================================================
    # Motor de Cálculo de Indicadores (CalculusManager)
    # ==================================================================

    def _load_session_profiles(self) -> None:
        """
        Construye _active_profiles desde la configuración de sesión.

        Fuente de verdad:
          - session_config.json / session_cfg inyectado: define los perfiles
            disponibles (directorios, módulos). El campo "active" se ignora —
            su presencia en el JSON es legacy y no tiene efecto.
          - active_tracking.json: define exclusivamente qué tokens se levantan.
            Solo los tokens en active_tracking tienen entrada en _active_profiles.

        Si un token está en active_tracking pero no tiene perfil en
        session_config, se usa profile_path vacío (el LM usará el módulo global).
        """
        self._active_profiles = {}

        # ── Leer active_tokens desde el session_cfg inyectado ───────────────
        # El bootstrap ya leyó active_tracking.json y lo inyectó en session_cfg.
        injected = self._cfg.get("session_cfg", {})
        active_tokens: list = []
        if injected and isinstance(injected, dict):
            active_tokens = [t.upper() for t in injected.get("active_tokens", [])]

        # Si no viene en session_cfg, leer active_tracking.json directo
        if not active_tokens:
            tracking_path = os.path.join(
                os.path.dirname(self._session_config_path),
                "active_tracking.json",
            )
            try:
                if os.path.exists(tracking_path):
                    with open(tracking_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    active_tokens = [t.upper() for t in data.get("tracked", []) if t]
            except Exception as e:
                log.warning("[Calculus] No se pudo leer active_tracking.json: %s", e)

        if not active_tokens:
            log.info("[Calculus] active_tracking vacío — sin tokens secundarios.")
            return

        # ── Construir mapa token → profile_dir desde session_cfg ────────────
        profiles_map: Dict[str, str] = {}

        if injected and isinstance(injected, dict) and "profiles" in injected:
            profiles = injected.get("profiles", [])
            base_dir = os.path.dirname(self._session_config_path)
            for p in profiles:
                token        = p.get("token", "").upper()
                profile_path = p.get("profile_path", "")
                if not token or not profile_path:
                    continue
                abs_path = os.path.abspath(os.path.join(base_dir, profile_path))
                profiles_map[token] = abs_path
        elif os.path.exists(self._session_config_path):
            try:
                with open(self._session_config_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                profiles = raw.get("profiles", raw) if isinstance(raw, dict) else raw
                base_dir = os.path.dirname(self._session_config_path)
                for p in profiles:
                    token        = p.get("token", "").upper()
                    profile_path = p.get("profile_path", "")
                    if not token or not profile_path:
                        continue
                    abs_path = os.path.abspath(os.path.join(base_dir, profile_path))
                    profiles_map[token] = abs_path
            except Exception as e:
                log.error("[Calculus] Error leyendo session_config.json: %s", e)

        # ── Filtrar: solo tokens en active_tracking ──────────────────────────
        for token in active_tokens:
            profile_dir = profiles_map.get(token, "")
            self._active_profiles[token] = profile_dir

        log.info(
            "[Calculus] Perfiles activos (de active_tracking): %s",
            list(self._active_profiles.keys()),
        )
        print(
            f"[MainController] Perfiles activos: "
            f"{list(self._active_profiles.keys())}"
        )

    def _get_profile_dir(self, symbol: str) -> Optional[str]:
        """
        Retorna el profile_dir para un symbol, o None si no hay perfil.
        El caller decide si usar None como fallback al módulo global.
        """
        return self._active_profiles.get(symbol.upper())

    # ------------------------------------------------------------------
    # Pipeline de ingesta local — rutas, preloader, persistencia
    # ------------------------------------------------------------------

    def _resolve_local_data_paths(self, symbol: str) -> Dict[str, Optional[str]]:
        """
        Resuelve la ruta del .ender para un token y TF desde local_data_paths.json.

        Formato del JSON (Opcion A — lista de temporalidades por token):
            "tokens": {
                "RENDERUSDT": [
                    {"tf": "1", "ender": "DATASETS/RENDERUSDT/renderusdt_1m.ender"},
                    {"tf": "5", "ender": "DATASETS/RENDERUSDT/renderusdt_5m.ender"}
                ]
            }

        Compatibilidad con formato antiguo (dict con "ender" directo):
            "RENDERUSDT": {"ender": "/ruta/..."}
        Se convierte automaticamente a lista con tf="1".

        Retorna dict con clave "ender" (puede ser None si no existe).
        Cache en self._local_paths por (symbol, tf).
        """
        symbol = symbol.upper()
        tf     = str(getattr(self, "_tf", 1))
        cache_key = f"{symbol}_{tf}"

        if cache_key in self._local_paths:
            return self._local_paths[cache_key]

        result: Dict[str, Optional[str]] = {"ender": None}

        paths_file = self._cfg.get(
            "local_data_paths_file",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_data_paths.json"),
        )

        if not os.path.exists(paths_file):
            log.info("[LocalIngesta] local_data_paths.json no encontrado — modo REST puro.")
            self._local_paths[cache_key] = result
            return result

        try:
            with open(paths_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            tokens_map = data.get("tokens", {})
            raw_entry  = tokens_map.get(symbol)

            # Normalizar formato antiguo (dict) a lista
            if isinstance(raw_entry, dict):
                entries = [{"tf": raw_entry.get("tf", "1"), "ender": raw_entry.get("ender")}]
            elif isinstance(raw_entry, list):
                entries = raw_entry
            else:
                entries = []

            # Buscar la entrada que coincide con el TF activo
            ender_path = None
            for entry in entries:
                if str(entry.get("tf", "1")) == tf:
                    ender_path = entry.get("ender")
                    break

            # Fallback: usar la primera entrada disponible si no hay match exacto
            if ender_path is None and entries:
                ender_path = entries[0].get("ender")
                log.info(
                    "[LocalIngesta] %s: sin entrada para TF=%s, usando TF=%s como fallback.",
                    symbol, tf, entries[0].get("tf", "?"),
                )

            if ender_path:
                if not os.path.isabs(ender_path):
                    ender_path = os.path.abspath(
                        os.path.join(os.path.dirname(paths_file), ender_path)
                    )
                if not os.path.exists(ender_path):
                    log.info(
                        "[LocalIngesta] .ender no encontrado en disco: %s", ender_path
                    )
                    ender_path = None

            result["ender"] = ender_path

            log.info(
                "[LocalIngesta] Rutas para %s TF=%s → ender=%s sidecar=%s",
                symbol, tf,
                "ok" if ender_path else "no",
                "derivado" if ender_path else "no",
            )

        except Exception as e:
            log.warning("[LocalIngesta] Error leyendo local_data_paths.json: %s", e)

        self._local_paths[cache_key] = result
        return result

    def _build_gap_fill_fn(self, symbol: str):
        """
        Construye el callable thread-safe que LocalPreloader usa para
        rellenar gaps via REST.

        El callable bloquea hasta tener resultado (usa threading.Event
        en lugar de asyncio.Event porque corre en el QThread del preloader,
        no dentro del _rest_loop).

        Firma del callable: (symbol, since_ts, count) -> List[Dict]
        Retorna [] si falla o si el rest_loop no está disponible.
        """
        import threading as _threading

        def _gap_fill(sym: str, since_ts: int, count: int):
            if not self._scheduler or not self._rest_loop:
                return []

            result_hold: Dict = {}
            done = _threading.Event()

            def _on_result(s: str, candles: list):
                result_hold["candles"] = candles
                done.set()

            def _on_error(s: str, msg: str):
                result_hold["error"] = msg
                done.set()

            self._scheduler.enqueue_warmup(
                symbol      = sym.upper(),
                n_candles   = count,
                on_result   = _on_result,
                on_error    = _on_error,
                on_progress = lambda m: log.debug("[GapFill] %s", m),
            )

            done.wait(timeout=60)
            return result_hold.get("candles", [])

        return _gap_fill

    
