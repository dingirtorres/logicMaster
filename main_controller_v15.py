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

    def _fill_indicator_gap(
        self,
        symbol: str,
        ender_path: str,
        candles: List[Dict],
        tensor_dict: Dict[str, list],
        gap_count: int,
        interval_ms: int,
    ) -> tuple:
        """
        Calcula indicadores para las velas del gap usando SeriesExtractor
        y persiste al .iend.

        Usa SeriesExtractor.run_incremental — calcula solo las filas
        faltantes desde iend.n_rows hasta el final del .ender, con
        contexto de warmup incluido. No usa LogicMaster (streaming only).

        Loop de verificacion: si cerro una vela nueva durante el calculo,
        la descarga, appendea al .ender, y re-corre el extractor.

        Retorna (tensor_dict_actualizado, candles_actualizadas).
        """
        import sys as _sys
        from sidecar_io import ender_to_iend_path, SidecarReader
        from binary_io import append_candles as _append_candles

        iend_path = ender_to_iend_path(ender_path)

        # Resolver ruta al extractor
        _extractor_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "EXTRACTOR-V-25"
        )
        if _extractor_dir not in _sys.path:
            _sys.path.insert(0, _extractor_dir)

        from series_extractor import SeriesExtractor

        # Resolver lm_config_path desde el perfil activo
        lm_profile_dir = self._active_profiles.get(symbol.upper(), "")
        lm_config_path = os.path.join(lm_profile_dir, "logicMaster.json") if lm_profile_dir else ""
        if not lm_config_path or not os.path.exists(lm_config_path):
            log.warning(
                "[LocalIngesta] %s: sin lm_config_path — no se puede calcular gap.",
                symbol,
            )
            return tensor_dict, candles

        def _run_extractor():
            """Corre run_incremental y retorna n_rows_added."""
            try:
                with SidecarReader(iend_path) as sc:
                    since_row = sc.n_rows
                ext = SeriesExtractor(
                    ender_path           = ender_path,
                    lm_config_path       = lm_config_path,
                    source_format        = "ender",
                    interval_ms_override = interval_ms,
                )
                res = ext.run_incremental(since_row=since_row, iend_path=iend_path)
                if res["status"] == "ok":
                    log.info(
                        "[LocalIngesta] %s: gap calculado — +%d filas en %.1fs.",
                        symbol, res["n_rows_added"], res["elapsed_s"],
                    )
                    print(
                        f"[MainController] Gap → .iend: {symbol} "
                        f"{res['n_rows_added']} filas persistidas."
                    )
                    return res["n_rows_added"]
                else:
                    log.warning(
                        "[LocalIngesta] %s: SeriesExtractor fallo: %s",
                        symbol, res["message"],
                    )
                    return 0
            except Exception as e:
                log.warning("[LocalIngesta] %s: error en extractor: %s", symbol, e)
                return 0

        # Primera pasada
        _run_extractor()

        # Actualizar tensor_dict desde el .iend recien calculado
        try:
            with SidecarReader(iend_path) as sc:
                sc_last_ts = sc.get_last_ts()
                buf_first_ts = int(candles[0]["timestamp"])
                sc_idx = (buf_first_ts - sc.first_ts) // interval_ms
                sc_data = sc.read_slice(int(sc_idx), len(candles))
            sc_ts = sc_data.pop("_timestamps", None)
            if sc_ts is not None and len(sc_ts) == len(candles):
                tensor_dict = {k: arr.tolist() for k, arr in sc_data.items()}
                log.info(
                    "[LocalIngesta] %s: tensor recargado desde .iend — %d keys, %d filas.",
                    symbol, len(tensor_dict), len(sc_ts),
                )
        except Exception as e:
            log.warning("[LocalIngesta] %s: error recargando tensor: %s", symbol, e)

        # Loop de verificacion: velas nuevas cerradas durante el calculo
        gap_fill_fn = self._build_gap_fill_fn(symbol)
        max_retries = 10

        for _retry in range(max_retries):
            now_ms   = int(time.time() * 1000)
            last_ts  = int(candles[-1]["timestamp"])
            elapsed  = now_ms - last_ts

            if elapsed < interval_ms:
                break

            n_new    = min(elapsed // interval_ms, 50)
            since_ts = last_ts + interval_ms

            log.info(
                "[LocalIngesta] %s: %d vela(s) nueva(s) durante calculo. "
                "Descargando...", symbol, n_new,
            )

            try:
                new_candles = gap_fill_fn(symbol, since_ts, int(n_new))
            except Exception as e:
                log.warning("[LocalIngesta] %s: REST en verificacion fallo: %s", symbol, e)
                break

            if not new_candles:
                break

            filtered = [c for c in new_candles if int(c.get("timestamp", 0)) > last_ts]
            if not filtered:
                break

            _append_candles(ender_path, filtered)
            candles.extend(filtered)
            _run_extractor()

            log.info(
                "[LocalIngesta] %s: +%d vela(s) en verificacion. Total: %d velas.",
                symbol, len(filtered), len(candles),
            )

        # Recortar al tamanio del buffer si crecio
        n_limit = self._cfg.get("candle_limit", 600)
        if len(candles) > n_limit:
            excess   = len(candles) - n_limit
            candles  = candles[excess:]
            for key in tensor_dict:
                tensor_dict[key] = tensor_dict[key][excess:]

        return tensor_dict, candles

    def _load_token(self, symbol: str) -> Optional[Dict]:
        """
        Carga el buffer de velas e indicadores para un token.

        Patrón unificado para token primario y secundarios:
          1. Intentar LocalPreloader si existe .ender en disco.
          2. Si LocalPreloader falla (sig_error) o no hay .ender:
             caer al REST puro via _scheduler.enqueue_warmup.

        Bloquea hasta tener resultado (se invoca desde un thread,
        no desde el Qt main thread).

        Retorna dict con:
            {
                "symbol":       str,
                "candles":      List[Dict],       ← siempre presente si ok
                "tensor_dict":  Dict|None,        ← None si no hay sidecar
                "gap_filled":   int,              ← velas REST añadidas al .ender
                "source":       "local"|"rest",
            }
        O None si ambas vías fallaron.
        """
        import threading as _threading
        symbol = symbol.upper()

        # ── Intento 1: ingesta local ───────────────────────────────────
        paths = self._resolve_local_data_paths(symbol)
        ender_path   = paths.get("ender")

        if ender_path:
            # Calcular interval_ms desde el timeframe
            try:
                tf_str = str(self._timeframe)
                tf_min = 1440 if tf_str.upper() == "D" else int(tf_str)
                interval_ms = tf_min * 60_000
            except Exception:
                interval_ms = 300_000

            _lm_profile_dir = self._active_profiles.get(symbol.upper(), "")
            _lm_config_path = os.path.join(_lm_profile_dir, "logicMaster.json") if _lm_profile_dir else ""

            preloader = LocalPreloader(
                symbol                  = symbol,
                ender_path              = ender_path,
                n_candles               = self._cfg["candle_limit"],
                interval_ms             = interval_ms,
                gap_fill_fn             = self._build_gap_fill_fn(symbol),
                lm_config_path          = _lm_config_path if os.path.exists(_lm_config_path) else None,
                extractor_gap_threshold = 600,
            )

            # Conectar señales para progreso/UI — no se usan para
            # sincronización, solo para logging en el MC.
            preloader.sig_progress.connect(
                lambda sym, msg: self.sig_status.emit(f"[{sym}] {msg}")
            )

            # Arrancar y esperar — QThread.wait() bloquea el thread
            # actual (executor) sin tocar el Qt main thread.
            # Los resultados se leen desde atributos del objeto,
            # no desde señales (que requieren event loop Qt activo).
            preloader.start()
            finished = preloader.wait(65_000)  # ms — 65s timeout

            if not finished:
                log.warning(
                    "[LocalIngesta] %s: preloader timeout — fallback REST.",
                    symbol,
                )
                print(f"[MainController] {symbol}: preloader timeout → REST.")
            elif preloader.result_ok and preloader.result_candles:
                candles    = preloader.result_candles
                tensor     = preloader.result_tensor_dict
                gap_filled = preloader.result_gap_filled

                log.info(
                    "[LocalIngesta] %s cargado desde .ender "
                    "(%d velas, sidecar=%s, gap=%d).",
                    symbol, len(candles),
                    "sí" if tensor else "no",
                    gap_filled,
                )
                print(
                    f"[MainController] {symbol}: carga local OK — "
                    f"{len(candles)} velas, "
                    f"sidecar={'sí' if tensor else 'no'}, "
                    f"gap={gap_filled}"
                )

                # ── Llenar gap de indicadores ────────────────────────
                # Si el tensor no cubre todas las candles, calcular los
                # indicadores faltantes. El gap real se mide entre la
                # longitud del tensor y la de las candles — no se usa
                # gap_filled porque puede haber velas previas sin
                # indicadores en el .iend (sesión anterior incompleta).
                if tensor is not None:
                    first_key = next(iter(tensor))
                    actual_gap = len(candles) - len(tensor[first_key])
                    if actual_gap > 0:
                        tensor, candles = self._fill_indicator_gap(
                            symbol, ender_path, candles, tensor,
                            actual_gap, interval_ms,
                        )

                if self._persistence_manager and ender_path:
                    from sidecar_io import ender_to_iend_path as _e2i
                    _sc_path = _e2i(ender_path)
                    self._persistence_manager.register_token(
                        symbol, ender_path,
                        _sc_path if os.path.exists(_sc_path) else None,
                    )

                return {
                    "symbol":      symbol,
                    "candles":     candles,
                    "tensor_dict": tensor,
                    "gap_filled":  gap_filled,
                    "source":      "local",
                }
            else:
                log.info(
                    "[LocalIngesta] %s: preloader falló (%s) — fallback REST.",
                    symbol, preloader.result_error or "sin detalle",
                )
                print(
                    f"[MainController] {symbol}: preloader falló → REST. "
                    f"({preloader.result_error or 'sin detalle'})"
                )

        # ── Intento 2: REST puro ───────────────────────────────────────
        import threading as _threading

        rest_result: Dict = {}
        rest_done = _threading.Event()

        def _on_rest_result(sym: str, candles: list):
            rest_result["candles"] = candles
            rest_done.set()

        def _on_rest_error(sym: str, msg: str):
            rest_result["error"] = msg
            rest_done.set()

        if not self._scheduler:
            log.error("[LocalIngesta] _load_token(%s): scheduler no disponible.", symbol)
            return None

        self._scheduler.enqueue_warmup(
            symbol      = symbol,
            n_candles   = self._cfg["candle_limit"],
            on_result   = _on_rest_result,
            on_error    = _on_rest_error,
            on_progress = lambda m: self.sig_status.emit(m),
        )
        rest_done.wait(timeout=120)

        candles = rest_result.get("candles")
        if not candles:
            log.error(
                "[LocalIngesta] %s: REST también falló: %s",
                symbol, rest_result.get("error", "sin detalle"),
            )
            return None

        log.info(
            "[LocalIngesta] %s cargado desde REST (%d velas).", symbol, len(candles)
        )
        print(f"[MainController] {symbol}: carga REST OK — {len(candles)} velas.")

        return {
            "symbol":      symbol,
            "candles":     candles,
            "tensor_dict": None,
            "gap_filled":  0,
            "source":      "rest",
        }

    def set_active_token(self, symbol: str) -> None:
        """
        Cambia el token activo en la UI sin recargar datos desde la API.

        Requiere que el buffer de velas del símbolo ya esté cargado en
        memoria (load_initial_candles llamado previamente). Si no hay
        velas en el buffer, loguea un warning y no hace nada.

        Secuencia:
          1. Suspender capa visual (_is_running=False, limpiar pendientes).
          2. Actualizar _symbol y filtro del buffer.
          3. Redibujar canvas desde buffer del nuevo token.
          4. Sincronizar _last_rendered_ts y _warmup_complete.
          5. Notificar al CalculusManager.
          6. Resetear estado visual.
          7. Activar capa visual via mark_canvas_ready() — drena pendientes.

        NO detiene ni reinicia el WS — el WS ya está recibiendo datos
        de todos los símbolos de la lista activa simultáneamente.
        """
        symbol = symbol.upper()

        # Verificar que el buffer esté completamente cargado.
        # _ready_tokens solo se popula cuando _load_history_async o
        # _load_secondary_tokens terminan exitosamente — no cuando el
        # buffer tiene una sola vela del WS.
        if symbol not in self._ready_tokens:
            log.warning(
                "[MainController] set_active_token(%s): buffer no listo "
                "para fast switch — esperando carga completa.",
                symbol,
            )
            print(
                f"[MainController] set_active_token({symbol}): "
                f"buffer aún no cargado. Tokens listos: {self._ready_tokens}"
            )
            return

        candles = self._buffer.get_candles(symbol) if self._buffer else []
        if not candles:
            log.warning(
                "[MainController] set_active_token(%s): sin velas en buffer.",
                symbol,
            )
            print(f"[MainController] set_active_token: no hay buffer para {symbol}.")
            return

        prev_symbol = self._symbol

        # Aplicar el modo del nuevo token al activarlo
        mode = self._token_modes.get(symbol, "calc")
        self._lm_signal_enabled     = mode in ("signal", "auto")
        self._lm_auto_trade_enabled = mode == "auto"

        # ── 1. Suspender capa visual ───────────────────────────────
        # _is_running=False: _on_candle acumulará en _pending_ticks
        # en lugar de intentar renderizar en un canvas que aún está
        # siendo repintado con datos del nuevo token.
        self._is_running = False
        self._pending_ticks.clear()

        # ── 2. Actualizar símbolo y filtro del buffer ──────────────
        self._symbol = symbol
        # El buffer ya no filtra por símbolo activo — acumula todo.
        # set_symbol se mantiene para compatibilidad con get_symbol().
        if self._buffer:
            self._buffer.set_symbol(symbol)

        # ── 3. Redibujar canvas desde buffer ───────────────────────
        df = self._data_candles_to_dataframe(candles)
        if df is not None and not df.empty and self._canvas:
            self._canvas.plot_full_chart(df)
            log.info(
                "[MainController] Canvas redibujado para %s (%d velas).",
                symbol, len(candles),
            )

        # Renderizar serie de indicadores desde el buffer persistente.
        # La serie ya está calculada — no hay recálculo, solo pintado.
        indicator_series = self._buffer.get_indicator_series(symbol) if self._buffer else {}
        if indicator_series and self._canvas:
            self._render_indicators_from_series(symbol, indicator_series)

        # ── 4. Sincronizar timestamps de referencia ────────────────
        # _last_rendered_ts debe apuntar a la última vela que
        # plot_full_chart acaba de pintar — es la referencia para que
        # _gateway_render_candle decida append vs update correctamente.
        self._last_rendered_ts = int(candles[-1].get("timestamp", 0))
        self._warmup_complete  = bool(indicator_series)

        print(
            f"[GW] set_active_token _last_rendered_ts={self._last_rendered_ts} "
            f"candles={len(candles)} warmup={self._warmup_complete}"
        )

        # ── 5. Notificar al CalculusManager ───────────────────────
        if self._calculus_control_q:
            if symbol not in getattr(self, '_known_calculus_tokens', set()):
                self._calculus_control_q.put({
                    "type":     CMD_WARMUP,
                    "token_id": symbol,
                    "candles":  candles,
                })
                if not hasattr(self, '_known_calculus_tokens'):
                    self._known_calculus_tokens = set()
                self._known_calculus_tokens.add(symbol)
            else:
                self._calculus_control_q.put({
                    "type":     CMD_RESET,
                    "token_id": symbol,
                })

        # ── 6. Resetear estado visual ──────────────────────────────
        self._last_price             = 0.0
        self._last_gex_price         = 0.0
        self._last_rendered_ohlcv    = {}
        self._last_rendered_countdown = ""
        self._last_rendered_price    = 0.0
        self._force_refresh          = True
        self._initial_ranges_pending = True

        # ── 7. Activar capa visual — drena pendientes ──────────────
        # mark_canvas_ready() setea _is_running=True y drena
        # _pending_ticks con filtro de timestamp.
        self.mark_canvas_ready()

        # Refrescar GEX desde buffer del nuevo token.
        QTimer.singleShot(0, self._visual_refresh_gex)

        # Emitir señal de estado para la UI.
        self.sig_status.emit(f"{symbol} | {self._timeframe}m")

        log.info(
            "[MainController] Token activo: %s → %s",
            prev_symbol, symbol,
        )
        print(f"[MainController] Token activo: {prev_symbol} → {symbol}")

    # ==================================================================
    # Fast switch — reset completo de la capa visual
    # ==================================================================

    def reset_visual_layer(
        self,
        symbol: str,
        base_coin: str,
        has_options: bool,
        new_canvas,
        new_liquidity_canvas,
    ) -> None:
        """
        Reinicia la capa visual completa para un cambio de token.

        La capa de datos (buffer handler, CalculusManager, WS,
        HTTP client, RestScheduler) permanece intacta — los 8 canales
        siguen transmitiendo en background.

        Los canvas nuevos llegan ya creados por LM_GEX_O — este método
        no crea ni destruye widgets Qt. Solo los recibe, los conecta
        a la capa de datos, y pinta lo que ya está en el buffer.

        Secuencia:
          1. Suspender capa visual (timers + flag _is_running).
          2. Destruir liquidity canvas viejo via UIController.
          3. Actualizar estado del token y referencias a canvas.
          4. Vincular liquidity canvas nuevo al UIController.
          5. Recrear clientes de opciones/orderbook según modo.
          6. Inicializar canvas fresco (init_chart + plot_full_chart).
          7. Renderizar indicadores desde buffer persistente.
          8. Resetear estado visual.
          9. Reconectar señales del canvas y marcar ready.
         10. Reiniciar timers visuales.
         11. Emitir señal de estado.

        Parámetros
        ----------
        symbol             : ej. "ETHUSDT"
        base_coin          : ej. "ETH" (para OptionsClient)
        has_options         : True=FULL (GEX+orderbook), False=LIGHT (solo orderbook)
        new_canvas          : CanvasCore recién creado (velas)
        new_liquidity_canvas: LiquidityContainer o LiquidityDistributionCanvas
        """
        prev_symbol = self._symbol

        # ── 1. Suspender capa visual ───────────────────────────────
        # Desactivar _is_running ANTES de tocar canvas — los callbacks
        # del WS drain (_on_candle, _on_ticker) salen temprano.
        # Limpiar la cola de pendientes: cualquier tick del token
        # anterior que no se haya drenado ya no es relevante.
        self._is_running = False
        self._pending_ticks.clear()

        for timer in (
            self._timer_options,
            self._timer_gex,
            self._timer_rotation,
            self._timer_orderbook,
        ):
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass

        self._timer_options   = None
        self._timer_gex       = None
        self._timer_rotation  = None
        self._timer_orderbook = None

        # ── 2. Destruir liquidity canvas viejo via UIController ────
        if self._ui_controller:
            try:
                self._ui_controller.destroy_canvas()
            except Exception as e:
                log.warning(
                    "reset_visual_layer: error en destroy_canvas: %s", e
                )

        # ── 3. Actualizar estado del token y referencias ───────────
        self._symbol      = symbol.upper()
        self._base_coin   = base_coin.upper()
        self._has_options  = has_options
        self._canvas              = new_canvas
        self._liquidity_canvas    = new_liquidity_canvas

        # ── 4. Vincular liquidity canvas nuevo al UIController ─────
        if self._ui_controller:
            self._ui_controller.attach_canvas(
                new_liquidity_canvas, has_options,
            )

        # ── 5. Filtro de símbolo del buffer ────────────────────────
        if self._buffer:
            self._buffer.set_symbol(self._symbol)

        # ── 6. Recrear clientes de opciones/orderbook ──────────────
        # Los clientes son wrappers stateless sobre el BybitHttpClient
        # compartido — recrearlos es instantáneo.
        if has_options:
            self._gex     = GexCalculator()
            self._options  = OptionsClient(self._http, self._base_coin)
        else:
            self._gex     = None
            self._options  = None

        self._orderbook = OrderbookClient(self._http, self._base_coin)

        # ── 7. Inicializar canvas fresco y pintar desde buffer ─────
        new_canvas.init_chart()

        candles = self._buffer.get_candles(self._symbol) if self._buffer else []
        if candles:
            df = self._data_candles_to_dataframe(candles)
            if df is not None and not df.empty:
                new_canvas.plot_full_chart(df)

        # ── 8. Renderizar indicadores desde buffer persistente ─────
        indicator_series = (
            self._buffer.get_indicator_series(self._symbol)
            if self._buffer else {}
        )
        if indicator_series:
            self._render_indicators_from_series(self._symbol, indicator_series)
        else:
            # Sin serie aún — el _on_warmup_ack la poblará cuando
            # el CalculusManager responda.
            self._indicator_buffer.clear()
            self._key_to_base.clear()

        # ── 8b. Repintar señales históricas del LM ─────────────────
        # Solo si el modo señal está activo y hay señales en el buffer.
        if self._lm_signal_enabled and self._buffer:
            signals = self._buffer.get_signals(self._symbol)
            om = getattr(self, "_operative_manager", None)
            if signals and om is not None:
                om.render_signal_history(signals)

        # ── 9. Resetear estado visual ──────────────────────────────
        self._last_price              = 0.0
        self._last_gex_price          = 0.0
        self._last_orderbook_perp     = []
        self._last_orderbook_spot     = []
        self._last_gex_series         = {}
        self._last_gex_keys           = set()
        self._rotation_index          = 0
        self._force_refresh           = True
        self._options_updated         = False
        self._initial_ranges_pending  = True
        self._last_rendered_ohlcv     = {}
        self._last_rendered_countdown = ""
        self._last_rendered_price     = 0.0

        # _last_rendered_ts: el timestamp de la última vela que el canvas
        # acaba de pintar con plot_full_chart. Es la referencia para decidir
        # append vs update cuando lleguen los ACKs del CM.
        if candles:
            self._last_rendered_ts = int(candles[-1].get("timestamp", 0))
        else:
            self._last_rendered_ts = 0

        print(
            f"[GW] INIT _last_rendered_ts={self._last_rendered_ts} "
            f"candles={len(candles)} x_counter={new_canvas._x_counter}"
        )

        # _warmup_complete se mantiene True si el CM ya tiene contexto
        # para este token (ya pasó por WARMUP_ACK). Si no, se activará
        # cuando llegue el ack.
        self._warmup_complete = bool(indicator_series)

        # ── 10. Reconectar señales del canvas y marcar ready ───────
        # mark_canvas_ready() setea _is_running=True y conecta
        # sigVerticalLinePlaced + sigDrawingsCleared al canvas nuevo.
        # Como el canvas anterior fue destruido, no hay acumulación.
        self.mark_canvas_ready()

        # ── 11. Reiniciar timers visuales ──────────────────────────
        self._restart_visual_timers()

        # ── 12. Notificar al CalculusManager ───────────────────────
        # Si el CM ya tiene contexto para este token, hacerlo activo.
        # Si no, enviar CMD_WARMUP con las velas del buffer.
        if self._calculus_control_q:
            known = getattr(self, '_known_calculus_tokens', set())
            if self._symbol in known:
                # No resetear el CM — el LM tiene el histórico completo
                pass
            elif candles:
                self._calculus_control_q.put({
                    "type":     CMD_WARMUP,
                    "token_id": self._symbol,
                    "candles":  candles,
                })
                if not hasattr(self, '_known_calculus_tokens'):
                    self._known_calculus_tokens = set()
                self._known_calculus_tokens.add(self._symbol)

        # ── 13. Emitir señal de estado ─────────────────────────────
        self.sig_status.emit(f"{self._symbol} | {self._timeframe}m")

        log.info(
            "[MainController] reset_visual_layer: %s → %s "
            "(options=%s, candles=%d, indicators=%d)",
            prev_symbol, self._symbol,
            "FULL" if has_options else "LIGHT",
            len(candles),
            len(indicator_series),
        )
        print(
            f"[MainController] Visual layer reset: {prev_symbol} → "
            f"{self._symbol} ({'FULL' if has_options else 'LIGHT'}) | "
            f"{len(candles)} velas | {len(indicator_series)} series"
        )

    def _restart_visual_timers(self) -> None:
        """
        Recrea y arranca los timers de polling visual.

        Separado de reset_visual_layer para claridad.
        Los timers de la capa de datos (_timer_ws, calculus) no se tocan.
        """
        # Timer — opciones, GEX, rotación (solo si FULL)
        if self._has_options:
            self._timer_options = QTimer(self)
            self._timer_options.setInterval(
                self._cfg["options_poll_interval_ms"]
            )
            self._timer_options.timeout.connect(self._poll_options)
            self._timer_options.start()

            self._timer_gex = QTimer(self)
            self._timer_gex.setInterval(
                self._cfg["gex_refresh_interval_ms"]
            )
            self._timer_gex.timeout.connect(self._periodic_refresh)
            self._timer_gex.start()

            rotation_ms = self._presentation.ROTATION_INTERVAL_MS
            self._timer_rotation = QTimer(self)
            self._timer_rotation.setInterval(rotation_ms)
            self._timer_rotation.timeout.connect(self._advance_rotation)
            self._timer_rotation.start()

        # Timer — orderbook (siempre activo)
        self._timer_orderbook = QTimer(self)
        self._timer_orderbook.setInterval(
            self._cfg["orderbook_poll_interval_ms"]
        )
        self._timer_orderbook.timeout.connect(self._poll_orderbook)
        self._timer_orderbook.start()

    def _warmup_all_tokens(self) -> None:
        """
        Dispara la carga de buffers de tokens secundarios en background.

        Se llama desde LM_GEX_J con QTimer.singleShot(5000) después del
        arranque del token principal.

        _load_secondary_tokens corre en el _rest_loop, carga las velas de
        cada token secuencialmente via el RestScheduler, y para cada buffer
        cargado envía el CMD_WARMUP al CalculusManager con las 3000 velas
        reales (no con la única vela del WS). Por eso este método ya no
        envía ningún CMD_WARMUP directamente — todo se hace en el callback
        de carga, garantizando que el warmup use el buffer completo.
        """
        if not self._rest_loop or not self._scheduler:
            log.warning(
                "[MainController] _warmup_all_tokens: rest_loop o "
                "scheduler no disponibles."
            )
            return

        if not hasattr(self, '_known_calculus_tokens'):
            self._known_calculus_tokens = set()

        secondary = [
            sym for sym in self._active_profiles
            if sym != self._symbol
        ]
        if not secondary:
            return

        # Lanzar la carga REST de tokens secundarios en el rest_loop.
        # El CMD_WARMUP de cada token se envía dentro de la corrutina
        # cuando su buffer termina de cargar (con las 3000 velas reales).
        asyncio.run_coroutine_threadsafe(
            self._load_secondary_tokens(),
            self._rest_loop,
        )

    def _init_calculus_engine(self) -> None:
        """
        Arranca el CalculusManager como proceso hijo.
        Invocado en load_history_sync() después de la carga exitosa.
        """
        # Legacy config para _render_indicator_buffer (red de seguridad)
        self._indicators_config = load_indicators_config()

        try:
            engine, pipe, ctrl_q, stop_evt = create_calculus_engine(
                config={
                    "tensor_length":      self._cfg.get("buffer_size", 3000),
                    "pausa_ms":           10,
                    "pausa_cada_n_velas": 500,
                    "buffer_size_config": self._cfg.get("buffer_size", 3000),
                    "log_level":          "WARNING",
                },
                logic_module_path=self._cfg.get(
                    "logic_master_path", "logicMaster_008_8.py"
                ),
                logic_class_name="LogicMaster",
            )
            engine.start()
            self._calculus_engine = engine
            self._calculus_pipe = pipe
            self._calculus_control_q = ctrl_q
            self._calculus_stop_event = stop_evt

            log.info("[Calculus] Motor de cálculo arrancado.")
            print("[MainController] Motor de cálculo de indicadores arrancado.")

        except Exception as e:
            log.error("[Calculus] Error arrancando motor: %s", e, exc_info=True)
            print(f"[MainController] Error arrancando motor de cálculo: {e}")
            self._calculus_engine = None

    def _start_calculus_timer(self) -> None:
        """Arranca el QTimer de drenaje del Pipe del CalculusManager."""
        if not self._calculus_pipe:
            return

        self._timer_calculus = QTimer(self)
        self._timer_calculus.setInterval(
            self._cfg.get("ws_queue_drain_interval_ms", 100)
        )
        self._timer_calculus.timeout.connect(self._drain_calculus_pipe)
        self._timer_calculus.start()
        log.info("[Calculus] Timer de drenaje arrancado.")

    def _warmup_calculus_engine(self) -> None:
        """
        Envía las velas históricas del buffer al CalculusManager (flujo clásico).
        Mantenido para compatibilidad — cuando _using_warmup_pool=True no se invoca.
        """
        if not self._calculus_control_q or not self._buffer:
            return

        candles = self._buffer.get_candles(self._symbol)
        if not candles:
            log.warning("[Calculus] Sin velas en buffer para warmup.")
            return

        self._calculus_control_q.put({
            "type":     CMD_WARMUP,
            "token_id": self._symbol,
            "candles":  candles,
        })

        if not hasattr(self, '_known_calculus_tokens'):
            self._known_calculus_tokens = set()
        self._known_calculus_tokens.add(self._symbol)

        log.info(
            "[Calculus] Warmup enviado: %d velas para %s.",
            len(candles), self._symbol,
        )
        print(
            f"[MainController] Warmup de indicadores: "
            f"{len(candles)} velas para {self._symbol}"
        )

    # ------------------------------------------------------------------
    # WarmupPool — pool paralelo de cálculo de indicadores
    # ------------------------------------------------------------------

    def _resolve_lm_module_path(self, symbol: str) -> str:
        """
        Resuelve la ruta del módulo LogicMaster para un token.

        Prioridad:
          1. calculus_manager.json (mismo archivo que usa el CalculusManager).
          2. Primer logicMaster*.py encontrado en el profile_dir del token.
          3. logic_master_path global desde config.
        """
        import glob
        import json as _json

        # 1. calculus_manager.json
        try:
            cfg_path = os.path.join(os.getcwd(), "calculus_manager.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    data = _json.load(f)
                if isinstance(data, dict):
                    path = data.get(symbol.upper(), "")
                    if path and os.path.exists(path):
                        return path
        except Exception:
            pass

        # 2. logicMaster*.py en el profile dir
        profile_dir = self._active_profiles.get(symbol.upper())
        if profile_dir and os.path.isdir(profile_dir):
            candidates = sorted(glob.glob(
                os.path.join(profile_dir, "logicMaster*.py")
            ))
            if candidates:
                return candidates[0]

        # 3. Fallback global
        return self._cfg.get("logic_master_path", "logicMaster_008_8.py")

    def _start_warmup_pool(self) -> None:
        """
        Crea el WarmupPoolManager, conecta señales, y envía el token primario.
        Invocado en start_websocket() (Phase A) en lugar de _warmup_calculus_engine.

        Los tokens secundarios se envían al pool desde _load_secondary_tokens
        conforme se descargan sus velas.

        Optimización con sidecar:
        Si _sidecar_preloaded=True el tensor ya está en el buffer —
        no hace falta correr el WarmupPool completo para el token primario.
        Se envía mini-warmup directo al CM (skip_series=True) y el WarmupPool
        solo se inicia para los tokens secundarios que lo necesiten.
        """
        total = len(self._active_profiles) or 1

        self._warmup_pool = WarmupPoolManager(
            primary_token = self._symbol,
            total_tokens  = total,
            parent        = self,
        )
        self._warmup_pool.sig_primary_ready.connect(self._on_pool_primary_ready)
        self._warmup_pool.sig_tensor_ready.connect(self._on_pool_tensor_ready)
        self._warmup_pool.sig_progress.connect(self._on_pool_progress)
        self._warmup_pool.sig_error.connect(self._on_pool_error)
        self._warmup_pool.sig_pool_complete.connect(
            lambda: log.info("[WarmupPool] Pool completo.")
        )

        if self._buffer:
            candles = self._buffer.get_candles(self._symbol)
            if candles:
                # ── Fast path: sidecar ya inyectado ─────────────────────────
                if self._sidecar_preloaded:
                    log.info(
                        "[WarmupPool] Tensor del sidecar disponible para %s "
                        "— mini-warmup directo al CM.",
                        self._symbol,
                    )
                    print(
                        f"[MainController] {self._symbol}: sidecar preloaded — "
                        f"skip WarmupPool, mini-warmup CM directo."
                    )
                    self._using_warmup_pool = True   # para que _on_warmup_ack procese is_mini

                    # Calcular preheating_min desde el perfil
                    profile_dir = self._active_profiles.get(self._symbol, "")
                    preheating_min = 200  # fallback
                    if profile_dir:
                        lm_json = os.path.join(profile_dir, "logicMaster.json")
                        try:
                            with open(lm_json, "r", encoding="utf-8") as f:
                                lm_cfg = json.load(f)
                            preheating_min = int(lm_cfg.get("preheating_size", 200))
                        except Exception:
                            pass

                    if self._calculus_control_q and self._symbol not in self._known_calculus_tokens:
                        mini = candles[-preheating_min:]
                        self._calculus_control_q.put({
                            "type":        CMD_WARMUP,
                            "token_id":    self._symbol,
                            "candles":     mini,
                            "skip_series": True,
                        })
                        self._known_calculus_tokens.add(self._symbol)
                        log.info(
                            "[WarmupPool] Mini-warmup CM (sidecar path): %s (%d velas).",
                            self._symbol, len(mini),
                        )
                        print(
                            f"[MainController] Mini-warmup CM (sidecar): "
                            f"{self._symbol} ({len(mini)} velas — skip_series=True)"
                        )

                    # ── Inicializar timestamps que _on_pool_primary_ready
                    # normalmente setea, pero no corre en el fast path
                    # porque submit_token() se saltea y sig_primary_ready
                    # nunca se emite para el token primario.
                    # Sin esto: _last_indicator_ts=0 y _last_rendered_ts=0
                    # → el canvas no sabe en qué punto del tiempo está
                    # → los primeros ticks del WS se descartan o
                    #   se appenden desde x=0 en vez de continuar desde
                    #   la última vela del buffer.
                    last_c = candles[-1] if candles else None
                    if last_c:
                        _sidecar_ts = int(last_c.get("timestamp", 0))
                        self._last_indicator_ts = _sidecar_ts
                        self._last_rendered_ts  = _sidecar_ts
                        self._indicator_ts_by_token[self._symbol] = _sidecar_ts
                        log.info(
                            "[WarmupPool] Sidecar fast path — timestamps init: "
                            "%s ts=%d",
                            self._symbol, _sidecar_ts,
                        )
                        print(
                            f"[MainController] Sidecar fast path — "
                            f"_last_indicator_ts={_sidecar_ts} "
                            f"_last_rendered_ts={_sidecar_ts}"
                        )

                    self._warmup_pool.start()
                    return

                # ── Partial sidecar path: CM clásico (~1.5s) ──────────────
                # El sidecar cubría parte del buffer. Los datos parciales ya
                # están en el MarketBufferHandler (para fast switch futuro).
                # El CM procesa las 600 velas completas con skip_series=False
                # y el WARMUP_ACK trae la serie completa → deques correctos.
                # Mucho más rápido que el WarmupPool (1.5s vs 16s).
                if self._partial_sidecar:
                    log.info(
                        "[WarmupPool] Sidecar parcial — CM clásico para %s "
                        "(%d velas, skip WarmupPool).",
                        self._symbol, len(candles),
                    )
                    print(
                        f"[MainController] {self._symbol}: sidecar parcial — "
                        f"CM clásico ({len(candles)} velas, skip WarmupPool)"
                    )
                    self._using_warmup_pool = False
                    self._warmup_calculus_engine()
                    self._warmup_pool.start()  # pool vacío, cierra limpio
                    return

                # ── Normal path: calcular tensor completo ────────────────────
                lm_module = self._resolve_lm_module_path(self._symbol)
                lm_dir    = self._active_profiles.get(self._symbol, os.getcwd())
                self._warmup_pool.submit_token(
                    self._symbol, candles, lm_module, lm_dir
                )
                self._using_warmup_pool = True
                self._warmup_pool.start()
                # Transición visual: loading → calculando (0%)
                self.sig_token_progress.emit(self._symbol, 0)
                log.info(
                    "[WarmupPool] Pool iniciado. Token primario: %s (%d velas).",
                    self._symbol, len(candles),
                )
                return

        # Fallback: sin candles → flujo clásico
        log.warning("[WarmupPool] Sin candles para el pool — fallback a CM directo.")
        self._warmup_calculus_engine()

    def _on_pool_primary_ready(self, token_id: str, preheating_min: int) -> None:
        """
        Slot: tensor del token primario listo desde el WarmupPool.

        1. Inyecta tensor en MarketBufferHandler (batch, un solo lock).
        2. Construye _indicator_buffer deques en el MainController.
        3. Actualiza timestamps.
        4. Envía mini CMD_WARMUP al CalculusManager (skip_series=True).
           El CM procesa solo preheating_min velas para construir estado —
           el WARMUP_ACK resultante dispara Phase B (WS + timers).
        """
        if not self._warmup_pool:
            return
        result = self._warmup_pool.pop_tensor(token_id)
        if result is None:
            log.warning("[WarmupPool] Tensor primario vacío para %s.", token_id)
            return

        tensor, timestamps = result
        log.info(
            "[WarmupPool] Primario listo: %s | %d keys | %d puntos.",
            token_id, len(tensor), len(timestamps),
        )

        # 1. Inyectar en MarketBufferHandler (método batch — un solo lock)
        if self._buffer:
            self._buffer.load_indicator_series(token_id, tensor)

        # 2. Construir _indicator_buffer deques (deque.extend — nivel C)
        from collections import deque as _deque
        buffer_size = self._cfg.get("buffer_size", 3000)
        self._indicator_buffer = {}
        for output_key, arr in tensor.items():
            d = _deque(maxlen=buffer_size)
            d.extend(arr.tolist() if hasattr(arr, "tolist") else arr)
            self._indicator_buffer[output_key] = d

        # 3. Timestamps
        if len(timestamps) > 0:
            self._last_indicator_ts = int(timestamps[-1])
            self._last_rendered_ts  = self._last_indicator_ts
            self._indicator_ts_by_token[token_id] = self._last_indicator_ts
        print(
            f"[GW] POOL_INIT _last_rendered_ts={self._last_rendered_ts} "
            f"x_counter={self._canvas._x_counter if self._canvas else -1}"
        )

        # 3b. Persistencia del gap: appendear al .iend las filas del gap fill
        # que el preloader no pudo escribir (las velas REST llegaron antes
        # de que el PersistenceManager existiera).
        #
        # Condición: _primary_gap_filled > 0 → hay N velas al final del tensor
        # que no están en el .iend. Las appendeamos ahora usando los timestamps
        # del tensor y los valores de cada output_key para esas N filas.
        gap_n = self._primary_gap_filled
        if gap_n > 0 and self._persistence_manager and len(timestamps) >= gap_n:
            # Construir lookup ts → ohlcv desde el buffer (O(n) una sola vez)
            all_candles = self._buffer.get_candles(token_id) if self._buffer else []
            ts_to_candle = {int(c["timestamp"]): c for c in all_candles}

            gap_timestamps = timestamps[-gap_n:]
            gap_tensors    = {k: v[-gap_n:] for k, v in tensor.items()}
            n_written = 0
            for i, ts in enumerate(gap_timestamps):
                flat  = {k: float(v[i]) for k, v in gap_tensors.items()}
                ohlcv = ts_to_candle.get(int(ts), {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0})
                ok = self._persistence_manager.on_candle_close(
                    symbol          = token_id,
                    ts              = int(ts),
                    ohlcv           = ohlcv,
                    indicators_flat = flat,
                )
                if ok:
                    n_written += 1
            log.info(
                "[WarmupPool] Gap fill persistido: %s — %d/%d filas al .iend.",
                token_id, n_written, gap_n,
            )
            print(
                f"[MainController] Gap fill → .iend: {token_id} "
                f"{n_written}/{gap_n} filas appendeadas."
            )
            self._primary_gap_filled = 0

        # 4. Mini CMD_WARMUP → CM construye estado interno (no toca el buffer)
        if self._calculus_control_q and self._buffer:
            all_c      = self._buffer.get_candles(token_id)
            mini       = all_c[-preheating_min:] if all_c else []
            if mini and token_id not in self._known_calculus_tokens:
                self._calculus_control_q.put({
                    "type":        CMD_WARMUP,
                    "token_id":    token_id,
                    "candles":     mini,
                    "skip_series": True,    # ACK slim — tensor ya en buffer
                })
                self._known_calculus_tokens.add(token_id)
                log.info(
                    "[WarmupPool] Mini-warmup CM: %s (%d velas).",
                    token_id, len(mini),
                )
                print(
                    f"[MainController] Mini-warmup CM: {token_id} "
                    f"({len(mini)} velas — skip_series=True)"
                )

    def _on_pool_tensor_ready(self, token_id: str, preheating_min: int) -> None:
        """
        Slot: tensor de un token secundario listo desde el WarmupPool.

        1. Inyecta tensor en MarketBufferHandler (batch).
        2. Actualiza _indicator_ts_by_token.
        3. Envía mini CMD_WARMUP al CM (skip_series=True).
           El WARMUP_ACK resultante suscribe el token al WS y emite sig_token_ready.
        """
        if not self._warmup_pool:
            return
        result = self._warmup_pool.pop_tensor(token_id)
        if result is None:
            log.warning("[WarmupPool] Tensor secundario vacío para %s.", token_id)
            return

        tensor, timestamps = result
        log.info(
            "[WarmupPool] Secundario listo: %s | %d keys | %d puntos.",
            token_id, len(tensor), len(timestamps),
        )

        # 1. Inyectar en MarketBufferHandler (batch)
        if self._buffer:
            self._buffer.load_indicator_series(token_id, tensor)

        # 2. Timestamp
        if len(timestamps) > 0:
            self._indicator_ts_by_token[token_id] = int(timestamps[-1])
        else:
            last_c = self._buffer.get_last_candle(token_id) if self._buffer else None
            self._indicator_ts_by_token[token_id] = (
                int(last_c["timestamp"]) if last_c else 0
            )

        # 3. Mini CMD_WARMUP → CM construye estado del token secundario
        if self._calculus_control_q and self._buffer:
            all_c = self._buffer.get_candles(token_id)
            mini  = all_c[-preheating_min:] if all_c else []
            if mini and token_id not in self._known_calculus_tokens:
                self._calculus_control_q.put({
                    "type":        CMD_WARMUP,
                    "token_id":    token_id,
                    "candles":     mini,
                    "skip_series": True,
                })
                self._known_calculus_tokens.add(token_id)
                log.info(
                    "[WarmupPool] Mini-warmup CM secundario: %s (%d velas).",
                    token_id, len(mini),
                )

    def _on_pool_progress(self, token_id: str, current: int, total: int) -> None:
        pct = int(current / total * 100) if total else 0
        self.sig_token_progress.emit(token_id, pct)
        self.sig_status.emit(
            f"[WarmupPool] {token_id}: {current}/{total} ({pct}%)"
        )
        log.debug("[WarmupPool] %s: %d/%d (%d%%)", token_id, current, total, pct)

    def _on_pool_error(self, token_id: str, error: str) -> None:
        log.error("[WarmupPool] Error en %s: %s", token_id, error)
        self.sig_status.emit(f"[WarmupPool] Error {token_id}: {error}")

    def _reset_calculus_engine(self) -> None:
        """Envía CMD_RESET al CalculusManager durante switch_token()."""
        if not self._calculus_control_q:
            return
        self._calculus_control_q.put({
            "type": CMD_RESET,
            "token_id": self._symbol,
        })
        log.info("[Calculus] Reset enviado para %s.", self._symbol)

    def _shutdown_calculus_engine(self) -> None:
        """
        Cierre ordenado del CalculusManager durante stop().
        Timer ya detenido en el loop de timers de stop().
        """
        # Setear stop_event directamente — el proceso hijo lo chequea en
        # cada iteración del loop. Esto garantiza que salga incluso si la
        # Queue está bloqueada o el CMD_SHUTDOWN no llega a tiempo.
        if self._calculus_stop_event:
            try:
                self._calculus_stop_event.set()
            except Exception:
                pass

        # Mandar CMD_SHUTDOWN por Queue como señal formal adicional
        if self._calculus_control_q:
            try:
                self._calculus_control_q.put_nowait({"type": CMD_SHUTDOWN})
            except Exception:
                pass

        if self._calculus_engine:
            try:
                self._calculus_engine.join(timeout=5)
            except Exception:
                pass
            if self._calculus_engine.is_alive():
                try:
                    self._calculus_engine.terminate()
                    self._calculus_engine.join(timeout=2)
                    log.warning("[Calculus] Motor terminado forzosamente.")
                except Exception:
                    pass

        if self._calculus_pipe:
            try:
                self._calculus_pipe.close()
            except Exception:
                pass

        self._calculus_engine = None
        self._calculus_pipe = None
        self._calculus_control_q = None
        self._calculus_stop_event = None
        self._calculus_engine_pid = 0
        log.info("[Calculus] Motor de cálculo finalizado.")

    def _dispatch_tick_to_calculus(self, token_id: str, vela: Dict) -> None:
        """
        Envía un tick al CalculusManager por el Pipe.

        Invocado desde _on_candle() para todos los tokens con contexto
        activo en el CM (token primario y secundarios).

        token_id debe ser explícito — no se usa self._symbol para evitar
        que los ticks de tokens secundarios lleguen al CM con el
        token_id incorrecto y sus indicadores queden congelados.
        """
        if not self._calculus_pipe:
            return
        try:
            self._calculus_pipe.send({
                "type": MSG_TICK,
                "token_id": token_id,
                "candle": {
                    "timestamp": vela.get("timestamp", 0),
                    "open":      float(vela.get("open", 0)),
                    "high":      float(vela.get("high", 0)),
                    "low":       float(vela.get("low", 0)),
                    "close":     float(vela.get("close", 0)),
                    "volume":    float(vela.get("volume", 0)),
                },
            })
        except Exception as e:
            log.warning("[Calculus] Error enviando tick %s: %s", token_id, e)

    def _drain_calculus_pipe(self) -> None:
        """
        Drena mensajes del Pipe del CalculusManager.
        Patrón idéntico a _drain_ws_queue: poll no-bloqueante,
        máximo 20 mensajes por ciclo.
        """
        if not self._calculus_pipe:
            return

        drained = 0
        while drained < 20:
            try:
                if not self._calculus_pipe.poll(timeout=0):
                    break
                msg = self._calculus_pipe.recv()
                drained += 1
                self._handle_calculus_message(msg)
            except EOFError:
                log.warning("[Calculus] Pipe cerrado inesperadamente.")
                break
            except Exception as e:
                log.warning("[Calculus] Error drenando pipe: %s", e)
                break

    def _handle_calculus_message(self, msg: Dict) -> None:
        """Procesa un mensaje individual del CalculusManager."""
        msg_type = msg.get("type", "")

        if msg_type == MSG_WARMUP_ACK:
            self._on_warmup_ack(msg)

        elif msg_type == MSG_ACK:
            self._on_calculus_ack(msg)

        elif msg_type == MSG_PID_REPORT:
            self._calculus_engine_pid = msg.get("pid", 0)
            self._register_child_pid(self._calculus_engine_pid)
            log.info("[Calculus] PID del motor: %d", self._calculus_engine_pid)

        elif msg_type == MSG_EVENT:
            event     = msg.get("event", {})
            token_id  = msg.get("token_id", self._symbol)
            log.debug("[Calculus] Evento de LogicMaster token=%s: %s", token_id, event)

            # Dos formatos posibles del LM_008:
            #   - Apertura/alerta : tiene campo 'evento' = 'posicion_abierta'|'alerta'
            #   - Cierre          : NO tiene campo 'evento', tiene 'pct_change_movimiento'
            evento_tipo = event.get("evento", "")
            is_close    = not evento_tipo and "pct_change_movimiento" in event
            is_signal   = evento_tipo in (
                "posicion_abierta", "alerta", "posicion_actualizada"
            ) or is_close

            if is_signal:
                # Acumular en buffer para TODOS los tokens — permite
                # repintar marcadores históricos al hacer switch y
                # alimentar el módulo escritor JSONL.
                if self._buffer:
                    self._buffer.append_signal(token_id, event)

                # Enrutar al handler visual/operativo según token:
                # — Token activo: pintar en canvas + operativa automática.
                # — Token secundario: solo operativa automática si está
                #   habilitada (auto_trade), sin tocar el canvas.
                if token_id == self._symbol:
                    self._on_lm_signal(event)
                elif self._lm_auto_trade_enabled:
                    self._on_lm_signal_secondary(token_id, event)

        elif msg_type == MSG_STATUS:
            status = msg.get("status", "")
            if status == "warmup_progress":
                processed = msg.get("processed", 0)
                total = msg.get("total", 0)
                if total > 0 and processed % 1000 == 0:
                    log.info(
                        "[Calculus] Warmup: %d/%d (%.0f%%)",
                        processed, total, processed / total * 100,
                    )

        elif msg_type == MSG_ERROR:
            log.error("[Calculus] Error del motor: %s", msg.get("error", ""))

    # ==================================================================
    # LogicMaster — modos y sincronización de posición
    # ==================================================================

    def _load_token_modes(self) -> None:
        """
        Lee token_modes.json desde disco y popula _token_modes.
        Si el archivo no existe inicializa el dict vacío (todos los
        tokens quedan en modo "calc" por defecto).
        """
        self._token_modes = {}
        path = self._token_modes_path
        if not os.path.exists(path):
            log.info("[TokenModes] token_modes.json no encontrado — modo calc por defecto.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                valid = {"calc", "signal", "auto"}
                self._token_modes = {
                    k.upper(): v for k, v in data.items() if v in valid
                }
            log.info("[TokenModes] Modos cargados: %s", self._token_modes)
            print(f"[MainController] token_modes cargados: {self._token_modes}")
        except (json.JSONDecodeError, OSError) as e:
            log.warning("[TokenModes] Error leyendo token_modes.json: %s", e)

    def reload_token_modes(self, path: str = "") -> None:
        """
        Recarga token_modes.json en caliente.

        Llamado desde LM_GEX_P cuando el TokenModeDialog emite
        modes_saved — aplica la nueva configuración sin reiniciar.

        Si el token activo cambió de modo, actualiza _lm_signal_enabled
        y _lm_auto_trade_enabled inmediatamente para que los próximos
        eventos del LM se enruten con la configuración nueva.

        Parámetros
        ----------
        path : path al archivo guardado (opcional — si viene del
               diálogo se usa directamente; si no, usa _token_modes_path).
        """
        if path:
            self._token_modes_path = path
        self._load_token_modes()

        # Aplicar el modo del token activo en tiempo real
        mode = self._token_modes.get(self._symbol, "calc")
        signal_enabled = mode in ("signal", "auto")
        auto_enabled   = mode == "auto"
        self.set_lm_mode(
            signal_enabled     = signal_enabled,
            copilot_enabled    = self._lm_copilot_enabled,  # no cambia
            auto_trade_enabled = auto_enabled,
        )
        log.info(
            "[TokenModes] Recarga aplicada — %s → modo=%s "
            "signal=%s auto=%s",
            self._symbol, mode, signal_enabled, auto_enabled,
        )
        print(
            f"[MainController] reload_token_modes: {self._symbol} "
            f"modo={mode} signal={signal_enabled} auto={auto_enabled}"
        )

    def get_token_mode(self, symbol: str) -> str:
        """Retorna el modo configurado para un token ('calc'/'signal'/'auto')."""
        return self._token_modes.get(symbol.upper(), "calc")

    def set_lm_mode(
        self,
        signal_enabled:     bool = False,
        copilot_enabled:    bool = False,
        auto_trade_enabled: bool = False,
    ) -> None:
        """
        Configura el modo de operación del LogicMaster.

        Modos:
          Modo 1 — Calculadora pura (todos False):
              LM calcula indicadores y los renderiza. No pinta señales
              analíticas ni gestiona posición real.

          Modo 2 — Asistente analítico (signal_enabled=True):
              LM evalúa la lógica de apertura y pinta señales en canvas
              via OperativeManager. No sincroniza con posición real del
              exchange ni envía órdenes automáticas.

          Modo 3 — Copiloto / piloto automático:
              copilot_enabled=True  → sincroniza posición real con LM
                                      (LM gestiona SL/TP/trailing desde
                                       precio real de entrada).
              auto_trade_enabled=True → señales del LM fluyen al pipeline
                                        de órdenes (master_action_queue).

        Llamado desde LM_GEX_J cuando el usuario activa/desactiva
        los checkboxes de señal, copiloto y operativa automática.
        """
        prev_signal = self._lm_signal_enabled
        self._lm_signal_enabled     = signal_enabled
        self._lm_copilot_enabled    = copilot_enabled
        self._lm_auto_trade_enabled = auto_trade_enabled
        log.info(
            "[LM] Modo actualizado — señal=%s copiloto=%s auto=%s",
            signal_enabled, copilot_enabled, auto_trade_enabled,
        )
        om = getattr(self, "_operative_manager", None)
        if om is None:
            return
        if signal_enabled and not prev_signal:
            # Activar señal: pintar historial del token activo
            if self._buffer:
                signals = self._buffer.get_signals(self._symbol)
                if signals:
                    om.render_signal_history(signals)
        elif not signal_enabled and prev_signal:
            # Desactivar señal: limpiar marcadores del canvas
            om.clear_signal_markers()

    def _on_lm_signal(self, event: Dict) -> None:
        """
        Modo 2 y 3 — Procesa una señal analítica del LogicMaster.

        Recibe eventos de tipo posicion_abierta, alerta o
        posicion_actualizada emitidos por el LM vía MSG_EVENT.

        Responsabilidades:
          1. Si signal_enabled: delegar al OperativeManager para pintar
             el marcador en el canvas (estrella/triángulo según tipo).
          2. Si auto_trade_enabled: reenviar la señal al pipeline de
             órdenes (master_action_queue del proceso hijo).
             Solo aplica a eventos de apertura, no a alertas analíticas.

        El OperativeManager es referencia externa — se inyecta via
        set_operative_manager(). Si no está disponible, solo se loguea.
        """
        if not self._lm_signal_enabled:
            return

        evento_tipo = event.get("evento", "")
        log.info("[LM] Señal recibida: %s", evento_tipo)

        # --- Pintar en canvas ---
        om = getattr(self, '_operative_manager', None)
        if om is not None:
            try:
                om.on_lm_signal(event)
            except Exception as e:
                log.warning("[LM] Error despachando señal a OperativeManager: %s", e)

        # --- Operativa automática ---
        if not self._lm_auto_trade_enabled:
            return
        if evento_tipo != "posicion_abierta":
            return

        maq = getattr(self, '_master_action_queue', None)
        if maq is None:
            log.warning("[LM] auto_trade activo pero sin master_action_queue.")
            return

        try:
            tipo = event.get("tipo", "")
            side = "Buy" if tipo == "long" else "Sell"
            maq.put_nowait({
                "source": "LogicMaster",
                "info": {
                    "action": "open",
                    "token":  self._symbol,
                    "side":   side,
                    "sl":     event.get("sl"),
                    "tp":     event.get("tp"),
                    "codigo_modulo": event.get("codigo_modulo", ""),
                },
            })
            log.info(
                "[LM] Señal de apertura enviada al pipeline: %s %s",
                side, self._symbol,
            )
        except Exception as e:
            log.warning("[LM] Error enviando señal al pipeline de órdenes: %s", e)

    def _on_lm_signal_secondary(self, token_id: str, event: Dict) -> None:
        """
        Procesa una señal de apertura de un token secundario (no activo
        en el canvas) cuando auto_trade está habilitado.

        Solo envía al pipeline de órdenes — no toca el canvas ni el
        OperativeManager (eso es exclusivo del token activo).

        El marcador ya fue acumulado en el buffer por el caller
        (MSG_EVENT handler). Se pintará cuando el usuario haga
        switch a ese token y el canvas lo solicite.
        """
        if not self._lm_auto_trade_enabled:
            return

        evento_tipo = event.get("evento", "")
        if evento_tipo != "posicion_abierta":
            return

        maq = getattr(self, '_master_action_queue', None)
        if maq is None:
            log.warning(
                "[LM] auto_trade activo pero sin master_action_queue "
                "(token secundario: %s).", token_id
            )
            return

        try:
            tipo = event.get("tipo", "")
            side = "Buy" if tipo == "long" else "Sell"
            maq.put_nowait({
                "source": "LogicMaster",
                "info": {
                    "action": "open",
                    "token":  token_id,
                    "side":   side,
                    "sl":     event.get("sl"),
                    "tp":     event.get("tp"),
                    "codigo_modulo": event.get("codigo_modulo", ""),
                },
            })
            log.info(
                "[LM] Señal secundaria enviada al pipeline: %s %s",
                side, token_id,
            )
        except Exception as e:
            log.warning(
                "[LM] Error enviando señal secundaria al pipeline: %s", e
            )

    def _sync_lm_position(self, sync_data: Dict) -> None:
        """
        Modo 3 — Sincroniza la posición real del exchange con el LM.

        Llamado por LM_GEX_J cuando llega posicion_actualizada en la
        master_info_queue Y _lm_copilot_enabled es True.

        Envía MSG_SYNC al CalculusManager por el Pipe. El CM lo recibe
        en _handle_pipe_message y llama a logic_master.sync_active_position().

        No hace nada si el Pipe no está disponible o el modo copiloto
        está desactivado — el caller debería chequear el flag antes,
        pero esta función es defensiva.
        """
        if not self._lm_copilot_enabled:
            return
        if not self._calculus_pipe:
            log.warning("[LM] _sync_lm_position: Pipe no disponible.")
            return
        try:
            self._calculus_pipe.send({
                "type":      MSG_SYNC,
                "token_id":  self._symbol,
                "sync_data": sync_data,
            })
            log.info(
                "[LM] MSG_SYNC enviado para %s — precio=%.5f",
                self._symbol,
                float(sync_data.get("precio_entrada", 0) or 0),
            )
        except Exception as e:
            log.warning("[LM] Error enviando MSG_SYNC: %s", e)

    def _force_sync_lm(self, sync_data: Dict) -> None:
        """
        Sincronización manual — independiente del flag copiloto.

        Llamado por el botón de sincronización forzada de la UI.
        Bypasea _lm_copilot_enabled — el usuario decide explícitamente
        sincronizar en ese momento.

        Útil cuando el copiloto está desactivado pero se quiere actualizar
        el estado interno del LM una sola vez con el precio real actual.
        """
        if not self._calculus_pipe:
            log.warning("[LM] _force_sync_lm: Pipe no disponible.")
            return
        try:
            self._calculus_pipe.send({
                "type":      MSG_SYNC,
                "token_id":  self._symbol,
                "sync_data": sync_data,
            })
            log.info(
                "[LM] Sync forzado enviado para %s — precio=%.5f",
                self._symbol,
                float(sync_data.get("precio_entrada", 0) or 0),
            )
        except Exception as e:
            log.warning("[LM] Error en sync forzado: %s", e)

    def set_operative_manager(self, operative_manager) -> None:
        """
        Inyecta la referencia al OperativeManager para que _on_lm_signal
        pueda despachar señales visuales al canvas.
        Llamado desde LM_GEX_J después de instanciar el controller.
        """
        self._operative_manager = operative_manager

    def set_master_action_queue(self, queue) -> None:
        """
        Inyecta la master_action_queue del proceso hijo para que
        _on_lm_signal pueda enviar señales de operativa automática.
        Llamado desde LM_GEX_J cuando se crea o recrea la cola.
        """
        self._master_action_queue = queue

    def _on_warmup_ack(self, ack: Dict) -> None:
        """
        Procesa WARMUP_ACK del CalculusManager.

        Discrimina por token_id:
          - Token activo (self._symbol): inicializa _indicator_buffer,
            _key_to_base, _last_indicator_ts, dispara render.
          - Token secundario: acumula en el buffer persistente del
            MarketBufferHandler para que set_active_token lo use
            en un fast switch futuro — sin tocar _indicator_buffer
            ni disparar render.

        Pasos (solo para token activo):
          1. Construye _key_to_base desde 'indicators'.
          2. Inicializa deques desde 'indicators_series'.
          3. Inicializa _last_indicator_ts.
          4. Inspección y mapeo de indicadores.
          5. Dispara render batch inicial.
        """
        token_id   = ack.get("token_id", "")
        series     = ack.get("indicators_series", {})
        packed     = ack.get("indicators", {})
        timestamps = ack.get("timestamps", [])

        if not series:
            # is_mini=True: mini-warmup vía WarmupPool — no hay series en el ACK.
            # El tensor ya está en el buffer (inyectado por _on_pool_*_ready).
            # Continuar para: _key_to_base, _warmup_complete, render, Phase B.
            if ack.get("is_mini") and self._using_warmup_pool:
                pass   # continuar hacia el bloque del token activo
            else:
                log.warning("[Calculus] WARMUP_ACK sin series (token=%s).", token_id)
                if token_id == self._symbol:
                    self._warmup_complete = True
                return

        # --- Acumular en buffer persistente SOLO si no usamos WarmupPool ---
        # Con el pool, la inyección ya ocurrió en _on_pool_*_ready (batch).
        if not self._using_warmup_pool and self._buffer and series:
            self._buffer.clear_indicator_series(token_id)
            for output_key, values in series.items():
                for v in values:
                    self._buffer.append_indicator(token_id, output_key, v)

        # Inicializar el último timestamp conocido para este token.
        # Los ACKs incrementales posteriores usan este valor para
        # decidir update (misma vela) vs append (vela nueva).
        if timestamps:
            self._indicator_ts_by_token[token_id] = int(timestamps[-1])
        else:
            last_c = self._buffer.get_last_candle(token_id) if self._buffer else None
            self._indicator_ts_by_token[token_id] = int(last_c["timestamp"]) if last_c else 0

        # --- Token secundario: acumular, suscribir al WS y marcar listo ---
        if token_id != self._symbol:
            log.info(
                "[Calculus] Warmup ACK para %s acumulado en buffer "
                "persistente (token activo: %s, %d keys, %d puntos).",
                token_id, self._symbol, len(series),
                len(next(iter(series.values()), [])),
            )
            print(
                f"[MainController] Warmup secundario: {token_id} — "
                f"{len(series)} keys acumuladas en buffer persistente."
            )
            # Suscribir este token al WS activo (agrega sin reconectar)
            self._subscribe_token_to_ws(token_id)
            # Marcar como completamente listo y notificar a la UI
            self._fully_ready_tokens.add(token_id)
            self.sig_token_ready.emit(token_id)
            return

        # === De aquí en adelante: solo token activo (self._symbol) ===

        # --- 1. Construir _key_to_base ---
        self._key_to_base = {}
        for base_name, outputs in packed.items():
            for output_key in outputs:
                self._key_to_base[output_key] = base_name
        for output_key in series:
            if output_key not in self._key_to_base:
                self._key_to_base[output_key] = output_key

        # --- 2. Inicializar deques locales (solo si NO usamos WarmupPool) ---
        # Con el pool, _indicator_buffer ya fue construido en _on_pool_primary_ready.
        if not self._using_warmup_pool:
            buffer_size = self._cfg.get("buffer_size", 3000)
            self._indicator_buffer = {}
            for output_key, values in series.items():
                d = deque(maxlen=buffer_size)
                for v in values:
                    d.append(v)
                self._indicator_buffer[output_key] = d

        # --- 3. Inicializar _last_indicator_ts y _last_rendered_ts ---
        # Con el pool ya fueron inicializados en _on_pool_primary_ready.
        if not self._using_warmup_pool:
            if timestamps:
                self._last_indicator_ts = int(timestamps[-1])
            else:
                last_candle = self._buffer.get_last_candle(self._symbol)
                self._last_indicator_ts = (
                    int(last_candle["timestamp"]) if last_candle else 0
                )
            # _last_rendered_ts se inicializa al mismo valor que
            # _last_indicator_ts porque plot_full_chart ya pintó esa vela.
            self._last_rendered_ts = self._last_indicator_ts
            print(
                f"[GW] WARMUP_INIT _last_rendered_ts={self._last_rendered_ts} "
                f"x_counter={self._canvas._x_counter}"
            )

        self._warmup_complete = True
        log.info(
            "[Calculus] Warmup completo: %d output_keys, %d puntos.",
            len(self._indicator_buffer),
            len(next(iter(self._indicator_buffer.values()), [])),
        )
        print(
            f"[MainController] Warmup indicadores: {len(self._indicator_buffer)} keys | "
            f"x_counter={self._canvas._x_counter} | last_ts={self._last_indicator_ts}"
        )

        # --- 4. Inspección y mapeo de indicadores ---
        self._indicator_mapper.inspect_and_compile_matrix(packed)

        # --- 5. Render batch inicial (pipeline nuevo) ---
        QTimer.singleShot(0, self._render_all_indicators)

        # --- 6. Phase B: arrancar WS + timers ahora que el CM no está bloqueado ---
        # El WS solo arranca si aún no está activo — idempotente.
        if not self._ws_ready:
            QTimer.singleShot(0, self._do_start_websocket)

    def _persist_indicator_keys(self, packed: Dict) -> None:
        """
        Escribe last_indicators_keys.json solo si hay base_names nuevos
        o parámetros modificados. Evita escrituras innecesarias a disco.
        """
        import json as _json
        path = "last_indicators_keys.json"

        # Construir estructura actual: {base_name: sorted(output_keys)}
        current: Dict = {}
        for base_name, outputs in packed.items():
            current[base_name] = sorted(outputs.keys())

        # Leer archivo existente
        existing: Dict = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = _json.load(f)
        except (FileNotFoundError, Exception):
            pass

        # Comparar — solo escribir si hay diferencias
        if existing == current:
            log.debug("[Calculus] last_indicators_keys.json sin cambios — escritura omitida.")
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(current, f, indent=4)
            log.info("[Calculus] last_indicators_keys.json actualizado: %d base_names.", len(current))
            print(f"[MainController] last_indicators_keys.json → {len(current)} indicadores.")
        except Exception as e:
            log.warning("[Calculus] Error escribiendo last_indicators_keys.json: %s", e)

    def _on_calculus_ack(self, ack: Dict) -> None:
        """
        Único disparador gráfico autorizado.

        Discrimina por token_id:
          - Token activo: tres vías en deques + render atómico.
          - Token secundario: tres vías en buffer persistente, sin render.

        Ambas rutas usan update_or_append_indicator para mantener la
        serie persistente sincronizada 1:1 con las velas — sin duplicados
        por ticks intermedios de la vela en formación.
        """
        if not self._warmup_complete:
            print(f"[ACK-BLOCKED] warmup_complete=False token={ack.get('token_id','?')} ts={ack.get('timestamp',0)}")
            return

        flat = ack.get("indicators_flat", {})
        if not flat:
            return

        token_id = ack.get("token_id", self._symbol)
        ack_ts   = int(ack.get("timestamp", 0))

        # --- Token secundario: tres vías en buffer persistente, sin render ---
        if token_id != self._symbol:
            print(f"[ACK-SEC] token={token_id} sym={self._symbol} ts={ack_ts}")
            if self._buffer:
                last_ts = self._indicator_ts_by_token.get(token_id, 0)
                if ack_ts < last_ts:
                    return  # rezagado — descartar

                is_update = (ack_ts == last_ts)
                for output_key, value in flat.items():
                    self._buffer.update_or_append_indicator(
                        token_id, output_key, value, is_update,
                    )

                if ack_ts >= last_ts:
                    self._indicator_ts_by_token[token_id] = ack_ts
            return

        # === Token activo ===
        print(f"[ACK] token={token_id} ts={ack_ts} last_ind={self._last_indicator_ts} x={self._canvas._x_counter if self._canvas else -1} warmup={self._warmup_complete}")

        buffer_size = self._cfg.get("buffer_size", 3000)

        # --- Determinar si es update o append ---
        is_update = (ack_ts == self._last_indicator_ts)
        is_new    = (ack_ts > self._last_indicator_ts)

        # --- Tres vías en deques vs _last_indicator_ts ---
        for output_key, value in flat.items():
            if output_key not in self._indicator_buffer:
                self._indicator_buffer[output_key] = deque(maxlen=buffer_size)
                self._indicator_buffer[output_key].append(value)
                self._key_to_base.setdefault(output_key, output_key)
                # Primer valor — siempre append en persistente
                if self._buffer:
                    self._buffer.update_or_append_indicator(
                        self._symbol, output_key, value, False,
                    )
                continue

            d = self._indicator_buffer[output_key]

            if is_update:
                if len(d) > 0:
                    d[-1] = value
                else:
                    d.append(value)
            elif is_new:
                d.append(value)
            else:
                continue  # verdaderamente rezagado — descartar

            # Espejo en buffer persistente con la misma semántica
            if self._buffer:
                self._buffer.update_or_append_indicator(
                    self._symbol, output_key, value, is_update,
                )

        if ack_ts >= self._last_indicator_ts:
            self._last_indicator_ts = ack_ts
            self._indicator_ts_by_token[self._symbol] = ack_ts

        # --- Persistencia incremental: appendear vela cerrada al .ender + .iend ---
        # Solo cuando is_new (ack_ts > last_ts) — vela nueva confirmada.
        # El OHLCV se lee del buffer (última vela cargada) para no tener
        # que transmitirlo por el Pipe del CM.
        if is_new and self._persistence_manager:
            try:
                last_candle = self._buffer.get_last_candle(self._symbol) if self._buffer else None
                if last_candle and int(last_candle.get("timestamp", 0)) == ack_ts:
                    self._persistence_manager.on_candle_close(
                        symbol          = self._symbol,
                        ts              = ack_ts,
                        ohlcv           = last_candle,
                        indicators_flat = flat,
                    )
            except Exception as _pm_err:
                log.warning("[PM] Error en on_candle_close: %s", _pm_err)

        # --- Render de indicadores ---
        # Las velas ya las renderiza _on_candle via _gateway_render_candle.
        # El CM solo es responsable de los indicadores overlay.
        QTimer.singleShot(0, self._render_all_indicators)

        # GEX y OI — refrescar en el mismo ciclo
        QTimer.singleShot(0, self._visual_refresh_perp)
        QTimer.singleShot(0, self._visual_refresh_gex)

    def _render_indicators_from_series(
        self,
        symbol: str,
        indicator_series: Dict[str, List[float]],
    ) -> None:
        """
        Pinta la serie histórica de indicadores al cambiar de token.

        Recibe {output_key: [v0, v1, ..., vN]} del MarketBufferHandler
        (buffer persistente, llenado durante el warmup del CM).

        En vez de llamar a build_indicator_render_packets (legacy),
        pobla _indicator_buffer con los deques de la serie y delega
        a _render_all_indicators — el mismo pipeline que funciona
        para el render batch del warmup del token primario.

        También sincroniza _last_indicator_ts para que los ACKs
        incrementales posteriores continúen sin desfase.
        """
        if not self._canvas or not indicator_series:
            return

        try:
            # --- 1. Poblar _indicator_buffer desde la serie ---
            buffer_size = self._cfg.get("buffer_size", 3000)
            self._indicator_buffer = {}
            for output_key, values in indicator_series.items():
                d = deque(maxlen=buffer_size)
                for v in values:
                    d.append(v)
                self._indicator_buffer[output_key] = d

                # Asegurar que _key_to_base tenga la clave
                if output_key not in self._key_to_base:
                    self._key_to_base[output_key] = output_key

            # --- 2. Sincronizar _last_indicator_ts ---
            # SIEMPRE usar el timestamp de la última vela del buffer —
            # no el último ACK de background, que puede ser horas viejo.
            # Si _last_indicator_ts queda viejo, todos los ACKs posteriores
            # entran como is_new y hacen append infinito al deque.
            if self._buffer:
                last_candle = self._buffer.get_last_candle(symbol)
                if last_candle:
                    self._last_indicator_ts = int(last_candle["timestamp"])

            # --- 3. Render via pipeline existente ---
            self._render_all_indicators()

            log.info(
                "[MainController] Indicadores renderizados desde series: "
                "%s — %d keys, %d puntos.",
                symbol, len(indicator_series),
                len(next(iter(self._indicator_buffer.values()), [])),
            )
        except Exception as e:
            log.warning(
                "[MainController] _render_indicators_from_series falló: %s", e
            )

    def _render_indicator_buffer(self) -> None:
        """
        Agrupa deques por base_name usando _key_to_base.
        Construye paquete con estilos via build_indicator_render_packets.
        Despacha al canvas.
        """
        if not self._canvas or not self._canvas.main_plot:
            return
        if not self._indicator_buffer:
            return

        x_counter = self._canvas._x_counter

        # --- Agrupar por base_name ---
        # {base_name: {"keys": {output_key: y_arr}, "x": x_arr}}
        groups: Dict = {}
        for output_key, d in self._indicator_buffer.items():
            n = len(d)
            if n == 0:
                continue

            y_arr = np.array(
                [v if v is not None else np.nan for v in d],
                dtype=np.float64,
            )
            offset = x_counter - n + 1
            x_arr  = np.arange(offset, offset + n, dtype=np.float64)

            base_name = self._key_to_base.get(output_key, output_key)
            if base_name not in groups:
                groups[base_name] = {"keys": {}, "x": x_arr}
            groups[base_name]["keys"][output_key] = y_arr

        if not groups:
            return

        # --- Construir paquete con estilos ---
        packet = {}
        for base_name, group_data in groups.items():
            keys_dict = group_data["keys"]
            x_arr     = group_data["x"]
            indicators_by_base = {base_name: keys_dict}
            try:
                overlay_pkt, _ = build_indicator_render_packets(
                    indicators_by_base, x_arr, self._indicators_config,
                )
                packet.update(overlay_pkt)
            except Exception as e:
                log.warning("[Calculus] Error construyendo paquete para %s: %s", base_name, e)

        if packet:
            try:
                self._canvas.render_external_indicators(packet)
            except Exception as e:
                log.warning("[Calculus] Error en render: %s", e)

    def _render_all_indicators(self) -> None:
        """
        Pipeline nuevo: IndicatorPresentationManager bifurcado.

        Flujo:
          1. Computa x_array desde _x_counter (fuente de verdad del canvas).
          2. Pasa el caché plano de deques al PresentationManager.
          3. Despacha canvas_velas al canvas de velas.
          4. Despacha canvas_osciladores al router de osciladores.

        Red de seguridad:
          Si el pipeline nuevo falla por cualquier motivo, se desvía
          al legacy _render_indicator_buffer que pinta líneas blancas
          básicas sobre el canvas de velas.
        """
        if not self._canvas or not self._canvas.main_plot:
            return
        if not self._indicator_buffer:
            return

        # x_array estándar: desde _x_counter y longitud del deque representativo
        x_counter = self._canvas._x_counter
        ref_deque = next(iter(self._indicator_buffer.values()), None)
        if ref_deque is None or len(ref_deque) == 0:
            return

        ref_len = len(ref_deque)
        offset  = x_counter - ref_len + 1
        x_array = np.arange(offset, offset + ref_len, dtype=np.float64)

        try:
            result = self._presentation_manager.build_render_packets(
                self._indicator_buffer, x_array,
            )

            # --- Canvas de velas ---
            canvas_velas = result.get("canvas_velas", {})
            if canvas_velas:
                # Normalizar alpha: IndicatorPresentationManager emite float 0.0–1.0.
                # render_external_indicators espera int 0–255 para el formateo :02x
                # en LogicMasterCanvas (f"{alpha:02x}").
                # La corrección vive aquí — en la frontera entre las dos capas —
                # sin tocar ni el módulo de presentación ni el canvas.
                for _curve in canvas_velas.values():
                    _a = _curve.get("alpha", 1.0)
                    if isinstance(_a, float):
                        _curve["alpha"] = int(round(_a * 255))
                self._canvas.render_external_indicators(canvas_velas)

            # --- Canvas de osciladores ---
            canvas_osc = result.get("canvas_osciladores", {})
            if canvas_osc:
                self._route_oscillators_to_window(canvas_osc)

        except Exception as e:
            log.exception(
                "[Indicators] Pipeline nuevo falló — fallback legacy. Error:",
            )
            self._render_indicator_buffer()

    def _connect_vl_signal(self) -> None:
        """
        Conecta sigVerticalLinePlaced del canvas al slot _on_vl_placed.

        La señal se emite al colocar cualquier línea vertical (D o Shift+D).
        Firma: (x_index: int, ts_ms: int)

        Guard: solo conecta si el canvas existe y tiene la señal.
        Llamado desde mark_canvas_ready() una vez por sesión.
        """
        if not self._canvas:
            return
        try:
            self._canvas.sigVerticalLinePlaced.connect(self._on_vl_placed)
        except Exception as exc:
            log.warning("[VL] Error conectando sigVerticalLinePlaced: %s", exc)
        # Borrado total sincronizado: clear_all del canvas → oscilador
        try:
            self._canvas.sigDrawingsCleared.connect(self._on_drawings_cleared)
        except Exception as exc:
            log.warning("[VL] Error conectando sigDrawingsCleared: %s", exc)

    def _on_drawings_cleared(self) -> None:
        """
        Slot de borrado total. Disparado por clear_all_drawings del canvas.
        Borra todas las líneas verticales (efímeras + persistentes) del oscilador.
        """
        if not self._oscillator_window:
            return
        try:
            self._oscillator_window.clear_all_vertical_lines()
        except Exception as exc:
            log.warning("[VL] Error en clear_all_vertical_lines: %s", exc)

    def _on_vl_placed(self, x_index: int, ts_ms: int, persistent: bool) -> None:
        """
        Slot de línea vertical.

        Propaga la coordenada X al OscillatorWindow para dibujar la línea
        sincronizada en todos sus sub-plots, según el canal:
          - persistent=False (tecla D)     → draw_vertical_line (REEMPLAZA).
          - persistent=True  (Shift+D)     → add_persistent_line (ACUMULA).

        Parámetros:
            x_index    : índice de la vela en el eje X del canvas.
            ts_ms      : timestamp de apertura de esa vela en ms UTC.
            persistent : True si es línea persistente (Shift+D).
        """
        if not self._oscillator_window:
            return
        try:
            label = ""
            if ts_ms > 0:
                import pandas as _pd
                label = (
                    _pd.Timestamp(ts_ms, unit="ms", tz="UTC")
                    .tz_convert("America/Argentina/Buenos_Aires")
                    .strftime("%d/%m %H:%M")
                )
            if persistent:
                self._oscillator_window.add_persistent_line(
                    float(x_index), label_text=label or None,
                )
            else:
                self._oscillator_window.draw_vertical_line(
                    float(x_index), label_text=label or None,
                )
        except Exception as exc:
            log.warning("[VL] Error propagando línea vertical: %s", exc)

    def _route_oscillators_to_window(self, oscillator_packets: dict) -> None:
        """
        Despacha paquetes de osciladores al OscillatorWindow.

        Itera cada plot_id del dict emitido por
        IndicatorPresentationManager.build_render_packets() y llama:
          1. ensure_plot()   — idempotente, crea el sub-plot si no existe.
          2. render_curves() — limpia el render anterior y redibuja.

        Guards:
          - Si _oscillator_window es None, retorna sin hacer nada.
            El pipeline de velas no se ve afectado.
          - Si el paquete está vacío, retorna.
          - Cada plot_id se procesa dentro de un try/except individual
            para que un fallo en un indicador no bloquee los demás.
        """
        if not self._oscillator_window or not oscillator_packets:
            return

        for plot_id, packet in oscillator_packets.items():
            try:
                self._oscillator_window.ensure_plot(
                    plot_id,
                    y_range         = packet.get("y_range"),
                    reference_lines = packet.get("reference_lines"),
                )
                self._oscillator_window.render_curves(
                    plot_id,
                    packet.get("curves", {}),
                )
            except Exception as exc:
                log.warning(
                    "[Indicators] Error despachando '%s' al OscillatorWindow: %s",
                    plot_id, exc,
                )

    # ==================================================================
    # Placeholder para futuros datos de mercado spot
    # ==================================================================

    def _dispatch_spot_market(self, data: dict) -> None:
        """
        Placeholder para futuros diccionarios de volumen spot
        diferenciado del orderbook.
        """
        pass

    def _reset_indicator_buffer(self) -> None:
        """Limpia el buffer de indicadores y resetea el flag de warmup."""
        self._indicator_buffer.clear()
        self._warmup_complete    = False
        self._last_indicator_ts  = 0
        self._key_to_base.clear()
        self._last_rendered_price = 0.0
        self._last_rendered_ohlcv = {}
        if self._indicator_mapper:
            self._indicator_mapper.active_map.clear()
        log.info("[Calculus] Buffer de indicadores reseteado.")

    def reload_indicators_config(self, oscillator_window=None) -> None:
        """
        Recarga indicators_config.json en caliente y redibuja todos los
        indicadores con la nueva configuración.

        Invocado desde LM_GEX_T._on_config_requested() después de que
        el usuario cierra el diálogo de estilos con cambios.

        Secuencia:
          1. Recargar _presentation_manager desde disco.
          2. Si se pasó oscillator_window, destruir todos sus sub-plots
             (reset_all_plots) para que ensure_plot() los recree con
             los nuevos plot_ids y estilos.
          3. Disparar _render_all_indicators() vía QTimer para que corra
             en el hilo Qt con el buffer existente — sin necesidad de
             reiniciar la aplicación ni el WebSocket.

        Parámetros:
            oscillator_window : OscillatorWindow | None
                Si se pasa, se llama reset_all_plots() antes de redibujar.
                Si es None, solo se recargan estilos (sirve para cambios
                que no mueven indicadores entre plots).
        """
        # 1. Recargar config desde disco
        try:
            self._presentation_manager.reload_config()
            log.info("[Config] indicators_config.json recargado.")
            print("[MainController] indicators_config.json recargado.")
        except Exception as exc:
            log.warning("[Config] Error recargando indicators_config: %s", exc)

        # 2. Destruir sub-plots del OscillatorWindow (si se pasó)
        if oscillator_window is not None:
            try:
                oscillator_window.reset_all_plots()
                log.info("[Config] OscillatorWindow reseteado.")
            except Exception as exc:
                log.warning("[Config] Error reseteando OscillatorWindow: %s", exc)

        # 3. Redibujar indicadores desde el buffer existente
        if self._warmup_complete and self._indicator_buffer:
            QTimer.singleShot(0, self._render_all_indicators)
            log.info("[Config] Redibujado de indicadores programado.")
        else:
            log.info(
                "[Config] Sin buffer de indicadores activo — "
                "el redibujado ocurrirá en el próximo WARMUP_ACK."
            )

    def _trigger_canvas_refresh(self) -> None:
        """
        Timer de 1000ms — actualiza el countdown y refresca la vela
        en formación via gateway (misma vela, solo cambia el countdown).
        """
        if not self._is_running or not self._canvas:
            return
        ohlcv = self._last_rendered_ohlcv
        if not ohlcv:
            return

        o = ohlcv.get("o", 0)
        h = ohlcv.get("h", 0)
        l = ohlcv.get("l", 0)
        c = ohlcv.get("c", 0)
        v = ohlcv.get("v", 0)
        candle_ts = int(ohlcv.get("timestamp", 0))

        # Recalcular countdown con tiempo actual
        countdown_str = ""
        try:
            tf_str = str(self._timeframe)
            tf_minutes = 1440 if tf_str.upper() == "D" else int(tf_str)
            tf_ms  = tf_minutes * 60 * 1000
            now_ms = int(time.time() * 1000)
            remaining_ms = (candle_ts + tf_ms) - now_ms
            if remaining_ms > 0:
                remaining_s = remaining_ms // 1000
                mins, secs  = divmod(remaining_s, 60)
                countdown_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        except Exception:
            countdown_str = ""

        # Pasar por el gateway — misma vela, mismo timestamp
        self._gateway_render_candle(
            candle_ts, o, h, l, c, v, countdown_str, _caller="timer",
        )

    # ==================================================================
    # Centinela anti-zombis
    # ==================================================================

    def _register_child_pid(self, pid: int) -> None:
        """Registra un PID de proceso hijo para limpieza garantizada."""
        if pid and pid not in self._child_pids:
            self._child_pids.append(pid)
            log.debug("[Centinela] PID registrado: %d", pid)

    def _kill_children(self) -> None:
        """
        Mata todos los procesos hijos registrados que sigan vivos.
        Invocado por stop() y registrado en atexit como red de seguridad.
        Usa SIGTERM (cierre ordenado) seguido de SIGKILL si no muere.
        """
        for pid in list(self._child_pids):
            try:
                os.kill(pid, 0)  # Verificar si sigue vivo
            except OSError:
                self._child_pids.remove(pid)
                continue

            try:
                os.kill(pid, signal.SIGTERM)
                log.info("[Centinela] SIGTERM enviado a PID %d.", pid)

                deadline = time.time() + 2.0
                while time.time() < deadline:
                    try:
                        os.kill(pid, 0)
                        time.sleep(0.1)
                    except OSError:
                        break

                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                    log.warning("[Centinela] SIGKILL forzado a PID %d.", pid)
                except OSError:
                    pass

            except Exception as e:
                log.debug("[Centinela] Error limpiando PID %d: %s", pid, e)

            self._child_pids.remove(pid)

        if not self._child_pids:
            log.debug("[Centinela] Todos los procesos hijos limpiados.")