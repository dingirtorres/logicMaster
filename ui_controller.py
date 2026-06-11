# ui_controller.py
#
# Mediador pasivo entre el MainController (Backend/Data) y el Canvas
# de liquidez (Vista pasiva).
#
# Responsabilidades:
#   - Gestión del ciclo de vida del widget de visualización (Kill & Rebirth).
#   - Cálculo externo de rangos Y con padding del 1.5%.
#   - Protección de la soberanía del zoom manual del usuario.
#   - Centinela anti-zombis: monitoreo del PID padre con QTimer.
#   - Inyección de parámetros estéticos desde PresentationManager.
#
# Instanciación Binaria:
#   - Modo FULL (has_options=True): gestiona un LiquidityContainer (tríada).
#   - Modo LIGHT (has_options=False): gestiona un LiquidityDistributionCanvas
#     solitario. SensitivitiesCanvas y CoverageFlowCanvas NO EXISTEN en RAM.
#   - Prohibido el uso de setVisible(False) para simular ausencia.
#
# Lo que NO hace:
#   - No instancia ni destruye widgets — los recibe del MainController.
#   - No calcula GEX, no accede a la API, no persiste datos de mercado.
#   - No toma decisiones de presentación — consulta PresentationManager.

from __future__ import annotations

import logging
import os
import sys
from typing import Any, List, Optional, Tuple, Union

from PyQt6.QtCore import QObject, QTimer

log = logging.getLogger("ui_controller")


class UIController(QObject):
    """
    Controlador de presentación y ciclo de vida visual.

    Actúa como mediador entre el MainController y el Canvas activo,
    gestionando rangos Y, zoom, color de línea de precio y el
    protocolo Centinela anti-zombis.

    Polimorfismo por tipo de canvas:
      - LiquidityContainer     → modo FULL  (tríada completa)
      - LiquidityDistributionCanvas → modo LIGHT (canvas solitario)

    Los métodos internos detectan el tipo almacenado en self._canvas
    para invocar la API correspondiente sin romper el pipeline.
    """

    # Padding simétrico para el encuadre inicial del eje Y.
    RANGE_PADDING_PCT: float = 0.015

    # Intervalo del heartbeat del Centinela (ms).
    HEARTBEAT_INTERVAL_MS: int = 5000

    # TTL de seguridad antes de os._exit() en hard_shutdown (ms).
    HARD_SHUTDOWN_TTL_MS: int = 5000

    def __init__(
        self,
        parent_pid: int,
        parent: QObject = None,
    ):
        super().__init__(parent)

        # --- Canvas gestionado (polimórfico) ---
        # Puede ser LiquidityContainer (FULL) o
        # LiquidityDistributionCanvas (LIGHT).
        # None cuando no hay canvas vinculado.
        self._canvas: Any = None

        # --- Modo de operación ---
        # None = sin canvas vinculado.
        # True = FULL (tríada). False = LIGHT (solitario).
        self._has_options: Optional[bool] = None

        # --- Centinela ---
        self._parent_pid: int = parent_pid
        self._is_alive: bool = True

        self._heartbeat_timer: QTimer = QTimer(self)
        self._heartbeat_timer.setInterval(self.HEARTBEAT_INTERVAL_MS)
        self._heartbeat_timer.timeout.connect(self._check_parent_alive)
        self._heartbeat_timer.start()

        # --- Estado de zoom ---
        self._manual_zoom_active: bool = False
        self._last_y_min: float = 0.0
        self._last_y_max: float = 0.0

        # --- Estética ---
        self._price_line_color: tuple = (255, 220, 0, 200)

        log.info(
            "UIController instanciado. parent_pid=%d, heartbeat=%dms",
            self._parent_pid,
            self.HEARTBEAT_INTERVAL_MS,
        )

    # ==================================================================
    # Cálculo de Rango — Algoritmo del 1.5%
    # ==================================================================

    def calculate_y_range(
        self,
        dataset: Optional[List] = None,
        current_price: float = 0.0,
        orderbook_levels: Optional[List] = None,
    ) -> Tuple[float, float]:
        """
        Algoritmo de Rango Dinámico con pivote automático de fuente.

        Modo FULL (has_options=True):
          Fuente primaria: strikes del dataset de opciones.
          strike_min, strike_max → padding 1.5%.

        Modo LIGHT (has_options=False):
          No existen strikes. El cálculo pivota a:
          1. Niveles de precio del orderbook (si disponibles).
          2. Fallback: precio de mercado actual ± 1.5%.

        Retorna (y_min, y_max) con tolerancia incluida.
        Retorna (0.0, 0.0) si no hay datos suficientes.
        """
        pad = self.RANGE_PADDING_PCT

        # --- Modo FULL: strikes del dataset ---
        if self._has_options and dataset:
            strikes = [
                float(d.get("strike", 0))
                for d in dataset
                if d.get("strike") is not None and float(d.get("strike", 0)) > 0
            ]
            if strikes:
                s_min = min(strikes)
                s_max = max(strikes)
                margin = (s_max - s_min) * pad
                y_min = s_min - margin
                y_max = s_max + margin
                self._last_y_min = y_min
                self._last_y_max = y_max
                return (y_min, y_max)

        # --- Modo LIGHT (o FULL sin strikes válidos): orderbook ---
        if orderbook_levels:
            prices = [
                float(p) for p, _ in orderbook_levels
                if float(p) > 0
            ]
            if prices:
                p_min = min(prices)
                p_max = max(prices)
                margin = (p_max - p_min) * pad
                y_min = p_min - margin
                y_max = p_max + margin
                self._last_y_min = y_min
                self._last_y_max = y_max
                return (y_min, y_max)

        # --- Fallback absoluto: precio actual ± 1.5% ---
        if current_price > 0:
            y_min = current_price * (1.0 - pad)
            y_max = current_price * (1.0 + pad)
            self._last_y_min = y_min
            self._last_y_max = y_max
            return (y_min, y_max)

        return (0.0, 0.0)

    # ==================================================================
    # Aplicación de Rango al Canvas
    # ==================================================================

    def apply_sync_ranges(
        self,
        y_min: float,
        y_max: float,
        force: bool = False,
    ) -> None:
        """
        Inyecta el rango vertical Y en el canvas activo.

        No-op si:
          - No hay canvas vinculado.
          - manual_zoom_active es True y force es False.
          - y_min >= y_max (rango inválido).

        Modo FULL: invoca LiquidityContainer.set_sync_ranges().
        Modo LIGHT: invoca LiquidityDistributionCanvas.set_price_range()
                    + recalcula rango X del orderbook.
        """
        if not self._canvas:
            return
        if self._manual_zoom_active and not force:
            log.debug("apply_sync_ranges: bloqueado por manual_zoom_active.")
            return
        if y_min >= y_max:
            log.warning("apply_sync_ranges: rango inválido (%.2f >= %.2f).", y_min, y_max)
            return

        self._last_y_min = y_min
        self._last_y_max = y_max

        if self._has_options:
            # Modo FULL — LiquidityContainer
            self._canvas.set_sync_ranges(y_min, y_max)
        else:
            # Modo LIGHT — LiquidityDistributionCanvas solitario
            # set_price_range aplica su propio padding interno del 5%.
            # Pasamos el rango ya con el 1.5% calculado, así que
            # usamos la API de bajo nivel del ViewBox directamente.
            self._canvas.plot_liq.vb.disableAutoRange()
            self._canvas.plot_liq.setYRange(y_min, y_max, padding=0)
            self._canvas.vb_book.disableAutoRange()
            self._canvas.vb_book.setYRange(y_min, y_max, padding=0)

            # Recalcular rango X del orderbook desde _last_book_max
            book_max = self._canvas._last_book_max
            if book_max > 0:
                self._canvas.vb_book.setXRange(0, book_max * 1.02, padding=0)

        log.debug(
            "apply_sync_ranges: y_min=%.2f, y_max=%.2f, mode=%s",
            y_min, y_max, "FULL" if self._has_options else "LIGHT",
        )

    # ==================================================================
    # Inyección Inicial de Rango (Carga / Reset)
    # ==================================================================

    def initialize_ranges(
        self,
        dataset: Optional[List] = None,
        current_price: float = 0.0,
        orderbook_levels: Optional[List] = None,
    ) -> None:
        """
        Calcula y aplica el rango en un solo paso.
        Resetea manual_zoom_active a False.

        Invocado por MainController después de:
          - Primera carga de datos (load_history_sync).
          - Reset explícito del usuario.
          - Cambio de token (post attach_canvas + primer dataset).
        """
        self._manual_zoom_active = False
        y_min, y_max = self.calculate_y_range(
            dataset=dataset,
            current_price=current_price,
            orderbook_levels=orderbook_levels,
        )
        if y_min < y_max:
            self.apply_sync_ranges(y_min, y_max, force=True)

    # ==================================================================
    # Control de Zoom Manual
    # ==================================================================

    def set_manual_zoom(self, active: bool) -> None:
        """
        Activa/desactiva la persistencia de zoom del usuario.

        Cuando active=True, apply_sync_ranges() se convierte en no-op
        (excepto invocaciones con force=True).
        """
        self._manual_zoom_active = active
        log.debug("manual_zoom_active = %s", active)

    def reset_zoom(
        self,
        dataset: Optional[List] = None,
        current_price: float = 0.0,
        orderbook_levels: Optional[List] = None,
    ) -> None:
        """
        Reset explícito del usuario (botón / shortcut / doble clic).

        Desactiva manual_zoom_active.
        Recalcula y aplica el rango desde los datos provistos.
        """
        self.initialize_ranges(
            dataset=dataset,
            current_price=current_price,
            orderbook_levels=orderbook_levels,
        )

    # ==================================================================
    # Gestión de Color de Línea de Precio
    # ==================================================================

    def set_price_line_color(self, color: tuple) -> None:
        """
        Propaga el color RGBA a los paneles activos del canvas.

        Modo FULL: inyecta en los tres hijos del LiquidityContainer.
        Modo LIGHT: inyecta en el LiquidityDistributionCanvas solitario.
        """
        self._price_line_color = color

        if not self._canvas:
            return

        if self._has_options:
            # LiquidityContainer — los tres hijos
            self._canvas.sensitivities.set_price_line_color(color)
            self._canvas.coverage_flow.set_price_line_color(color)
            self._canvas.liquidity_distribution.set_price_line_color(color)
        else:
            # LiquidityDistributionCanvas solitario
            self._canvas.set_price_line_color(color)

        log.debug("price_line_color actualizado: %s", color)

    # ==================================================================
    # Ciclo de Vida — Kill & Rebirth
    # ==================================================================

    def destroy_canvas(self) -> None:
        """
        Protocolo de destrucción para cambio de activo.

        1. Detiene el heartbeat timer.
        2. Ejecuta close_cleanup() en el canvas activo para liberar
           objetos C++ (deleteLater en ViewBox esclavos) y vaciar
           datasets y buffers internos.
           - FULL: LiquidityContainer.close_cleanup() → los 3 hijos.
           - LIGHT: LiquidityDistributionCanvas.close_cleanup() → 1 hijo.
        3. Ejecuta clear_all() (FULL) o clear() (LIGHT) para limpiar
           curvas, tooltips y líneas de referencia.
        4. Desvincula la referencia interna.
        5. Resetea estado de zoom y rangos.

        El MainController espera el retorno de este método antes de
        instanciar el nuevo widget. Si destroy_canvas falla, el
        MainController actúa como segundo nivel. Si el MainController
        también falla, el Centinela ejecuta hard_shutdown.
        """
        self._heartbeat_timer.stop()

        if self._canvas is not None:
            try:
                self._canvas.close_cleanup()
            except Exception as e:
                log.warning("Error en close_cleanup: %s", e)

            try:
                if self._has_options:
                    self._canvas.clear_all()
                else:
                    self._canvas.clear()
            except Exception as e:
                log.warning("Error en clear: %s", e)

        self._canvas = None
        self._has_options = None
        self._manual_zoom_active = False
        self._last_y_min = 0.0
        self._last_y_max = 0.0

        log.info("destroy_canvas: canvas destruido y desvinculado.")

    def attach_canvas(
        self,
        new_canvas,
        has_options: bool,
    ) -> None:
        """
        Registra un nuevo canvas post-cambio de token.

        has_options=True (FULL):
          new_canvas es un LiquidityContainer que instanció los tres
          hijos internamente. Los tres existen en RAM.

        has_options=False (LIGHT):
          new_canvas es un LiquidityDistributionCanvas solitario.
          SensitivitiesCanvas y CoverageFlowCanvas NUNCA fueron
          instanciados — no existen en memoria.

        No aplica rangos — initialize_ranges() se invocará cuando
        el MainController entregue el primer dataset.
        """
        self._canvas = new_canvas
        self._has_options = has_options

        # Inyectar el color de línea de precio persistente
        self.set_price_line_color(self._price_line_color)

        # Reiniciar el heartbeat
        if not self._heartbeat_timer.isActive():
            self._heartbeat_timer.start()

        log.info(
            "attach_canvas: mode=%s, canvas=%s",
            "FULL" if has_options else "LIGHT",
            type(new_canvas).__name__,
        )

    # ==================================================================
    # Centinela — Heartbeat Inverso (Anti-Zombis)
    # ==================================================================

    def _check_parent_alive(self) -> None:
        """
        Callback del heartbeat timer (cada HEARTBEAT_INTERVAL_MS).

        Usa os.kill(parent_pid, 0) para verificar que el proceso
        padre sigue vivo.

        - Signal 0: no mata al proceso, solo verifica existencia.
        - Si OSError → el padre murió o fue adoptado por init/kernel.
        - En ese caso → ejecutar hard_shutdown() inmediatamente.
        """
        try:
            os.kill(self._parent_pid, 0)
        except OSError:
            log.critical(
                "Centinela: parent_pid=%d no responde. "
                "Ejecutando hard_shutdown.",
                self._parent_pid,
            )
            self.hard_shutdown()
        except Exception as e:
            log.warning("Centinela: error inesperado verificando PID: %s", e)

    def hard_shutdown(self) -> None:
        """
        Protocolo de cierre forzoso en 2 fases.

        Fase 1 — Limpieza de memoria:
          - close_cleanup() en el canvas activo para liberar objetos C++.
          - Detiene el heartbeat timer.
          - Setea _is_alive = False.

        Fase 2 — Finalización de proceso:
          - Programa un QTimer.singleShot con TTL de HARD_SHUTDOWN_TTL_MS.
          - Si el proceso no ha cerrado en ese tiempo → _force_exit()
            ejecuta os._exit(0) para garantizar liberación de RAM.
        """
        if not self._is_alive:
            return  # Ya en proceso de shutdown — evitar reentrada.

        self._is_alive = False
        self._heartbeat_timer.stop()

        # Fase 1: limpieza de memoria
        if self._canvas is not None:
            try:
                self._canvas.close_cleanup()
            except Exception as e:
                log.warning("hard_shutdown: error en close_cleanup: %s", e)
            self._canvas = None

        log.warning("hard_shutdown: Fase 1 completada. TTL=%dms", self.HARD_SHUTDOWN_TTL_MS)

        # Fase 2: TTL de seguridad
        QTimer.singleShot(self.HARD_SHUTDOWN_TTL_MS, self._force_exit)

    @staticmethod
    def _force_exit() -> None:
        """
        Último recurso tras el TTL de hard_shutdown.

        os._exit(0): cierre inmediato sin cleanup de Python.
        Garantiza que la RAM sea liberada al SO incluso si hay
        threads/callbacks bloqueados.
        """
        log.critical("_force_exit: TTL expirado. os._exit(0).")
        os._exit(0)

    # ==================================================================
    # Propiedades de consulta
    # ==================================================================

    @property
    def is_alive(self) -> bool:
        """Estado operativo del controlador."""
        return self._is_alive

    @property
    def has_canvas(self) -> bool:
        """True si hay un canvas vinculado."""
        return self._canvas is not None

    @property
    def has_options(self) -> Optional[bool]:
        """Modo de operación actual (True=FULL, False=LIGHT, None=sin canvas)."""
        return self._has_options

    @property
    def manual_zoom_active(self) -> bool:
        """True si el zoom manual del usuario está activo."""
        return self._manual_zoom_active

    @property
    def last_y_range(self) -> Tuple[float, float]:
        """Último rango Y aplicado (y_min, y_max)."""
        return (self._last_y_min, self._last_y_max)

    @property
    def price_line_color(self) -> tuple:
        """Color RGBA actual de la línea de precio."""
        return self._price_line_color