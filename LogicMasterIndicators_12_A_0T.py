# LogicMasterIndicators_A7.py
#NO ES LA GRAN COSA PERO ZAFA BASTANTE SI LA SABÉS USAR.
# ==============================================================================
# 0. DEPENDENCIAS EXTERNAS
# ==============================================================================
import traceback
import pandas as pd
import numpy as np
import threading
import logging
import re
from typing import Optional, Union, List, Dict, Any
from cachetools import LRUCache
import numba
from scipy.signal import find_peaks

# ==============================================================================
# 1. CABECERA Y CONFIGURACIÓN
# ==============================================================================
if not logging.getLogger('UnifiedIndicators').handlers:
    logging.basicConfig(level=logging.INFO, format='[UnifiedIndicators] %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('UnifiedIndicators')

# ==============================================================================
# 2. COMPONENTES DE SOPORTE Y SANITIZACIÓN
# ==============================================================================

def _sanitize_value(val: Any) -> Union[float, int, None]:
    """
    Sanitiza un valor individual para cumplir con el estándar JSONL:
    - Convierte NaN, Inf, -Inf a None.
    - Redondea floats a 12 decimales.
    - Preserva enteros.
    """
    if val is None:
        return None
    
    # Manejo de tipos numéricos de numpy y python
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return round(float(val), 12)
    
    if isinstance(val, (int, np.integer)):
        return int(val)
        
    return val

def _sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Aplica sanitización a todo un diccionario de resultados."""
    return {k: _sanitize_value(v) for k, v in data.items()}

@numba.jit(nopython=True)
def rolling_mean_abs_dev(data: np.ndarray, window: int) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_slice = data[i - window + 1 : i + 1]
        mean = np.mean(window_slice)
        abs_dev_sum = np.sum(np.abs(window_slice - mean))
        result[i] = abs_dev_sum / window
    return result

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()

 # <--- Asegúrate de importar esto arriba

def indicator_error_handler(output_keys_on_fail: list):
    """
    Maneja errores en métodos aplicados en __init__ (Bound Methods).
    IMPRIME EL ERROR EN CONSOLA y devuelve un diccionario de None (sanitizado).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Recuperar información de contexto (Clase y Método)
                instance = getattr(func, '__self__', None)
                cls_name = instance.__class__.__name__ if instance else "UnknownClass"
                method_name = func.__name__
                
                # LOGUEO EXPLÍCITO DEL ERROR
                error_msg = f"\n[CRITICAL INDICATOR ERROR] en {cls_name}.{method_name}:"
                error_msg += f"\nArgs: {args[1:] if len(args)>1 else 'No args'}" # Omitimos self
                error_msg += f"\nException: {str(e)}"
                error_msg += f"\nTraceback: {traceback.format_exc()}"
                
                # Usar el logger configurado o print si no hay logger
                if 'logger' in globals():
                    logger.error(error_msg)
                else:
                    print(error_msg) # Fallback a stdout
                
                # Devolver estructura vacía para no romper el orquestador
                return {key: None for key in output_keys_on_fail}
        return wrapper
    return decorator

    
def _translate_timeframe_alias(timeframe: str) -> str:
    """
    Traduce de forma robusta los alias de timeframe comunes al estándar moderno de pandas.
    """
    patterns = {
        r'(\d+)\s*(m|min|T)$': r'\1min',
        r'(\d+)\s*(h|hour)$': r'\1h',
        r'(\d+)\s*(d|day)$': r'\1D',
        r'(\d+)\s*(w|week)$': r'\1W',
    }
    
    original_timeframe = timeframe
    for pattern, replacement in patterns.items():
        if re.search(pattern, timeframe, re.IGNORECASE):
            return re.sub(pattern, replacement, timeframe, flags=re.IGNORECASE)
            
    return original_timeframe

class ResamplerCache:
    # maxsize: entradas del LRU cache. Con 40 activos y múltiples timeframes,
    # 128 se llena rápido. Default aumentado a 512 (activos * timeframes * margen).
    # Ajustar según cantidad de activos en producción: num_activos * num_timeframes * 2.
    def __init__(self, maxsize: int = 512):
        self._cache = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()

    def get_resampled_df(self, df_raw: pd.DataFrame, timeframe: Optional[str]) -> pd.DataFrame:
        if not timeframe or timeframe == '1min': return df_raw
        
        with self._lock:
            last_ts = df_raw['timestamp'].iloc[-1]
            cache_key = (last_ts, timeframe)
            if cache_key in self._cache: return self._cache[cache_key]

            # 1. Copia y Conversión temporal EFÍMERA (Solo para el motor de Pandas)
            df_working = df_raw.copy()
            df_working.index = pd.to_datetime(df_working['timestamp'], unit='ms')
            
            # 2. Ejecución real del remuestreo
            tf_corr = _translate_timeframe_alias(timeframe)
            agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            resampled_df = df_working.resample(tf_corr).agg(agg_dict).dropna()
            
            # 3. RESTAURACIÓN DE RUTA (FIX CRÍTICO)
            # Volvemos a milisegundos para que el 'period' se aplique a velas de 15m 
            # pero el índice sea compatible con el orquestador (int64)
            resampled_df['timestamp'] = (resampled_df.index.view(np.int64) // 10**6).astype(np.int64)
            resampled_df.index = resampled_df['timestamp']
            resampled_df.index.name = 'timestamp'

            self._cache[cache_key] = resampled_df
            return resampled_df
# ==============================================================================
# 2.1 HELPER FUNCTIONS PARA MODO BATCH (ENTRENAMIENTO)
# ==============================================================================

def _align_to_index(df_to_align: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    """
    Alinea resultados manteniendo la integridad de dtypes (int64).
    """
    if df_to_align.empty:
        return pd.DataFrame(index=target_index, columns=df_to_align.columns)

    # Si los índices ya son compatibles, reindexar directamente
    if df_to_align.index.dtype == target_index.dtype:
        return df_to_align.reindex(target_index, method='ffill')

    # Si hay colisión (datetime vs int64), forzar alineación por valores de timestamp
    # Se asume que el índice de df_to_align son los milisegundos reales
    try:
        return df_to_align.reindex(target_index, method='ffill')
    except TypeError:
        # Fallback: Si falla por tipos, devolvemos reindexado vacío para no romper el hilo
        # y que el desarrollador vea el error lógico sin crash
        logger.error("Fallo crítico de alineación: dtypes incompatibles en _align_to_index")
        return pd.DataFrame(index=target_index, columns=df_to_align.columns)

def batch_error_handler(output_is_dataframe: bool = False, cols: List[str] = None):
    """
    Maneja errores en métodos Batch aplicados en __init__.
    Devuelve pd.Series o pd.DataFrame de NaNs.
    """
    def decorator(func):
        def wrapper(df, *args, **kwargs):
            try:
                return func(df, *args, **kwargs)
            except Exception as e:
                instance = getattr(func, '__self__', None)
                cls_name = instance.__class__.__name__ if instance else "Unknown"
                logger.error(f"Error Batch en {cls_name}: {e}")
                
                if output_is_dataframe and cols:
                    return pd.DataFrame(np.nan, index=df.index, columns=cols)
                else:
                    return pd.Series(np.nan, index=df.index)
        return wrapper
    return decorator

_resampler_cache_instance = ResamplerCache()

# ==============================================================================
# 3. BLOQUE DE CLASES DE INDICADORES
# ==============================================================================

class SMA:
    def __init__(self, period: int = 20, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'SMA_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.Series(np.nan, index=df.index)
        result = sma(data_df['close'], self.period)
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class EMA:
    def __init__(self, period: int = 9, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'EMA_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.Series(np.nan, index=df.index)
        result = ema(data_df['close'], self.period)
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class DEMA:
    def __init__(self, period: int = 20, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'DEMA_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.Series(np.nan, index=df.index)
        ema1 = ema(data_df['close'], self.period)
        ema2 = ema(ema1, self.period)
        result = 2.0 * ema1 - ema2
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class TEMA:
    def __init__(self, period: int = 20, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'TEMA_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.Series(np.nan, index=df.index)
        ema1 = ema(data_df['close'], self.period)
        ema2 = ema(ema1, self.period)
        ema3 = ema(ema2, self.period)
        result = 3.0 * (ema1 - ema2) + ema3
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class BollingerBands:
    def __init__(self, period: int = 20, std_dev: float = 2.0, timeframe: Optional[str] = None):
        self.period = period
        self.std_dev = std_dev
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''

        
        self.base_name = f'BB_{self.period}_{self.std_dev}{timeframe_suffix}'

        self.output_keys = [
            f'{self.base_name}_Middle',
            f'{self.base_name}_Upper',
            f'{self.base_name}_Lower'
        ]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)
        middle_band = sma(data_df['close'], self.period)
        std = data_df['close'].rolling(window=self.period, min_periods=self.period).std()
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        result_df = pd.DataFrame({
            self.output_keys[0]: middle_band,
            self.output_keys[1]: upper_band,
            self.output_keys[2]: lower_band
        })
        return _align_to_index(result_df, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        full_df = self.calculate_series(df)
        return _sanitize_dict(full_df.iloc[-1].to_dict())

class RSI:
    def __init__(self, period: int = 14, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'RSI_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period + 1:
            return pd.Series(np.nan, index=df.index)
        delta = data_df['close'].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / self.period, adjust=False, min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, adjust=False, min_periods=self.period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        result = 100.0 - (100.0 / (1.0 + rs))
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class MACD:
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, timeframe: Optional[str] = None):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.base_name = f'MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}{timeframe_suffix}'
        
        self.output_keys = [
            f'{self.base_name}_Line',   # Antes variaba según periodo
            f'{self.base_name}_Signal', # Antes colisionaba si cambiabas fast/slow
            f'{self.base_name}_Hist'    # Antes colisionaba siempre
        ]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.slow_period:
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)
        ema_fast = ema(data_df['close'], self.fast_period)
        ema_slow = ema(data_df['close'], self.slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, self.signal_period)
        histogram = macd_line - signal_line
        result_df = pd.DataFrame({
            self.output_keys[0]: macd_line,
            self.output_keys[1]: signal_line,
            self.output_keys[2]: histogram
        })
        return _align_to_index(result_df, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        full_df = self.calculate_series(df)
        return _sanitize_dict(full_df.iloc[-1].to_dict())



##########################
## class Stochastic--->###############################################
##########################







class Stochastic:
    def __init__(self, k_period: int = 14, d_period: int = 3, timeframe: Optional[str] = None):
        self.k_period = k_period
        self.d_period = d_period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        # Lógica de Nomenclatura Estandarizada
        self.base_name = f'STOCH_{self.k_period}_{self.d_period}{timeframe_suffix}'
        
        self.output_keys = [
            f'{self.base_name}_K',
            f'{self.base_name}_D'
        ]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.k_period:
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)
        lowest_low = data_df['low'].rolling(window=self.k_period).min()
        highest_high = data_df['high'].rolling(window=self.k_period).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        stoch_k = 100.0 * (data_df['close'] - lowest_low) / denom
        stoch_k = stoch_k.clip(0.0, 100.0)
        stoch_d = sma(stoch_k, self.d_period)
        result_df = pd.DataFrame({
            self.output_keys[0]: stoch_k,
            self.output_keys[1]: stoch_d
        })
        return _align_to_index(result_df, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        full_df = self.calculate_series(df)
        return _sanitize_dict(full_df.iloc[-1].to_dict())



##########################
## class Stochastic--->###############################################
##########################        

class CCI:
    def __init__(self, period: int = 20, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'CCI_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.Series(np.nan, index=df.index)
        tp = (data_df['high'] + data_df['low'] + data_df['close']) / 3
        tp_sma = sma(tp, self.period)
        mean_dev_arr = rolling_mean_abs_dev(tp.to_numpy(), self.period)
        mean_dev = pd.Series(mean_dev_arr, index=tp.index)
        result = (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class ATR:
    def __init__(self, period: int = 14, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'ATR_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.Series(np.nan, index=df.index)
        high_low = (data_df['high'] - data_df['low']).abs()
        high_close = (data_df['high'] - data_df['close'].shift(1)).abs()
        low_close = (data_df['low'] - data_df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        result = ema(tr, self.period)
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class NATR:
    def __init__(self, period: int = 14, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'NATR_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.Series(np.nan, index=df.index)
        high_low = (data_df['high'] - data_df['low']).abs()
        high_close = (data_df['high'] - data_df['close'].shift(1)).abs()
        low_close = (data_df['low'] - data_df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        atr_series = ema(tr, self.period)
        result = (100.0 * atr_series / data_df['close'].replace(0, np.nan))
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})
        
class ADX:
    def __init__(self, period: int = 14, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        # Lógica de Nomenclatura Estandarizada
        self.base_name = f'ADX_{self.period}{timeframe_suffix}'
        
        self.output_keys = [
            f'{self.base_name}_Main',  # Valor del ADX
            f'{self.base_name}_Plus',  # DI+
            f'{self.base_name}_Minus'  # DI-
        ]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)
        up_move = data_df['high'].diff()
        down_move = -data_df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = pd.Series(plus_dm, index=data_df.index)
        minus_dm = pd.Series(minus_dm, index=data_df.index)
        high_low = (data_df['high'] - data_df['low']).abs()
        high_close = (data_df['high'] - data_df['close'].shift(1)).abs()
        low_close = (data_df['low'] - data_df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_adx = ema(tr, self.period)
        plus_dm_smooth = ema(plus_dm, self.period)
        minus_dm_smooth = ema(minus_dm, self.period)
        denom = atr_adx.replace(0, np.nan)
        plus_di = (100.0 * plus_dm_smooth / denom).clip(0.0, 100.0)
        minus_di = (100.0 * minus_dm_smooth / denom).clip(0.0, 100.0)
        dx_denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / dx_denom
        adx_series = ema(dx, self.period).clip(0.0, 100.0)
        result_df = pd.DataFrame({
            self.output_keys[0]: adx_series,
            self.output_keys[1]: plus_di,
            self.output_keys[2]: minus_di
        })
        return _align_to_index(result_df, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        full_df = self.calculate_series(df)
        return _sanitize_dict(full_df.iloc[-1].to_dict())
        
class OBV:
    def __init__(self, timeframe: Optional[str] = None):
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'OBV{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < 2:
            return pd.Series(np.nan, index=df.index)
        direction = np.sign(data_df['close'].diff().fillna(0.0))
        result = (data_df['volume'] * direction).fillna(0.0).cumsum()
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class MFI:
    def __init__(self, period: int = 14, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'MFI_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.Series(np.nan, index=df.index)
        tp = (data_df['high'] + data_df['low'] + data_df['close']) / 3.0
        mf = tp * data_df['volume']
        price_diff = tp.diff()
        pos_mf = mf.where(price_diff > 0, 0)
        neg_mf = mf.where(price_diff < 0, 0)
        pos_sum = pos_mf.rolling(window=self.period).sum()
        neg_sum = neg_mf.rolling(window=self.period).sum()
        money_ratio = pos_sum / neg_sum.replace(0, np.nan)
        mfi_series = 100.0 - (100.0 / (1.0 + money_ratio))
        mfi_series.loc[(pos_sum > 0) & (neg_sum == 0)] = 100.0
        mfi_series.loc[(pos_sum == 0) & (neg_sum == 0)] = 50.0
        result = mfi_series.clip(0.0, 100.0)
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class TRIX:
    def __init__(self, period: int = 30, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'TRIX_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period * 3:
            return pd.Series(np.nan, index=df.index)
        ema1 = ema(data_df['close'], self.period)
        ema2 = ema(ema1, self.period)
        ema3 = ema(ema2, self.period)
        result = 100 * (ema3.diff() / ema3.replace(0, np.nan))
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class WilliamsR:
    def __init__(self, period: int = 14, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'WILLR_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period:
            return pd.Series(np.nan, index=df.index)
        highest_high = data_df['high'].rolling(window=self.period).max()
        lowest_low = data_df['low'].rolling(window=self.period).min()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        result = -100.0 * (highest_high - data_df['close']) / denom
        result = result.clip(-100.0, 0.0)
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

class CMO:
    def __init__(self, period: int = 14, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.output_keys = [f'CMO_{self.period}{timeframe_suffix}']
        self.base_name = self.output_keys[0]
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period + 1:
            return pd.Series(np.nan, index=df.index)
        delta = data_df['close'].diff()
        sum_up = delta.clip(lower=0).rolling(window=self.period).sum()
        sum_down = delta.clip(upper=0).abs().rolling(window=self.period).sum()
        denom = (sum_up + sum_down).replace(0, np.nan)
        result = 100 * (sum_up - sum_down) / denom
        return _align_to_index(result, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})

# ==============================================================================
# CLASES PERSONALIZADAS
# =============================================================================
