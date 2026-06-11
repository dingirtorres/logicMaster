# LogicMasterIndicators_A7.py

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

##########################
## StochasticCinematico--->###############################################
##########################





##########################
#LogMACDDivergenceAngle---> ###############################################
##########################

class LogMacdDivergenceAngle:
    def __init__(
        self, 
        # Parámetros del Oscilador (MACD/PPO)
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9,
        
        # Parámetros de la Geometría del Precio (Curva Base + Tangente)
        ema_period: int = 5,             # Periodo de la curva base (Media Móvil) sobre la que se mide
        tangent_period_price: int = 14,  # Ventana de regresión para la tangente del precio
        
        # Parámetros de la Geometría del MACD (Tangente del Oscilador)
        tangent_period_macd: int = 10,   # Ventana de regresión para la tangente del MACD
        
        # Factores de Escala (Sensibilidad angular)
        scale_factor_price: float = 100.0, 
        scale_factor_macd: float = 1.0,
        
        # Infraestructura
        timeframe: Optional[str] = None
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        self.ema_period = ema_period
        self.tangent_period_price = tangent_period_price
        self.tangent_period_macd = tangent_period_macd
        
        self.scale_factor_price = scale_factor_price
        self.scale_factor_macd = scale_factor_macd
        self.timeframe = timeframe
        
        # Nomenclatura Estandarizada (Versión 7C)
        # Se incluyen los periodos estructurales para diferenciar instancias en el JSON de salida.
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.base_name = f'DIVA_LOG_MACD_{fast_period}_{slow_period}_{ema_period}_{tangent_period_price}_{tangent_period_macd}{timeframe_suffix}'
        
        self.output_keys = [
            f'{self.base_name}_Slope_Diff',  # Diferencia Angular (Divergencia)
            f'{self.base_name}_Price_Angle', # Ángulo de la Tangente del Precio
            f'{self.base_name}_MACD_Angle'   # Ángulo de la Tangente del MACD
        ]
        
        # Inyección de manejadores de errores
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_log_slope_static(data_arr, window, scale):
        """
        Calcula la pendiente de la regresión lineal sobre el LOGARITMO de los datos.
        Se utiliza para obtener el ángulo geométrico de la curva de precios.
        """
        n = len(data_arr)
        result = np.full(n, np.nan)
        
        # Pre-cálculo de términos constantes para la regresión
        x = np.arange(window)
        sum_x = np.sum(x)
        sum_x_sq = np.sum(x*x)
        denom = window * sum_x_sq - sum_x * sum_x
        
        # Log-transformación vectorizada (Numba maneja nans/inf implícitamente)
        log_data = np.log(data_arr)
        
        for i in range(window, n + 1):
            y_slice = log_data[i-window : i]
            # Fórmula de pendiente (m) por Mínimos Cuadrados Ordinarios
            slope = (window * np.sum(x * y_slice) - sum_x * np.sum(y_slice)) / denom
            # Conversión a grados
            result[i-1] = np.degrees(np.arctan(slope * scale))
        return result

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_linear_slope_static(data_arr, window, scale):
        """
        Calcula la pendiente de la regresión lineal sobre datos LINEALES.
        Se utiliza para obtener el ángulo del oscilador (MACD/PPO).
        """
        n = len(data_arr)
        result = np.full(n, np.nan)
        
        x = np.arange(window)
        sum_x = np.sum(x)
        sum_x_sq = np.sum(x*x)
        denom = window * sum_x_sq - sum_x * sum_x
        
        for i in range(window, n + 1):
            y_slice = data_arr[i-window : i]
            slope = (window * np.sum(x * y_slice) - sum_x * np.sum(y_slice)) / denom
            result[i-1] = np.degrees(np.arctan(slope * scale))
        return result

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Obtención y Resampleo (Si timeframe is None, es un bypass directo)
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        
        # Validación de longitud mínima requerida
        # Se necesita data suficiente para la EMA más lenta O para la ventana de regresión más larga
        min_len_indicators = self.slow_period 
        min_len_geometry = max(self.ema_period + self.tangent_period_price, self.tangent_period_macd)
        
        if len(data_df) < max(min_len_indicators, min_len_geometry):
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)

        # 2. DEFINICIÓN DE LA CURVA DE PRECIO (Base para la Tangente)
        # Desacoplamiento: Si ema_period > 1, usamos la curva de la EMA. Si es 1, usamos el precio crudo.
        if self.ema_period > 1:
            price_curve = ema(data_df['close'], self.ema_period).bfill().values
        else:
            price_curve = data_df['close'].astype(np.float64).values

        # 3. CÁLCULO DEL PPO (Percentage Price Oscillator)
        # Normalización del MACD: ((Fast - Slow) / Close) * 100
        ema_fast = ema(data_df['close'], self.fast_period)
        ema_slow = ema(data_df['close'], self.slow_period)
        
        ppo_series = ((ema_fast - ema_slow) / data_df['close']) * 100.0
        ppo_vals = ppo_series.fillna(0).values 

        # 4. CÁLCULO DE TANGENTES (Regresión Lineal vía Numba)
        
        # A. Tangente del Precio (Logarítmica sobre price_curve)
        angle_price = self._calculate_log_slope_static(
            price_curve, 
            self.tangent_period_price, 
            float(self.scale_factor_price)
        )
        
        # B. Tangente del MACD (Lineal sobre ppo_vals)
        angle_macd = self._calculate_linear_slope_static(
            ppo_vals, 
            self.tangent_period_macd, 
            float(self.scale_factor_macd)
        )

        # 5. CONSTRUCCIÓN Y SALIDA
        s_price_angle = pd.Series(angle_price, index=data_df.index)
        s_macd_angle = pd.Series(angle_macd, index=data_df.index)
        
        # Divergencia: Ángulo Oscilador - Ángulo Precio
        slope_diff = s_macd_angle - s_price_angle

        result_df = pd.DataFrame({
            self.output_keys[0]: slope_diff,
            self.output_keys[1]: s_price_angle,
            self.output_keys[2]: s_macd_angle
        }, index=data_df.index)

        return _align_to_index(result_df, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        full_df = self.calculate_series(df)
        return _sanitize_dict(full_df.iloc[-1].to_dict())


###############################################################




class LogDynamicSR:
    def __init__(self, period: int = 50, timeframe: Optional[str] = None):
        self.period = period
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        
        # Estándar de Nomenclatura 7C: Identificador LOG explícito + Params + Timeframe
        self.base_name = f'DSR_LOG_{self.period}{timeframe_suffix}'
        
        self.output_keys = [
            f'{self.base_name}_Dist_Res',
            f'{self.base_name}_Dist_Sup',
            f'{self.base_name}_Pressure'
        ]
        
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_dynamic_sr_numba(high, low, close, window):
        """
        Cálculo optimizado de Soportes/Resistencias Dinámicos y Presión.
        Complexity: O(N)
        """
        n = len(close)
        dist_res = np.full(n, np.nan)
        dist_sup = np.full(n, np.nan)
        pressure = np.full(n, np.nan)
        
        for i in range(window, n + 1):
            # Ventana deslizante para encontrar extremos locales
            w_high = high[i-window : i]
            w_low = low[i-window : i]
            
            curr_r = np.max(w_high)
            curr_s = np.min(w_low)
            curr_c = close[i-1] # Precio actual (último de la ventana)
            
            height = curr_r - curr_s
            
            # Protección contra rangos nulos o precios cero
            if height < 1e-9 or curr_c < 1e-9:
                dist_res[i-1] = 0.0
                dist_sup[i-1] = 0.0
                pressure[i-1] = 0.0
            else:
                # Distancia % a la Resistencia (usando Close como denominador)
                # Positivo: Precio por debajo de Resistencia
                dist_res[i-1] = (curr_r - curr_c) / curr_c * 100.0
                
                # Distancia % al Soporte (usando Close como denominador)
                # Positivo: Precio por encima de Soporte
                dist_sup[i-1] = (curr_c - curr_s) / curr_c * 100.0
                
                # Presión (-1 a 1): Posición relativa dentro del rango
                # 1.0 = Tocando Resistencia, -1.0 = Tocando Soporte, 0.0 = Justo en el medio
                mid = (curr_r + curr_s) / 2.0
                pressure[i-1] = (curr_c - mid) / (height / 2.0)
                
        return dist_res, dist_sup, pressure

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        
        # Validación de longitud mínima
        if len(data_df) < self.period:
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)
            
        # Preparación de datos (Casting explícito + bfill para seguridad numérica)
        h = data_df['high'].astype(np.float64).bfill().values
        l = data_df['low'].astype(np.float64).bfill().values
        c = data_df['close'].astype(np.float64).bfill().values
        
        # Ejecución Numba
        d_res, d_sup, press = self._calculate_dynamic_sr_numba(h, l, c, self.period)
        
        result_df = pd.DataFrame({
            self.output_keys[0]: d_res,
            self.output_keys[1]: d_sup,
            self.output_keys[2]: press
        }, index=data_df.index)
        
        return _align_to_index(result_df, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        full_df = self.calculate_series(df)
        return _sanitize_dict(full_df.iloc[-1].to_dict())


class RSITrendAngle:
    def __init__(self, rsi_period: int = 14, slope_period: int = 14, 
                 scale_factor: float = 1.0, timeframe: Optional[str] = None):
        """
        Calcula el ángulo de la tendencia del RSI mediante regresión lineal sobre una ventana móvil.
        Optimizado con Numba.
        
        Args:
            rsi_period: Periodo del RSI (Wilder's Smoothing).
            slope_period: Ventana de regresión lineal para calcular la pendiente.
            scale_factor: Factor multiplicador de la pendiente antes de calcular el ángulo (Sensibilidad).
            timeframe: Temporalidad de los datos.
        """
        self.rsi_period = rsi_period
        self.slope_period = slope_period
        self.scale_factor = scale_factor
        self.timeframe = timeframe
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        
        # Estándar de Nomenclatura 7C: Identificador + Params + Timeframe
        # Identificador: RSI_TA (RSI Trend Angle)
        self.base_name = f'RSI_TA_{self.rsi_period}_{self.slope_period}_{self.scale_factor}{timeframe_suffix}'
        
        # Estandarización de Salida: Sufijo _Angle
        self.output_keys = [f'{self.base_name}_Angle']
        
        # Inyección de manejadores de errores estandarizados (7E)
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_rsi_numba(close_arr, period):
        """
        Cálculo vectorizado del RSI (Wilder's Smoothing).
        Optimizado para ejecución en tiempo real.
        """
        n = len(close_arr)
        rsi = np.full(n, np.nan)
        
        if n <= period:
            return rsi
            
        # 1. Deltas iniciales
        deltas = np.diff(close_arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        # 2. Promedio inicial (SMA)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # 3. Primer valor RSI
        if avg_loss == 0:
            rsi[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100.0 - (100.0 / (1.0 + rs))
            
        # 4. Suavizado Wilder para el resto de la serie
        for i in range(period + 1, n):
            # Cambio respecto al anterior (i-1 vs i-2 en deltas indexado, aquí usamos lógica directa)
            change = close_arr[i] - close_arr[i-1]
            
            gain = change if change > 0 else 0.0
            loss = -change if change < 0 else 0.0
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
                
        return rsi

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_linear_slope_degrees(data_arr, window, scale):
        """
        Calcula el ángulo en grados de la pendiente lineal (OLS) sobre una ventana móvil.
        Formula: Degrees(Arctan(Slope * Scale))
        """
        n = len(data_arr)
        result = np.full(n, np.nan)
        
        # Pre-cálculo de constantes de regresión (X fijo)
        x = np.arange(window)
        sum_x = np.sum(x)
        sum_x_sq = np.sum(x*x)
        denom = window * sum_x_sq - sum_x * sum_x
        
        for i in range(window, n + 1):
            y_slice = data_arr[i-window : i]
            
            # Validación de NaNs en la ventana para evitar propagación silenciosa
            if np.isnan(y_slice).any():
                continue
                
            sum_y = np.sum(y_slice)
            sum_xy = np.sum(x * y_slice)
            
            # Pendiente (m)
            slope = (window * sum_xy - sum_x * sum_y) / denom
            
            # Conversión a Ángulo
            result[i-1] = np.degrees(np.arctan(slope * scale))
            
        return result

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        # 1. Obtención de datos con cache de resampleo
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        
        # Validación de longitud mínima
        if len(data_df) < self.rsi_period + self.slope_period:
            return pd.Series(np.nan, index=df.index)
            
        # 2. Preparación de datos (Casting estricto para Numba)
        close_values = data_df['close'].values.astype(np.float64)
        
        # 3. Cálculo del RSI (Base 0-100)
        rsi_values = self._calculate_rsi_numba(close_values, self.rsi_period)
        
        # 4. Cálculo del Ángulo sobre el RSI
        # Nota: RSI ya es lineal, no aplicamos logaritmos.
        angle_values = self._calculate_linear_slope_degrees(
            rsi_values, 
            self.slope_period, 
            float(self.scale_factor)
        )
        
        # 5. Alineación temporal con el DataFrame original
        return _align_to_index(pd.Series(angle_values, index=data_df.index), df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return _sanitize_dict({self.output_keys[0]: series.iloc[-1]})


# =============================================================================
# CLASE: CINEMÁTICA DEL PRECIO (Histórico / Close)
# =============================================================================
class LogPriceCinematics:
    """
    Calcula la cinemática del precio (Velocidad, Aceleración, Jerk) 
    usando transformaciones logarítmicas naturales para normalización porcentual intrínseca.
    
    Salidas:
        - Vel: Retorno Logarítmico (Velocidad instantánea).
        - Acc: Cambio en la velocidad (Aceleración).
        - Jerk: Cambio en la aceleración (Violencia del movimiento).
    """
    def __init__(self, timeframe: str = None):
        self.timeframe = timeframe
        self.base_name = f'CIN_LOG{("_" + timeframe) if timeframe else ""}'
        
        self.output_keys = [
            f'{self.base_name}_Vel',
            f'{self.base_name}_Acc',
            f'{self.base_name}_Jerk'
        ]
        # Inyección de manejadores de errores estándar (Asumidos en el contexto del archivo)
        # self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        # self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    def calculate(self, df: pd.DataFrame) -> dict:
        """
        Cálculo vectorizado sobre la serie completa de cierres.
        Devuelve solo el último estado (el del tick actual).
        """
        if df is None or df.empty or 'close' not in df.columns:
            return {k: None for k in self.output_keys}

        # Extracción de serie
        closes = df['close'].values.astype(np.float64)
        # Protección matemática
        closes = np.where(closes <= 0, 1e-9, closes)

        # 1. Posición Logarítmica
        log_p = np.log(closes)
        
        # 2. Velocidad (Diff 1)
        vel = np.diff(log_p, prepend=np.nan)
        
        # 3. Aceleración (Diff 2)
        acc = np.diff(vel, prepend=np.nan)
        
        # 4. Jerk (Diff 3)
        jerk = np.diff(acc, prepend=np.nan)
        
        return {
            f'{self.base_name}_Vel': _sanitize_value(vel[-1]),
            f'{self.base_name}_Acc': _sanitize_value(acc[-1]),
            f'{self.base_name}_Jerk': _sanitize_value(jerk[-1])
        }

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve la serie histórica completa."""
        if df is None or df.empty: return pd.DataFrame(columns=self.output_keys)

        closes = df['close'].values.astype(np.float64)
        closes = np.where(closes <= 0, 1e-9, closes)
        
        log_p = np.log(closes)
        vel = np.diff(log_p, prepend=np.nan)
        acc = np.diff(vel, prepend=np.nan)
        jerk = np.diff(acc, prepend=np.nan)
        
        res_df = pd.DataFrame(index=df.index)
        res_df[f'{self.base_name}_Vel'] = vel
        res_df[f'{self.base_name}_Acc'] = acc
        res_df[f'{self.base_name}_Jerk'] = jerk
        
        return res_df # _align_to_index(res_df, df.index)


# =============================================================================
# CLASE: GEOMETRÍA DE LA VELA (Histórico / OHLC / Firmas)
# =============================================================================
# =============================================================================
# CLASE: GEOMETRÍA DE LA VELA - CORREGIDA
# =============================================================================
# =============================================================================
# CLASE: GEOMETRÍA DE LA VELA - CORREGIDA
# =============================================================================

class LogCandleGeometry:
    def __init__(self, timeframe: str = None):
        self.timeframe = timeframe
        suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.base_name = f'GEO_LOG{suffix}'
        
        # Contrato de Salida Estricto
        self.output_keys = [
            f'{self.base_name}_Range_Polar',     # Polar: Alcista (+), Bajista (-)
            f'{self.base_name}_Body_Log',        # Polar: Neto del movimiento
            f'{self.base_name}_Efficiency_Osc',  # Oscilador: 0 a 100 con signo
            f'{self.base_name}_Upper_Shadow',    # Siempre >= 0
            f'{self.base_name}_Lower_Shadow'     # Siempre <= 0
        ]
        
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        
        o, h, l, c = [data_df[k].values.astype(np.float64) for k in ['open', 'high', 'low', 'close']]
        epsilon = 1e-12 
        o, h, l, c = [np.maximum(v, epsilon) for v in [o, h, l, c]]

        # 1. Cuerpo y Dirección
        body_log = np.log(c / o) * 100.0
        direction = np.sign(body_log)

        # 2. Rango Polar (Tu requisito: signo según la vela)
        range_mag = np.log(h / l) * 100.0
        range_polar = range_mag * direction

        # 3. Eficiencia como Oscilador 0-100 (Tu requisito visual)
        # Usamos las variables normalizadas o, h, l, c ya cargadas arriba
        epsilon_eff = 1e-9 
        
        # Calculamos la magnitud del rango y del cuerpo para la división segura
        range_mag_eff = np.abs(h - l) # Usando 'h' y 'l' que ya tienen el log o la escala
        body_abs_eff = np.abs(c - o)  # Usando 'c' y 'o'

        eff_raw = np.divide(
            body_abs_eff, 
            range_mag_eff, 
            out=np.zeros_like(body_abs_eff), 
            where=range_mag_eff > epsilon_eff
        )
        eff_osc = eff_raw * 100.0 * direction

        # 4. Mechas (Upper +, Lower -)
        u_sh = np.log(h / np.maximum(o, c)) * 100.0
        l_sh = -np.log(np.minimum(o, c) / l) * 100.0

        # Empaquetado con Alineación de Índices para evitar Look-ahead bias
        res_df = pd.DataFrame(index=data_df.index)
        res_df[self.output_keys[0]] = range_polar
        res_df[self.output_keys[1]] = body_log
        res_df[self.output_keys[2]] = eff_osc
        res_df[self.output_keys[3]] = u_sh
        res_df[self.output_keys[4]] = l_sh
        
        return _align_to_index(res_df, df.index)

    def calculate(self, df_or_dict: Union[pd.DataFrame, dict]) -> dict:
        if isinstance(df_or_dict, pd.DataFrame):
            full_df = self.calculate_series(df_or_dict)
            return _sanitize_dict(full_df.iloc[-1].to_dict())
        
        try:
            o, h, l, c = [float(df_or_dict[k]) for k in ['open', 'high', 'low', 'close']]
            o, h, l, c = [max(v, 1e-12) for v in [o, h, l, c]]
            
            body = np.log(c / o) * 100.0
            direction = np.sign(body)
            rng_mag = np.log(h / l) * 100.0
            
            vals = [
                rng_mag * direction,              # Range Polar
                body,                              # Body Log
                (abs(body) / rng_mag * 100.0 * direction) if rng_mag > 1e-12 else 0.0, # Eff Osc
                np.log(h / max(o, c)) * 100.0,     # Upper Shadow (+)
                -np.log(min(o, c) / l) * 100.0     # Lower Shadow (-)
            ]
            return _sanitize_dict({k: v for k, v in zip(self.output_keys, vals)})
        except:
            return {k: None for k in self.output_keys}




# =============================================================================
# CLASE: MICRO-ESTRUCTURA (Streaming / Ticks)
# =============================================================================
from collections import deque

class TickMicroStructure:
    """
    Sensor de flujo de ticks con memoria de corto plazo.
    Calcula métricas exclusivas del flujo que NO existen en OHLC histórico.
    NO calcula rangos ni geometrías de vela (eso es responsabilidad de LogCandleGeometry).
    """
    def __init__(self, buffer_size: int = 50, curvature_scale: float = 100.0):
        self.buffer_size = buffer_size
        self.curvature_scale = curvature_scale 
        self.tick_buffer = deque(maxlen=self.buffer_size)
        
        self.base_name = 'MICRO_TICK'
        self.output_keys = [
            f'{self.base_name}_SMA_{buffer_size}',
            f'{self.base_name}_Curvature_K',      
            f'{self.base_name}_Turn_Intensity'    
        ]

    def reset_state(self):
        """Reinicia el buffer."""
        self.tick_buffer.clear()

    def calculate(self, current_tick_price: float) -> dict:
        """
        Procesa un nuevo tick y devuelve métricas de flujo.
        """
        self.tick_buffer.append(current_tick_price)
        
        # A. Media Móvil de Ticks
        buffer_arr = np.array(self.tick_buffer, dtype=np.float64)
        tick_sma = np.mean(buffer_arr)
        
        # B. Curvatura y Giro
        curvature_k = 0.0
        turn_intensity = 0.0
        
        if len(buffer_arr) >= 3:
            # Curvatura de Menger (3 puntos finales)
            y1 = buffer_arr[-3]
            y2 = buffer_arr[-2]
            y3 = buffer_arr[-1]
            
            dy = (y3 - y1) / 2.0
            d2y = y3 - 2*y2 + y1
            
            denominator = (1 + dy**2) ** 1.5
            if denominator != 0:
                curvature_k = abs(d2y) / denominator
            
            # Normalización (0 a 1)
            turn_intensity = np.tanh(curvature_k * self.curvature_scale)

        return {
            f'{self.base_name}_SMA_{self.buffer_size}': _sanitize_value(tick_sma),
            f'{self.base_name}_Curvature_K': _sanitize_value(curvature_k),
            f'{self.base_name}_Turn_Intensity': _sanitize_value(turn_intensity)
        }

# ==============================================================================
# BLOQUE 2 — Clases incorporadas desde unificación.py
# Familia MaxMin: RelativeTrendAngle, TemporalTrendAcceleration, ComplexTrendAcceleration,
#                 RelativeAngleAcceleration
# Familia MACD:   MACDTrendAngle, OLSMACDTrendAngle, MACDTemporalAcceleration
# Familia Stoch:  StochasticKATrendAngle, FlatStochasticKATrendAngle,
#                 StochasticKATemporalAcceleration (V2 — modo tick)
# Familia Volumen: VolumeChangePercentage, VolumeTrendAngle, RVOLVolumeTrendAngle,
#                  LiteVolumeTrendAngle
# Otros: RSITrendAngleNorm (MaxMin), LogPolarCandleGeometry (renombrada de _1),
#        HarmonicBandPassSlope, DynamicSR
# ==============================================================================


# ──────────────────────────────────────────────────────────────────────────
# RelativeTrendAngle
# ──────────────────────────────────────────────────────────────────────────

class RelativeTrendAngle:
    def __init__(self, ema_period_1: int = 9, slope_period_1: int = 14, 
                 ema_period_2: int = 50, slope_period_2: int = 14,
                 normalize: bool = True, timeframe: Optional[str] = None):
        self.ema_period_1 = ema_period_1
        self.slope_period_1 = slope_period_1
        self.ema_period_2 = ema_period_2
        self.slope_period_2 = slope_period_2
        self.timeframe = timeframe
        self.normalize = normalize
        
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        norm_suffix = '_norm' if self.normalize else ''
        self.base_name = f'RTA_{self.ema_period_1}_{self.slope_period_1}_{self.ema_period_2}_{self.slope_period_2}{norm_suffix}{timeframe_suffix}'
        self.output_keys = [
            f'{self.base_name}_Slope_Diff',
            f'{self.base_name}_Slope_Ratio'
        ]
        
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    # Reutilizamos los métodos estáticos de regresión (copiar lógica si es necesario o asumir helpers globales)
    # Para robustez, los defino aquí como estáticos
    @staticmethod
    def _get_normalized_slope_static(y):
        y_min, y_max = np.min(y), np.max(y)
        denom = y_max - y_min
        if denom < 1e-9: return 0.0
        n = len(y)
        x_norm = np.linspace(0, 1, n)
        y_norm = (y - y_min) / denom
        return np.polyfit(x_norm, y_norm, 1)[0]

    @staticmethod
    def _get_absolute_slope_static(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        min_len = max(self.ema_period_1 + self.slope_period_1, self.ema_period_2 + self.slope_period_2)
        
        if len(data_df) < min_len:
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)
            
        # 1. Series Base
        ema_1 = ema(data_df['close'], self.ema_period_1)
        ema_2 = ema(data_df['close'], self.ema_period_2)
        
        # 2. Rolling Slopes
        target_func = self._get_normalized_slope_static if self.normalize else self._get_absolute_slope_static
        
        slope_1 = ema_1.rolling(window=self.slope_period_1).apply(target_func, raw=True)
        slope_2 = ema_2.rolling(window=self.slope_period_2).apply(target_func, raw=True)
        
        # 3. Métricas Derivadas
        slope_diff = slope_1 - slope_2
        
        # Ratio vectorizado con manejo de ceros (Lógica 91/-91)
        # B=0 -> 91.0, A=0 -> -91.0, A=0 y B=0 -> 0.0
        slope_ratio = slope_1 / slope_2.replace(0, np.nan) # División normal
        
        # Corrección de valores infinitos/NaN por división por cero
        # Nota: Pandas maneja division por cero como inf o nan. Ajustamos según tu lógica "centinela".
        # Para Batch simple, a veces es mejor dejar np.nan o inf, pero replicaré tu lógica exacta:
        
        # Mascara donde denominador es cero (o muy cercano)
        is_zero_2 = np.isclose(slope_2, 0.0)
        is_zero_1 = np.isclose(slope_1, 0.0)
        
        # Convertir a numpy para asignación rápida con np.where
        ratio_vals = slope_1.values / np.where(is_zero_2, 1e-9, slope_2.values) # Evitar crash temporal
        
        # Aplicar lógica centinela
        ratio_vals = np.where(is_zero_2 & ~is_zero_1, 91.0, ratio_vals)
        ratio_vals = np.where(is_zero_1 & ~is_zero_2, -91.0, ratio_vals)
        ratio_vals = np.where(is_zero_1 & is_zero_2, 0.0, ratio_vals)
        
        slope_ratio = pd.Series(ratio_vals, index=slope_1.index)

        result_df = pd.DataFrame({
            self.output_keys[0]: slope_diff,
            self.output_keys[1]: slope_ratio
        })
        return _align_to_index(result_df, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        full_df = self.calculate_series(df)
        return full_df.iloc[-1].to_dict()


# ──────────────────────────────────────────────────────────────────────────
# TemporalTrendAcceleration
# ──────────────────────────────────────────────────────────────────────────

class TemporalTrendAcceleration:
    def __init__(self, 
                 ema_period: int = 50, 
                 slope_period: int = 14, 
                 interval_n: int = 10,
                 normalize: bool = True,
                 timeframe: Optional[str] = None):
        
        self.ema_period = ema_period
        self.slope_period = slope_period
        self.interval_n = interval_n
        self.timeframe = timeframe
        self.normalize = normalize
        
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        norm_suffix = '_norm' if self.normalize else ''
        
        self.base_name = f'TTA_{self.ema_period}_{self.slope_period}_{self.interval_n}{norm_suffix}{timeframe_suffix}'
        self.output_keys = [
            f'{self.base_name}_Slope_Diff',
            f'{self.base_name}_Slope_Ratio'
        ]
        
        # Decoradores de seguridad
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    # --- Métodos Estáticos de Regresión (Requeridos para Rolling Apply) ---
    @staticmethod
    def _get_normalized_slope_static(y):
        y_min, y_max = np.min(y), np.max(y)
        denom = y_max - y_min
        if denom < 1e-9: return 0.0
        x_norm = np.linspace(0, 1, len(y))
        y_norm = (y - y_min) / denom
        return np.polyfit(x_norm, y_norm, 1)[0]

    @staticmethod
    def _get_absolute_slope_static(y):
        return np.polyfit(np.arange(len(y)), y, 1)[0]

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Resampling
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        min_len = self.ema_period + self.slope_period + self.interval_n
        
        if len(data_df) < min_len:
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)
            
        # 2. Cálculo de EMA
        ema_series = ema(data_df['close'], self.ema_period)
        
        # 3. Selección de función de pendiente
        target_func = self._get_normalized_slope_static if self.normalize else self._get_absolute_slope_static
        
        # 4. Rolling Slope (Ahora T=0)
        slope_now = ema_series.rolling(window=self.slope_period).apply(target_func, raw=True)
        
        # 5. Rolling Slope (Pasado T-n) -> Shift
        slope_past = slope_now.shift(self.interval_n)
        
        # 6. Métricas
        slope_diff = slope_now - slope_past
        
        # Ratio Vectorizado con manejo de ceros (Lógica 91/-91)
        is_zero_past = np.isclose(slope_past, 0.0)
        is_zero_now = np.isclose(slope_now, 0.0)
        
        ratio_vals = slope_now.values / np.where(is_zero_past, 1e-9, slope_past.values)
        ratio_vals = np.where(is_zero_past & ~is_zero_now, 91.0, ratio_vals)
        ratio_vals = np.where(is_zero_now & ~is_zero_past, -91.0, ratio_vals)
        ratio_vals = np.where(is_zero_now & is_zero_past, 0.0, ratio_vals)
        
        slope_ratio = pd.Series(ratio_vals, index=slope_now.index)

        result_df = pd.DataFrame({
            self.output_keys[0]: slope_diff,
            self.output_keys[1]: slope_ratio
        }, index=data_df.index)

        return _align_to_index(result_df, df.index)

    # Mantiene compatibilidad con el método calculate original (sin cambios en lógica interna)
    def _get_absolute_slope(self, y_values, x_axis): return np.polyfit(x_axis, y_values, 1)[0]
    def _get_normalized_slope(self, y_values, x_axis):
        y_min, y_max = y_values.min(), y_values.max()
        denom_y = y_max - y_min
        if denom_y < 1e-9: return 0.0
        y_norm = (y_values - y_min) / denom_y
        x_norm = (x_axis - x_axis[0]) / (x_axis[-1] - x_axis[0])
        return np.polyfit(x_norm, y_norm, 1)[0]
    def _calculate_ratio(self, a, b):
        if np.isclose(a, 0.0) and np.isclose(b, 0.0): return 0.0
        if np.isclose(a, 0.0): return -91.0
        if np.isclose(b, 0.0): return 91.0
        return a / b

    def calculate(self, df: pd.DataFrame) -> dict:
        # (Tu código calculate existente se mantiene igual, solo asegúrate de que esté identado aquí)
        # Para ahorrar espacio asumo que mantienes tu método calculate original aquí.
        # ... lógica de calculate ...
        # Si necesitas el código completo de calculate pídemelo, pero con calculate_series es suficiente para el fix.
        # IMPLEMENTACIÓN MÍNIMA DE CALCULATE PARA QUE NO DE ERROR SI LO LLAMAN:
        full_df = self.calculate_series(df)
        return full_df.iloc[-1].to_dict()



# ──────────────────────────────────────────────────────────────────────────
# StochasticKATrendAngle
# ──────────────────────────────────────────────────────────────────────────

class StochasticKATrendAngle:
    def __init__(self, k_period: int = 14, slope_period: int = 9, normalize: bool = True, timeframe: Optional[str] = None):
        self.k_period = k_period
        self.slope_period = slope_period
        self.timeframe = timeframe
        self.normalize = normalize
        
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        norm_suffix = '_norm' if self.normalize else ''
        self.base_name = f'STOCHKTA_{self.k_period}_{self.slope_period}{norm_suffix}{timeframe_suffix}'
        self.output_keys = [f'{self.base_name}_slope']
        
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    @staticmethod
    def _get_normalized_slope_static(y):
        y_min, y_max = np.min(y), np.max(y)
        denom = y_max - y_min
        if denom < 1e-9: return 0.0
        x_norm = np.linspace(0, 1, len(y))
        y_norm = (y - y_min) / denom
        return np.polyfit(x_norm, y_norm, 1)[0]

    @staticmethod
    def _get_absolute_slope_static(y):
        return np.polyfit(np.arange(len(y)), y, 1)[0]

    def _calculate_stoch_k_batch(self, df):
        lowest_low = df['low'].rolling(window=self.k_period).min()
        highest_high = df['high'].rolling(window=self.k_period).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        stoch_k = 100.0 * (df['close'] - lowest_low) / denom
        return stoch_k.clip(0.0, 100.0)

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.k_period + self.slope_period:
            return pd.Series(np.nan, index=df.index)
        
        stoch_k = self._calculate_stoch_k_batch(data_df)
        
        target_func = self._get_normalized_slope_static if self.normalize else self._get_absolute_slope_static
        slope_series = stoch_k.rolling(window=self.slope_period).apply(target_func, raw=True)
        
        return _align_to_index(slope_series, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return {self.output_keys[0]: series.iloc[-1]}


# ──────────────────────────────────────────────────────────────────────────
# FlatStochasticKATrendAngle
# ──────────────────────────────────────────────────────────────────────────

class FlatStochasticKATrendAngle:
    """
    Calcula la pendiente de la línea %K del Estocástico.
    
    Lógica:
    - Normalización intrínseca (0-100).
    - Usa Regresión Lineal Directa.
    - Pendiente alta en zona extrema (>80) = Momentum de ruptura ("Power Trend").
    - Pendiente girando en zona extrema = Reversión inminente.
    """
    def __init__(self, k_period: int = 14, slope_period: int = 9, scale_factor: float = 1.0, timeframe: Optional[str] = None):
        self.k_period = k_period
        self.slope_period = slope_period
        self.scale_factor = scale_factor
        self.timeframe = timeframe
        
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.base_name = f'STOCHK_Angle_{self.k_period}_{self.slope_period}{timeframe_suffix}'
        self.output_keys = [f'{self.base_name}_slope']
        
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=False)(self.calculate_series)

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_stoch_slope_numba(high, low, close, k_period, window, scale):
        n = len(close)
        result = np.full(n, np.nan)
        
        # Pre-calcular %K
        # No podemos usar rolling_max de pandas, hacemos ventana deslizante numba manual o simplificada
        # Para eficiencia en bucle único:
        stoch_k = np.full(n, np.nan)
        
        for i in range(k_period, n):
            # Ventana actual
            w_high = high[i-k_period+1 : i+1]
            w_low = low[i-k_period+1 : i+1]
            
            h_max = np.max(w_high)
            l_min = np.min(w_low)
            denom = h_max - l_min
            
            if denom == 0:
                stoch_k[i] = 50.0 # Centinela neutral si Flat
            else:
                stoch_k[i] = 100.0 * (close[i] - l_min) / denom
        
        # Regresión sobre %K
        x = np.arange(window)
        sum_x = np.sum(x)
        sum_x_sq = np.sum(x*x)
        denom_reg = window * sum_x_sq - sum_x * sum_x
        
        for i in range(window + k_period, n + 1):
            y_slice = stoch_k[i-window : i]
            # Validar nan en slice
            if np.isnan(y_slice[0]): continue
                
            slope = (window * np.sum(x * y_slice) - sum_x * np.sum(y_slice)) / denom_reg
            result[i-1] = np.degrees(np.arctan(slope * scale))
            
        return result

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.k_period + self.slope_period:
            return pd.Series(np.nan, index=df.index)
            
        h = data_df['high'].values.astype(np.float64)
        l = data_df['low'].values.astype(np.float64)
        c = data_df['close'].values.astype(np.float64)
        
        slope_series = self._calculate_stoch_slope_numba(
            h, l, c, 
            self.k_period, 
            self.slope_period, 
            float(self.scale_factor)
        )
        
        return _align_to_index(pd.Series(slope_series, index=data_df.index), df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        series = self.calculate_series(df)
        return {self.output_keys[0]: series.iloc[-1]}


# ──────────────────────────────────────────────────────────────────────────
# HarmonicBandPassSlope
# ──────────────────────────────────────────────────────────────────────────

class HarmonicBandPassSlope:
    """
    Aísla el 'Formante' (Ciclo Dominante) del precio usando un Filtro Pasa-Banda DSP 
    y calcula su velocidad de cambio (Pendiente).

    Arquitectura:
    1. Filtro IIR de 2 Polos: Elimina ruido (High Freq) y Tendencia DC (Low Freq).
    2. Normalización %: Convierte la onda a términos relativos al precio.
    3. Regresión Lineal: Mide la velocidad de la onda.

    Salida:
    - Harmonic_Wave_Pct: La posición de la onda (% respecto al precio).
      (Cruce de 0 = Cruce de la media ideal sin lag).
    - Harmonic_Slope: La velocidad de la onda.
      (Pico/Valle de la onda = Pendiente 0 = Giro de mercado).
    """
    def __init__(self, period: int = 20, bandwidth: float = 0.3, slope_period: int = 5, scale_factor: float = 100.0, timeframe: Optional[str] = None):
        """
        Args:
            period: La longitud de onda central a aislar (ej. 20 velas).
            bandwidth: Ancho de banda del filtro (Delta). 
                       0.1 = Filtro muy selectivo (Onda pura, tarda en reaccionar).
                       0.3 = Estándar (Balance reactividad/ruido).
            slope_period: Ventana pequeña para medir la pendiente de la onda (ej. 3-5).
        """
        self.period = period
        self.bandwidth = bandwidth
        self.slope_period = slope_period
        self.scale_factor = scale_factor
        self.timeframe = timeframe
        
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.base_name = f'H_Wave_{self.period}_{self.bandwidth}_{self.slope_period}{timeframe_suffix}'
        
        self.output_keys = [
            f'{self.base_name}_Val',   # Valor de la onda (% del precio)
            f'{self.base_name}_Slope'  # Pendiente de la onda (Velocidad de giro)
        ]
        
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_harmonic_dsp(close_arr, period, bandwidth, window, scale):
        n = len(close_arr)
        wave = np.zeros(n, dtype=np.float64)
        slope_res = np.full(n, np.nan)
        
        # --- 1. Coeficientes del Filtro Pasa-Banda (2-Pole) ---
        # Basado en aproximación estándar para series financieras (Ehlers/DSP)
        # L1 = cos(2*pi / period)
        # L2 = sin(2*pi / period)
        # Pero usamos la formulación estable de BandPass:
        
        # Frecuencia angular normalizada
        f = (2.0 * np.pi) / period
        
        # Beta (Factor de amortiguamiento relacionado al ancho de banda)
        beta = np.cos(f)
        gamma = 1.0 / np.cos(f * bandwidth)
        alpha = gamma - np.sqrt(gamma * gamma - 1.0)
        
        # Coeficientes de la ecuación en diferencias
        # y[i] = 0.5*(1-alpha)*(x[i] - x[i-2]) + beta*(1+alpha)*y[i-1] - alpha*y[i-2]
        c0 = 0.5 * (1.0 - alpha)
        c1 = beta * (1.0 + alpha)
        c2 = -alpha
        
        # --- 2. Filtrado IIR ---
        # Requiere historial, iniciamos con ceros (warm-up)
        for i in range(2, n):
            # Filtro aplicado sobre el precio raw
            wave[i] = c0 * (close_arr[i] - close_arr[i-2]) + c1 * wave[i-1] + c2 * wave[i-2]
            
        # --- 3. Normalización y Pendiente ---
        # La onda está en precio. Dividimos por Close para tener %
        # Luego regresión lineal sobre la onda normalizada
        
        x = np.arange(window)
        sum_x = np.sum(x)
        sum_x_sq = np.sum(x*x)
        denom = window * sum_x_sq - sum_x * sum_x
        
        for i in range(window, n + 1):
            # Normalización Local (Wave / Close) * 100
            # Evitamos división por cero
            c_slice = close_arr[i-window : i]
            w_slice = wave[i-window : i]
            
            # Vectorizado: Wave Pct
            # Si el precio es 0 (improbable), protegemos
            w_pct = np.empty(window, dtype=np.float64)
            for k in range(window):
                if c_slice[k] > 1e-9:
                    w_pct[k] = (w_slice[k] / c_slice[k]) * 100.0
                else:
                    w_pct[k] = 0.0
            
            # Guardamos el último valor de la onda como output 1
            if i == n: # Solo necesito el último para el return en modo serie? No, todo el array
                pass 
            
            # Slope sobre w_pct
            sum_y = np.sum(w_pct)
            sum_xy = np.sum(x * w_pct)
            slope = (window * sum_xy - sum_x * sum_y) / denom
            
            slope_res[i-1] = np.degrees(np.arctan(slope * scale))
        
        # Normalizamos el array de onda completo para devolverlo también
        final_wave_pct = np.zeros(n)
        for i in range(n):
            if close_arr[i] > 1e-9:
                final_wave_pct[i] = (wave[i] / close_arr[i]) * 100.0
                
        return final_wave_pct, slope_res

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.period + self.slope_period:
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)
            
        close_vals = data_df['close'].astype(np.float64).values
        
        wave_vals, slope_vals = self._calculate_harmonic_dsp(
            close_vals, 
            self.period, 
            self.bandwidth, 
            self.slope_period, 
            float(self.scale_factor)
        )
        
        result_df = pd.DataFrame({
            self.output_keys[0]: wave_vals,
            self.output_keys[1]: slope_vals
        }, index=data_df.index)
        
        return _align_to_index(result_df, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        full_df = self.calculate_series(df)
        return full_df.iloc[-1].to_dict()


# ──────────────────────────────────────────────────────────────────────────
# DynamicSR
# ──────────────────────────────────────────────────────────────────────────

class DynamicSR:
    """
    Detecta los niveles de Soporte y Resistencia más recientes dentro de
    una ventana deslizante, basado en la prominencia estructural de picos y valles.
    Utiliza la lógica 'find_peaks' (Arquetipo 1A).
    
    Parámetros:
    - period: Ventana deslizante para el análisis (ej. 100 velas).
    - prominence_pct: Umbral de prominencia. Qué porcentaje del rango total
      (max-min) de la ventana debe "sobresalir" un pico/valle para ser 
      considerado significativo (ej. 0.3%).
    - timeframe: Timeframe de remuestreo.
    """
    def __init__(self, 
                 period: int = 100, 
                 prominence_pct: float = 0.3, 
                 timeframe: Optional[str] = None):
        
        self.period = period
        self.prominence_pct = prominence_pct / 100.0 # Convertir a decimal (ej. 0.003)
        self.timeframe = timeframe
        
        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        
        self.base_name = f'DSR_{self.period}_{prominence_pct}{timeframe_suffix}'
        self.output_keys = [
            f'{self.base_name}_R_PRICE', # Precio de la última Resistencia
            f'{self.base_name}_S_PRICE', # Precio del último Soporte
            f'{self.base_name}_R_COUNT', # Conteo de Resistencias en la ventana
            f'{self.base_name}_S_COUNT'  # Conteo de Soportes en la ventana
        ]
        
        self.calculate = indicator_error_handler(output_keys_on_fail=self.output_keys)(self.calculate)

    def calculate(self, df: pd.DataFrame) -> dict:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        
        if len(data_df) < self.period:
            return {key: np.nan for key in self.output_keys}
            
        # 1. Aplicar ventana deslizante
        window_df = data_df.tail(self.period)
        
        # 2. Calcular el umbral de prominencia
        price_range = window_df['high'].max() - window_df['low'].min()
        
        # Si el rango es 0 (precio plano), no se puede calcular
        if price_range < 1e-9:
            return {key: np.nan for key in self.output_keys}
            
        min_prominence = price_range * self.prominence_pct
        
        # 3. Detectar Picos (Resistencias) en 'high'
        peaks_indices, _ = find_peaks(window_df['high'], prominence=min_prominence)
        
        # 4. Detectar Valles (Soportes) en 'low' (invirtiendo la serie)
        valleys_indices, _ = find_peaks(-window_df['low'], prominence=min_prominence)
        
        # 5. Extraer métricas
        r_count = len(peaks_indices)
        s_count = len(valleys_indices)
        
        latest_resistance = np.nan
        if r_count > 0:
            latest_resistance = window_df['high'].iloc[peaks_indices[-1]]
            
        latest_support = np.nan
        if s_count > 0:
            latest_support = window_df['low'].iloc[valleys_indices[-1]]

        return {
            self.output_keys[0]: latest_resistance,
            self.output_keys[1]: latest_support,
            self.output_keys[2]: r_count,
            self.output_keys[3]: s_count
        }



# ──────────────────────────────────────────────────────────────────────────────
# MomentumTriangle
# ──────────────────────────────────────────────────────────────────────────────

class MomentumTriangle:
    """
    Implementación del Triángulo de Momento (Eficiencia Vectorial).

    Descompone el movimiento del precio en tres componentes físicos:
    - Momento Aparente (S): Energía total gastada (recorrido acumulado H-L).
    - Momento Activo (P):   Trabajo útil (desplazamiento neto del close).
    - Momento Reactivo (Q): Energía disipada (ruido/fricción). Q² = S² - P²

    Factor de Eficiencia (fp = P/S):
    - fp > 0.8 → Tendencia limpia, bajo slippage esperado.
    - fp < 0.3 → Mercado lateral/choppy, alto riesgo de falsas rupturas.

    Todos los cálculos operan en espacio logarítmico para invarianza de escala
    (mismo comportamiento en BTC a 60k que en EURUSD a 1.08).
    """

    def __init__(self, window: int = 14, scale_factor: float = 100.0,
                 timeframe: Optional[str] = None):
        self.window = window
        self.scale_factor = scale_factor
        self.timeframe = timeframe

        timeframe_suffix = f'_{self.timeframe}' if self.timeframe else ''
        self.base_name = f'MT_{self.window}{timeframe_suffix}'

        self.output_keys = [
            f'{self.base_name}_Efficiency',  # fp  (0 a 1, adimensional)
            f'{self.base_name}_Active_P',    # Desplazamiento neto  (× scale_factor)
            f'{self.base_name}_Reactive_Q',  # Ruido/Fricción       (× scale_factor)
            f'{self.base_name}_Apparent_S',  # Recorrido total      (× scale_factor)
        ]

        self.calculate = indicator_error_handler(
            output_keys_on_fail=self.output_keys)(self.calculate)
        self.calculate_series = batch_error_handler(
            output_is_dataframe=True, cols=self.output_keys)(self.calculate_series)

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_vector_efficiency(high_log, low_log, close_log, window):
        """
        Motor Numba del Triángulo de Momento.

        BUG CORREGIDO: parámetro era 'close_arr' (no definido).
        Ahora usa 'close_log' consistente con la firma.
        """
        n = len(close_log)          # ← CORRECCIÓN: era close_arr (NameError)

        fp_arr = np.full(n, np.nan)
        p_arr  = np.full(n, np.nan)
        q_arr  = np.full(n, np.nan)
        s_arr  = np.full(n, np.nan)

        for i in range(window, n + 1):
            c_start = close_log[i - window]
            c_end   = close_log[i - 1]

            # 1. Momento Activo (P): desplazamiento neto absoluto
            active_p = np.abs(c_end - c_start)

            # 2. Momento Aparente (S): suma de rangos intra-vela log(H/L)
            w_high = high_log[i - window : i]
            w_low  = low_log[i - window : i]
            apparent_s = np.sum(w_high - w_low)

            # Protección: mercado completamente plano
            if apparent_s < 1e-9:
                fp_arr[i - 1] = 0.0
                p_arr[i - 1]  = 0.0
                q_arr[i - 1]  = 0.0
                s_arr[i - 1]  = 0.0
                continue

            # 3. Corrección geométrica: P nunca puede superar S
            if active_p > apparent_s:
                apparent_s = active_p

            # 4. Factor de Eficiencia
            fp = active_p / apparent_s

            # 5. Momento Reactivo por Pitágoras: Q² = S² - P²
            reactive_q = np.sqrt(apparent_s ** 2 - active_p ** 2)

            fp_arr[i - 1] = fp
            p_arr[i - 1]  = active_p
            q_arr[i - 1]  = reactive_q
            s_arr[i - 1]  = apparent_s

        return fp_arr, p_arr, q_arr, s_arr

    def calculate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        data_df = _resampler_cache_instance.get_resampled_df(df, self.timeframe)
        if len(data_df) < self.window:
            return pd.DataFrame(np.nan, index=df.index, columns=self.output_keys)

        h_log = np.log(data_df['high'].astype(np.float64).values)
        l_log = np.log(data_df['low'].astype(np.float64).values)
        c_log = np.log(data_df['close'].astype(np.float64).values)

        fp, p, q, s = self._calculate_vector_efficiency(h_log, l_log, c_log, self.window)

        # P, Q, S se escalan para visualización; fp es adimensional (0-1)
        result_df = pd.DataFrame({
            self.output_keys[0]: fp,
            self.output_keys[1]: p * self.scale_factor,
            self.output_keys[2]: q * self.scale_factor,
            self.output_keys[3]: s * self.scale_factor,
        }, index=data_df.index)

        return _align_to_index(result_df, df.index)

    def calculate(self, df: pd.DataFrame) -> dict:
        full_df = self.calculate_series(df)
        return _sanitize_dict(full_df.iloc[-1].to_dict())