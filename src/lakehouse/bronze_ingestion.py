from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import polars as pl
from .contracts import MarketTickDTO, CandleDTO
from .delta_repository import PolarsDeltaRepository

logger = logging.getLogger(__name__)

class BronzeIngestor(ABC):
    """
    Interface abstrata para a camada Bronze do Lakehouse.
    Focada em escrita append-only rápida de payloads brutos.
    """
    
    @abstractmethod
    def save_raw_ticks(self, symbol: str, ticks: List[Dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    def save_raw_candles(self, symbol: str, timeframe: str, candles: List[Dict[str, Any]]) -> bool:
        pass


class LocalParquetBronzeIngestor(BronzeIngestor):
    """
    Ingestor simplificado que armazena payloads brutos em formato Delta Table local.
    """
    def __init__(self, base_path: str):
        self._base_path = base_path

    def save_raw_ticks(self, symbol: str, ticks: List[Dict[str, Any]]) -> bool:
        import time
        from datetime import datetime, timezone
        try:
            if not ticks:
                return True
                
            # SRE Circuit Breaker: Clock Drift & Backpressure
            last_tick = ticks[-1]
            if "time_msc" in last_tick:
                tick_time = last_tick["time_msc"] / 1000.0
                now_time = datetime.now(timezone.utc).timestamp()
                drift = now_time - tick_time
                if drift > 0.010:  # 10ms
                    logger.critical(f"🛑 [SRE CIRCUIT BREAKER] Clock Drift Crítico detectado ({drift:.4f}s > 10ms) para {symbol}. Lote descartado (Drop Tail) para evitar execução HFT envenenada.")
                    return False

            df = pl.DataFrame(ticks)
            PolarsDeltaRepository.save_partitioned(
                df=df,
                layer="bronze",
                symbol=symbol,
                timeframe="ticks",
                timestamp_col="time",
                base_path=self._base_path
            )
            logger.info("Persistindo %d ticks brutos na Delta Table Bronze para %s", len(ticks), symbol)
            return True
        except Exception:
            logger.exception("Falha ao salvar ticks brutos na camada Bronze")
            return False

    def save_raw_candles(self, symbol: str, timeframe: str, raw_data: Any, data_type: str = None) -> bool:
        try:
            if isinstance(raw_data, list) and not raw_data:
                return True
            
            if isinstance(raw_data, list):
                df = pl.DataFrame(raw_data)
            elif isinstance(raw_data, pl.LazyFrame):
                df = raw_data
            elif isinstance(raw_data, pl.DataFrame):
                df = raw_data.lazy()
            else:
                import pandas as pd
                if isinstance(raw_data, pd.DataFrame):
                    df = pl.from_pandas(raw_data).lazy()
                else:
                    return False

            PolarsDeltaRepository.save_partitioned(
                df=df,
                layer="bronze",
                symbol=symbol,
                timeframe=timeframe,
                timestamp_col="time",
                base_path=self._base_path,
                data_type=data_type
            )
            logger.info("Persistindo candles brutos na Delta Table Bronze para %s (%s)", symbol, timeframe)
            return True
        except Exception:
            logger.exception("Falha ao salvar candles brutos na camada Bronze")
            return False
