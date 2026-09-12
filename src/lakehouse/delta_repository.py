import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


class PolarsDeltaRepository:
    """
    Repositório de dados particionados (Delta/Parquet) usando Polars LazyFrames (Zero-Copy & OOM Protection).
    """

    @staticmethod
    def save_partitioned(
        df: pl.DataFrame | pl.LazyFrame,
        layer: str,
        symbol: str,
        timeframe: str,
        timestamp_col: str = "time",
        base_path: str = "./data",
        data_type: str | None = "futures",
    ) -> Path:
        """
        Salva um DataFrame ou LazyFrame particionado por camada, símbolo e timeframe.
        """
        base_dir = Path(base_path)
        data_type_str = data_type if data_type else "futures"
        target_dir = (
            base_dir
            / layer
            / f"data_type={data_type_str}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "data.parquet"

        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        df.write_parquet(target_file)
        logger.info(
            f"✅ [DeltaRepository] Dados salvos na camada {layer.upper()}: {target_file} ({df.height} linhas)"
        )
        return target_file

    @staticmethod
    def load_data(
        layer: str,
        symbol: str,
        timeframe: str,
        base_path: str = "./data",
        data_type: str = "futures",
    ) -> pl.LazyFrame:
        """
        Carrega dados como Polars LazyFrame para processamento seguro em memória.
        """
        base_dir = Path(base_path)
        target_dir = (
            base_dir
            / layer
            / f"data_type={data_type}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
        )
        files = list(target_dir.glob("*.parquet"))
        if not files:
            logger.warning(
                f"⚠️ [DeltaRepository] Nenhum arquivo encontrado em {target_dir}"
            )
            return pl.LazyFrame()

        return pl.scan_parquet(str(target_dir / "*.parquet"))
