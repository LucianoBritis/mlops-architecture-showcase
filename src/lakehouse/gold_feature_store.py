import logging

import polars as pl

from .delta_repository import PolarsDeltaRepository

logger = logging.getLogger(__name__)


class GoldFeatureStore:
    """
    Camada Gold: Geração de Features Matemáticas Prontas para Modelagem/Inferência usando LazyFrames Polars (OOM Protection).
    """

    def generate_technical_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if lf is None or "time" not in lf.collect_schema().names():
            return pl.LazyFrame()

        # 1. Base Log Returns & OHLC Ratios
        eps = 1e-6
        lf = lf.sort("time").with_columns(
            [
                # Log returns: log(close / close_lag_1)
                (pl.col("close").log() - pl.col("close").shift(1).log()).alias(
                    "log_return"
                ),
                # Garman-Klass Volatility (1980): 0.511 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
                (
                    0.511 * ((pl.col("high") / pl.col("low")).log() ** 2)
                    - 0.3862943611198906
                    * ((pl.col("close") / pl.col("open")).log() ** 2)
                )
                .clip(lower_bound=0.0)
                .sqrt()
                .alias("garman_klass_volatility"),
                # Parkinson Volatility (1980): (ln(H/L))^2 / (4 * ln(2))
                (((pl.col("high") / pl.col("low")).log() ** 2) / 2.772588722239781)
                .clip(lower_bound=0.0)
                .sqrt()
                .alias("parkinson_volatility"),
            ]
        )

        lf = lf.with_columns(
            [
                # 4. Volatilidade realizada histórica dos log retornos (Aït-Sahalia et al., 2005)
                pl.col("log_return")
                .rolling_std(window_size=10)
                .alias("realized_volatility"),
                # 5. Order Flow Imbalance (Cont, Kukanov & Stoikov, 2014): Assimetria de Pressão no Livro
                (
                    (
                        (pl.col("close") - pl.col("open"))
                        / (pl.col("high") - pl.col("low") + eps)
                    )
                    * pl.col("tick_volume")
                ).alias("order_flow_imbalance"),
                # 6. Volume Momentum: Taxa de variação do fluxo de ordens sobre a média móvel de volume
                (
                    pl.col("tick_volume")
                    / (pl.col("tick_volume").rolling_mean(window_size=10) + eps)
                ).alias("volume_momentum"),
                # 7. Proxy de Diferenciação Fracionária (López de Prado, 2018): Preservação de memória temporal com d=0.4
                (
                    pl.col("close")
                    - 0.4 * pl.col("close").shift(1)
                    - 0.12 * pl.col("close").shift(2)
                ).alias("frac_diff_proxy"),
            ]
        )

        # 8. Engenharia de Features Cíclicas (Seno e Cosseno) - Representação Circular de Tempo
        import numpy as np

        lf = lf.with_columns(
            [
                # Mês do ano (max 12)
                (2 * np.pi * pl.col("time").dt.month() / 12.0).sin().alias("month_sin"),
                (2 * np.pi * pl.col("time").dt.month() / 12.0).cos().alias("month_cos"),
                # Dia da semana (max 7: Segunda=1 a Domingo=7)
                (2 * np.pi * pl.col("time").dt.weekday() / 7.0)
                .sin()
                .alias("weekday_sin"),
                (2 * np.pi * pl.col("time").dt.weekday() / 7.0)
                .cos()
                .alias("weekday_cos"),
                # Hora do dia (max 24)
                (2 * np.pi * pl.col("time").dt.hour() / 24.0).sin().alias("hour_sin"),
                (2 * np.pi * pl.col("time").dt.hour() / 24.0).cos().alias("hour_cos"),
                # Minuto da hora (max 60)
                (2 * np.pi * pl.col("time").dt.minute() / 60.0)
                .sin()
                .alias("minute_sin"),
                (2 * np.pi * pl.col("time").dt.minute() / 60.0)
                .cos()
                .alias("minute_cos"),
            ]
        )

        # Remover linhas residuais sem histórico suficiente
        return lf.drop_nulls()

    def save_gold_features(
        self,
        lf: pl.LazyFrame,
        symbol: str,
        timeframe: str,
        base_path: str = "data",
        data_type: str | None = None,
    ) -> None:
        """Salva as features geradas na Gold Layer no Delta Lake."""
        try:
            PolarsDeltaRepository.save_partitioned(
                df=lf,
                layer="gold",
                symbol=symbol,
                timeframe=timeframe,
                timestamp_col="time",
                base_path=base_path,
                data_type=data_type,
            )
            logger.info(
                "Features Gold salvas com sucesso na Delta Table para %s (%s)",
                symbol,
                timeframe,
            )
        except Exception:
            logger.exception("Erro ao salvar features na camada Gold Delta")
            raise
