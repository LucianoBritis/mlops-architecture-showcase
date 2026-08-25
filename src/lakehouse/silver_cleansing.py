import logging
from typing import Any

import pandas as pd
import polars as pl


# Mocked imports for showcase
class PolarsDeltaRepository:
    @staticmethod
    def save_partitioned(*args, **kwargs):
        pass


logger = logging.getLogger(__name__)


class SilverCleanser:
    """
    Camada Silver: Limpeza de dados brutos e Validação usando LazyFrames Polars (SOTA OOM Protection).
    """

    def clean_candles(
        self,
        raw_data: list[dict[str, Any]] | pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        symbol: str,
    ) -> pl.LazyFrame:
        if (
            isinstance(raw_data, list)
            and not raw_data
            or isinstance(raw_data, pd.DataFrame)
            and raw_data.empty
        ):
            return pl.LazyFrame()

        try:
            if isinstance(raw_data, pd.DataFrame):
                lf = pl.from_pandas(raw_data).lazy()
            elif isinstance(raw_data, pl.DataFrame):
                lf = raw_data.lazy()
            elif isinstance(raw_data, list):
                lf = pl.LazyFrame(raw_data)
            else:
                lf = raw_data

            time_col = (
                "timestamp" if "timestamp" in lf.collect_schema().names() else "time"
            )

            if lf.collect_schema()[time_col] in [pl.Datetime, pl.Date]:
                time_expr = pl.col(time_col).dt.replace_time_zone(None)
            elif lf.collect_schema()[time_col] in [pl.Int64, pl.Int32]:
                time_expr = pl.from_epoch(pl.col(time_col), time_unit="s")
            elif lf.collect_schema()[time_col] == pl.String:
                time_expr = pl.coalesce(
                    [
                        pl.col(time_col).str.strptime(
                            pl.Datetime, "%a, %d %b %Y %H:%M:%S GMT", strict=False
                        ),
                        pl.col(time_col).str.strptime(
                            pl.Datetime, "%a, %d %b %Y %H:%M:%S %Z", strict=False
                        ),
                        pl.col(time_col).str.strptime(
                            pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False
                        ),
                        pl.col(time_col).str.strptime(
                            pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False
                        ),
                    ]
                ).dt.replace_time_zone(None)
            else:
                time_expr = pl.col(time_col)

            # SRE Lazy Evaluation: Processamento de fuso e tipagem ocorrem no Polars Lazy Engine
            open_hour = 9 if symbol.endswith("$") else 10
            lf = (
                lf.unique(subset=[time_col], keep="last")
                .with_columns([time_expr.alias("time"), pl.lit(symbol).alias("symbol")])
                .with_columns(
                    [
                        pl.when(
                            (pl.col("time").dt.hour() == 0)
                            & (pl.col("time").dt.minute() == 0)
                        )
                        .then(pl.col("time") + pl.duration(hours=open_hour))
                        .otherwise(pl.col("time"))
                        .alias("time")
                    ]
                )
            )

            # Garantir colunas essenciais preenchidas
            required_cols = [
                "symbol",
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
            ]
            available_cols = lf.collect_schema().names()
            select_exprs = []
            for col in required_cols:
                if col in available_cols:
                    select_exprs.append(pl.col(col))
                else:
                    select_exprs.append(pl.lit(0).alias(col))

            if "original_contract" in available_cols:
                select_exprs.append(pl.col("original_contract"))

            lf = lf.select(select_exprs)

            # Imputar 0 para colunas numéricas secundárias (volume/spread)
            lf = lf.with_columns(
                [
                    pl.col("tick_volume").fill_null(0),
                    pl.col("real_volume").fill_null(0),
                    pl.col("spread").fill_null(0),
                ]
            )

            # SRE Quant: Backward Difference Adjustment para Rolagem de Contratos (Emendas)
            if "original_contract" in lf.collect_schema().names() and (
                "WIN$" in symbol or "WDO$" in symbol
            ):
                lf = lf.sort("time")

                # Detecta as fronteiras de rolagem (is_rollover = True quando muda o contrato)
                lf = lf.with_columns(
                    (
                        pl.col("original_contract")
                        != pl.col("original_contract").shift(1)
                    )
                    .fill_null(False)
                    .alias("is_rollover")
                )

                # Calcula o GAP (Diferença entre o close atual e o close anterior na fronteira)
                lf = lf.with_columns(
                    pl.when(pl.col("is_rollover"))
                    .then(pl.col("close") - pl.col("close").shift(1))
                    .otherwise(0)
                    .alias("rollover_gap")
                )

                # Desloca o GAP para o contrato antigo e acumula de trás pra frente
                lf = lf.with_columns(
                    pl.col("rollover_gap")
                    .shift(-1)
                    .fill_null(0)
                    .reverse()
                    .cum_sum()
                    .reverse()
                    .alias("cum_gap")
                )

                # Aplica o gap acumulado a todas as colunas de preço
                lf = lf.with_columns(
                    [
                        (pl.col("open") + pl.col("cum_gap")).alias("open"),
                        (pl.col("high") + pl.col("cum_gap")).alias("high"),
                        (pl.col("low") + pl.col("cum_gap")).alias("low"),
                        (pl.col("close") + pl.col("cum_gap")).alias("close"),
                    ]
                )

                # Dropa as colunas temporárias
                lf = lf.drop(
                    ["is_rollover", "rollover_gap", "cum_gap", "original_contract"]
                )
            elif "original_contract" in lf.collect_schema().names():
                lf = lf.drop(["original_contract"])

            # Remover linhas que possuem preços ou timestamps nulos
            return lf.sort("time").drop_nulls(
                subset=["time", "open", "high", "low", "close"]
            )
        except Exception:
            logger.exception(
                "Erro durante o processamento da camada Silver para Candles"
            )
            return pl.LazyFrame()

    def save_silver_candles(
        self,
        lf: pl.LazyFrame,
        symbol: str,
        timeframe: str,
        base_path: str = "data",
        data_type: str | None = None,
    ) -> None:
        """Salva os candles limpos e higienizados na Silver Layer no Delta Lake com deduplicação."""
        try:
            if lf is None:
                return

            # SRE Cleansing: Materializamos e purificamos nulos residuais da Raw/Bronze Layer
            df = lf.collect().drop_nulls(
                subset=["time", "open", "high", "low", "close"]
            )
            df = df.with_columns(
                [
                    pl.col("tick_volume").fill_null(0),
                    pl.col("real_volume").fill_null(0),
                    pl.col("spread").fill_null(0),
                ]
            )

            if df.is_empty():
                logger.info(
                    f"ℹ️ [Silver Layer] Nenhum novo incremento de barras para {symbol} ({timeframe}). Base mantida íntegra."
                )
                return

            # Zero Tolerância para Infinitos/NaN nas colunas numéricas
            numeric_cols = [
                c
                for c in df.columns
                if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            for col in numeric_cols:
                if df[col].is_infinite().sum() > 0 or df[col].is_nan().sum() > 0:
                    df = df.filter(~pl.col(col).is_infinite() & ~pl.col(col).is_nan())

            PolarsDeltaRepository.save_partitioned(
                df=df.lazy(),
                layer="silver",
                symbol=symbol,
                timeframe=timeframe,
                timestamp_col="time",
                base_path=base_path,
                data_type=data_type,
            )
            logger.info(
                "Dados Silver salvos com sucesso (Garantia Zero Nulos) no Delta Lake %s (%s)",
                symbol,
                timeframe,
            )
        except Exception:
            logger.exception(
                "Erro ao salvar candles higienizados na camada Silver Delta"
            )
            raise
