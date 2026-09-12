import logging

import polars as pl


class DataQualityError(Exception):
    """Exceção levantada quando validações de Data Quality falham."""


class LakehouseDataValidator:
    """
    Motor de Observabilidade de Dados para a camada Silver.
    Aplica regras rígidas de integridade de OHLCV (Open, High, Low, Close, Volume) e detecta
    anomalias matemáticas (Spikes) em tempo real, utilizando a engine lazy do Polars.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_silver_candles(self, lf: pl.LazyFrame, symbol: str) -> pl.LazyFrame:
        """
        Executa uma suíte de validações de Data Quality na pipeline Silver.
        Se as regras rígidas falharem, levanta uma DataQualityError (Fail-Fast).

        Args:
            lf: Polars LazyFrame com os dados sendo limpos.
            symbol: O ativo sendo processado.

        Returns:
            O LazyFrame validado.
        """
        self.logger.info(f"[{symbol}] Executando Data Quality Check (OHLCV)...")

        # Como estamos numa pipeline lazy, não podemos fazer .filter().is_empty()
        # diretamente sem acionar o compute().
        # Mas podemos adicionar uma checagem eagerly rápida sobre o schema e integridade
        # OU embutir as validações como um .map_batches / assertions no LazyFrame.
        # Para ser compatível com grandes volumes e não quebrar o LazyFrame,
        # usamos expressões condicionais (when/then/otherwise) ou forçamos o cache temporário.

        # Materializamos um subconjunto pequeno apenas para validar se há anomalias estruturais críticas
        # Mas como é um pipeline de ETL contínuo, a melhor forma em Polars é adicionar
        # flag columns e então levantar erro. Para simplificar e manter a performance,
        # fazemos a checagem com `select` e `collect()` em modo streaming se possível.

        # Regras de Negócio OHLC:
        # 1. High >= Low
        # 2. High >= Open e High >= Close
        # 3. Low <= Open e Low <= Close
        # 4. Volume >= 0
        # 5. Não haver valores nulos nas colunas de preço

        validation_expr = (
            (pl.col("high") < pl.col("low"))
            | (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
            | (pl.col("tick_volume") < 0)
            | (pl.col("open").is_null())
            | (pl.col("close").is_null())
        )

        # Filtramos as violações
        violations = (
            lf.filter(validation_expr)
            .select(["time", "open", "high", "low", "close"])
            .collect()
        )

        if not violations.is_empty():
            self.logger.error(
                f"[{symbol}] 🚨 DATA QUALITY ALERT: Encontradas {len(violations)} linhas violando as regras de OHLCV."
            )
            self.logger.error(f"Exemplos de violação:\n{violations.head(5)}")
            raise DataQualityError(
                f"[{symbol}] Falha na validação de integridade de preços (High < Low, nulls, etc)."
            )

        # Verificação de Spikes (Variações maiores que 20% no mesmo candle - anomalia de feed)
        spike_expr = (pl.col("high") - pl.col("low")) / pl.col("open")
        spikes = (
            lf.filter(spike_expr > 0.20)
            .select(["time", "open", "high", "low", "close"])
            .collect()
        )

        if not spikes.is_empty():
            self.logger.warning(
                f"[{symbol}] ⚠️ DATA QUALITY WARNING: {len(spikes)} candles com spikes anormais (>20% intrabarra)."
            )
            # Spike pode acontecer em crashes (circuit breaker), não travamos o pipeline, apenas alertamos fortemente.
            self.logger.warning(f"Exemplos de Spikes:\n{spikes.head(5)}")

        self.logger.info(f"[{symbol}] ✅ Data Quality Checks passados com sucesso.")
        return lf
