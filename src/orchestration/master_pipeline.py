#!/usr/bin/env python3
"""
master_pipeline.py
===================
Orquestrador de Pipeline em Prefect 2.x para o Lakehouse de Derivativos (WIN / WDO).
Gerencia o fluxo completo de dados em 4 etapas assíncronas e resilientes com política de retries e logs detalhados:
  1. Ingestão Bronze (Append-Only Raw Data com Retry Backoff)
  2. Limpeza & Emenda Silver (Fractional Diff + Baradel Rollover com Retry Backoff)
  3. Feature Store Gold (Garman-Klass, Parkinson, OFI, Realized Volatility)
  4. Validação de Qualidade de Dados (Zero-Tolerance Quality Gates)
"""

import logging
import os
from typing import Any

import polars as pl

# Configura banco in-memory temporário para execução resiliente local sem depender de servidor externo
os.environ.setdefault(
    "PREFECT_API_DATABASE_CONNECTION_URL", "sqlite+aiosqlite:///:memory:"
)

from prefect import flow, get_run_logger, task

from src.lakehouse.bronze_ingestion import LocalParquetBronzeIngestor
from src.lakehouse.data_quality import LakehouseDataValidator
from src.lakehouse.delta_repository import PolarsDeltaRepository
from src.lakehouse.gold_feature_store import GoldFeatureStore
from src.lakehouse.silver_cleansing import SilverCleanser

logger = logging.getLogger("LakehouseMasterOrchestrator")


def sre_task_failure_hook(task, task_run, state):
    """Hook de observabilidade SRE acionado em falhas de task."""
    logger.critical(
        f"🚨 [SRE FAILURE ALERT] Task '{task.name}' falhou. Estado: {state.name}. Detalhes: {state.message}"
    )


@task(
    name="Ingest_Bronze_Data",
    retries=3,
    retry_delay_seconds=[1, 2, 5],
    log_prints=True,
    on_failure=[sre_task_failure_hook],
)
def task_ingest_bronze(
    symbol: str,
    timeframe: str,
    raw_data: list[dict[str, Any]] | pl.DataFrame,
    base_path: str = "./data",
) -> bool:
    """
    Task Prefect: Ingestão de dados brutos na camada Bronze.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"📥 [Bronze Task] Ingerindo dados brutos para {symbol} ({timeframe}). (Retries: 3, Backoff: 1s,2s,5s)"
    )
    ingestor = LocalParquetBronzeIngestor(base_path=base_path)
    success = ingestor.save_raw_candles(
        symbol=symbol, timeframe=timeframe, raw_data=raw_data, data_type="futures"
    )
    if not success:
        logger.error(
            f"❌ [Bronze Task] Ingestão falhou para {symbol}. Acionando política de retry..."
        )
        raise RuntimeError(f"Falha na ingestão Bronze para {symbol}")
    return True


@task(
    name="Cleanse_Silver_Data",
    retries=3,
    retry_delay_seconds=[1, 2, 5],
    log_prints=True,
    on_failure=[sre_task_failure_hook],
)
def task_cleanse_silver(
    symbol: str,
    timeframe: str,
    base_path: str = "./data",
    frac_d: float = 0.4,
) -> bool:
    """
    Task Prefect: Purificação, Rolagem Sintética Perpétua (Baradel et al., 2023)
    e Diferenciação Fracionária d* (López de Prado, 2018) na camada Silver.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"⚙️ [Silver Task] Higienizando e aplicando FracDiff d={frac_d} para {symbol}... (Retries: 3)"
    )

    # Carregar dados Bronze
    bronze_lf = PolarsDeltaRepository.load_data(
        layer="bronze", symbol=symbol, timeframe=timeframe, base_path=base_path
    )
    if bronze_lf is None or bronze_lf.collect_schema().len() == 0:
        logger.error(
            f"❌ [Silver Task] Dados Bronze ausentes para {symbol}. Acionando retry..."
        )
        raise ValueError(f"Nenhum dado Bronze encontrado para {symbol}")

    cleanser = SilverCleanser()
    silver_lf = cleanser.clean_candles(raw_data=bronze_lf, symbol=symbol)

    # Aplicação de Diferenciação Fracionária d*
    if "close" in silver_lf.collect_schema().names():
        silver_lf = cleanser.apply_fractional_differentiation(
            silver_lf, column="close", d=frac_d
        )

    # Persistir na camada Silver
    PolarsDeltaRepository.save_partitioned(
        df=silver_lf,
        layer="silver",
        symbol=symbol,
        timeframe=timeframe,
        base_path=base_path,
        data_type="futures",
    )
    logger.info(f"✅ [Silver Task] Concluída com sucesso para {symbol}")
    return True


@task(
    name="Build_Gold_Features",
    retries=3,
    retry_delay_seconds=[1, 2, 5],
    log_prints=True,
    on_failure=[sre_task_failure_hook],
)
def task_build_gold_features(
    symbol: str, timeframe: str, base_path: str = "./data"
) -> bool:
    """
    Task Prefect: Extração de Features Quantitativas na camada Gold.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"📊 [Gold Task] Gerando Features Quantitativas (Garman-Klass, Parkinson, OFI) para {symbol}... (Retries: 3)"
    )

    silver_lf = PolarsDeltaRepository.load_data(
        layer="silver", symbol=symbol, timeframe=timeframe, base_path=base_path
    )
    if silver_lf is None or silver_lf.collect_schema().len() == 0:
        logger.error(
            f"❌ [Gold Task] Dados Silver ausentes para {symbol}. Acionando retry..."
        )
        raise ValueError(f"Nenhum dado Silver encontrado para {symbol}")

    feature_store = GoldFeatureStore()
    gold_lf = feature_store.generate_technical_features(silver_lf)

    PolarsDeltaRepository.save_partitioned(
        df=gold_lf,
        layer="gold",
        symbol=symbol,
        timeframe=timeframe,
        base_path=base_path,
        data_type="futures",
    )
    logger.info(f"✅ [Gold Task] Concluída com sucesso para {symbol}")
    return True


@task(
    name="Validate_Data_Quality",
    retries=2,
    retry_delay_seconds=[1, 3],
    log_prints=True,
    on_failure=[sre_task_failure_hook],
)
def task_validate_data_quality(
    symbol: str, timeframe: str, base_path: str = "./data"
) -> dict[str, Any]:
    """
    Task Prefect: Portão de Qualidade de Dados (Zero-Tolerance Gates).
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"🛡️ [Quality Gate Task] Validando dados da camada Gold para {symbol}... (Retries: 2)"
    )

    gold_lf = PolarsDeltaRepository.load_data(
        layer="gold", symbol=symbol, timeframe=timeframe, base_path=base_path
    )
    validator = LakehouseDataValidator()
    validated_lf = validator.validate_silver_candles(gold_lf, symbol)
    row_count = validated_lf.collect().height

    logger.info(
        f"🎉 [Quality Gate] Aprovado com sucesso para {symbol}! ({row_count} registros)"
    )
    return {"symbol": symbol, "is_valid": True, "validated_rows": row_count}


@flow(name="Lakehouse Master Flow (Futures Derivatives)", log_prints=True)
def master_pipeline(
    symbols: list[str] | None = None,
    timeframe: str = "M5",
    sample_data: dict[str, list[dict[str, Any]]] | None = None,
    base_path: str = "./data",
) -> dict[str, Any]:
    """
    Prefect Master Flow para o Lakehouse de Futuros.
    Orquestra a transição assíncrona Bronze -> Silver -> Gold -> Data Quality com logs e retries resilientes.
    """
    run_logger = get_run_logger()
    if symbols is None:
        symbols = ["WIN$", "WDO$"]

    run_logger.info(
        f"🚀 [Prefect Master Flow] Iniciando orquestração Lakehouse (Logs & Retries habilitados) para ativos: {symbols}"
    )

    results = {}

    for symbol in symbols:
        run_logger.info(f"🔄 Executando pipeline para {symbol} ({timeframe})...")

        # 1. Obter ou gerar dados brutos
        raw_candles = (
            sample_data.get(symbol) if sample_data and symbol in sample_data else []
        )
        if not raw_candles:
            from datetime import datetime, timedelta, timezone

            base_time = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
            raw_candles = [
                {
                    "time": (base_time + timedelta(minutes=5 * i)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "open": 100.0 + i * 0.1,
                    "high": 100.5 + i * 0.1,
                    "low": 99.5 + i * 0.1,
                    "close": 100.2 + i * 0.1,
                    "tick_volume": 1500 + i * 10,
                    "symbol": symbol,
                }
                for i in range(100)
            ]

        # Execução encadeada das Tasks Prefect
        b_res = task_ingest_bronze(
            symbol=symbol,
            timeframe=timeframe,
            raw_data=raw_candles,
            base_path=base_path,
        )
        s_res = task_cleanse_silver(
            symbol=symbol, timeframe=timeframe, base_path=base_path
        )
        g_res = task_build_gold_features(
            symbol=symbol, timeframe=timeframe, base_path=base_path
        )
        q_report = task_validate_data_quality(
            symbol=symbol, timeframe=timeframe, base_path=base_path
        )

        results[symbol] = {
            "bronze": b_res,
            "silver": s_res,
            "gold": g_res,
            "quality_report": q_report,
        }

    run_logger.info("🎉 [Prefect Master Flow] Execução finalizada com 100% de sucesso!")
    return results


if __name__ == "__main__":
    res = master_pipeline(symbols=["WIN$", "WDO$"], timeframe="M5")
    print("Resultado da execução do Prefect Flow:", res)
