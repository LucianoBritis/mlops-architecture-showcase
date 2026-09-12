import tempfile

from prefect import flow, task

from src.orchestration.master_pipeline import (
    master_pipeline,
    task_build_gold_features,
    task_cleanse_silver,
    task_ingest_bronze,
    task_validate_data_quality,
)


def test_prefect_master_flow_execution():
    """
    Testa a execução completa do Flow Prefect (Bronze -> Silver -> Gold -> Data Quality).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = master_pipeline(
            symbols=["WIN$", "WDO$"], timeframe="M5", base_path=tmp_dir
        )

        assert "WIN$" in results
        assert "WDO$" in results

        for symbol in ["WIN$", "WDO$"]:
            res = results[symbol]
            assert res["bronze"] is True
            assert res["silver"] is True
            assert res["gold"] is True
            assert res["quality_report"]["is_valid"] is True
            assert res["quality_report"]["validated_rows"] > 0


def test_individual_prefect_tasks():
    """
    Testa cada Task Prefect individualmente para isolamento de falhas.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        symbol = "TEST$"
        timeframe = "M5"
        sample_candles = [
            {
                "time": f"2026-01-01 09:{i:02d}:00",
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "tick_volume": 1000 + i * 10,
                "symbol": symbol,
            }
            for i in range(25)
        ]

        # 1. Ingest Bronze Task
        assert (
            task_ingest_bronze.fn(
                symbol=symbol,
                timeframe=timeframe,
                raw_data=sample_candles,
                base_path=tmp_dir,
            )
            is True
        )

        # 2. Cleanse Silver Task
        assert (
            task_cleanse_silver.fn(
                symbol=symbol, timeframe=timeframe, base_path=tmp_dir
            )
            is True
        )

        # 3. Build Gold Features Task
        assert (
            task_build_gold_features.fn(
                symbol=symbol, timeframe=timeframe, base_path=tmp_dir
            )
            is True
        )

        # 4. Validate Data Quality Task
        report = task_validate_data_quality.fn(
            symbol=symbol, timeframe=timeframe, base_path=tmp_dir
        )
        assert report["is_valid"] is True
        assert report["validated_rows"] == 15


def test_task_retry_configuration():
    """
    Valida se as Tasks possuem configurações de retries e log_prints ativos.
    """
    assert task_ingest_bronze.retries == 3
    assert task_ingest_bronze.retry_delay_seconds == [1, 2, 5]
    assert task_ingest_bronze.log_prints is True

    assert task_cleanse_silver.retries == 3
    assert task_cleanse_silver.retry_delay_seconds == [1, 2, 5]

    assert task_build_gold_features.retries == 3
    assert task_build_gold_features.retry_delay_seconds == [1, 2, 5]

    assert task_validate_data_quality.retries == 2
    assert task_validate_data_quality.retry_delay_seconds == [1, 3]


def test_task_runtime_retry_execution():
    """
    Demonstra empiricamente a re-execução (retry) em tempo de execução do Prefect
    quando uma exceção transiente ocorre dentro de uma Task.
    """
    attempts = 0

    @task(name="Transient_Retry_Task", retries=2, retry_delay_seconds=0)
    def flaky_task():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("Falha transiente de rede/I-O simulada")
        return "RETRY_SUCCESS"

    @flow(name="Test_Runtime_Retry_Flow")
    def test_flow():
        return flaky_task()

    result = test_flow()
    assert result == "RETRY_SUCCESS"
    assert attempts == 2  # Comprova que o Prefect tentou novamente após a falha inicial
