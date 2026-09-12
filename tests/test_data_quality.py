import polars as pl
import pytest

from src.lakehouse.data_quality import DataQualityError, LakehouseDataValidator


def test_data_quality_valid_data():
    """Teste onde todos os candles são perfeitamente válidos."""
    validator = LakehouseDataValidator()

    df = pl.DataFrame(
        {
            "time": ["2026-08-19", "2026-08-20"],
            "open": [100.0, 105.0],
            "high": [110.0, 115.0],
            "low": [90.0, 100.0],
            "close": [105.0, 110.0],
            "tick_volume": [1000, 1500],
        }
    )

    # Should not raise exception
    validated_lf = validator.validate_silver_candles(df.lazy(), "WIN$")
    assert validated_lf.collect().shape == (2, 6)


def test_data_quality_high_lower_than_low():
    """Teste onde um candle possui máxima menor que mínima, o que é matematicamente impossível."""
    validator = LakehouseDataValidator()

    df = pl.DataFrame(
        {
            "time": ["2026-08-19", "2026-08-20"],
            "open": [100.0, 105.0],
            "high": [80.0, 115.0],  # HIGH < LOW no primeiro candle (80 < 90)
            "low": [90.0, 100.0],
            "close": [105.0, 110.0],
            "tick_volume": [1000, 1500],
        }
    )

    with pytest.raises(DataQualityError) as exc_info:
        validator.validate_silver_candles(df.lazy(), "WIN$")

    assert "Falha na validação de integridade de preços" in str(exc_info.value)


def test_data_quality_null_prices():
    """Teste onde um preço de fechamento é nulo."""
    validator = LakehouseDataValidator()

    df = pl.DataFrame(
        {
            "time": ["2026-08-19", "2026-08-20"],
            "open": [100.0, 105.0],
            "high": [110.0, 115.0],
            "low": [90.0, 100.0],
            "close": [None, 110.0],  # Null value
            "tick_volume": [1000, 1500],
        }
    )

    with pytest.raises(DataQualityError):
        validator.validate_silver_candles(df.lazy(), "WIN$")


def test_data_quality_spike_warning(caplog):
    """Teste onde existe um spike bizarro de >20%, apenas gera log de warning, não quebra."""
    import logging

    validator = LakehouseDataValidator()

    df = pl.DataFrame(
        {
            "time": ["2026-08-19"],
            "open": [100.0],
            "high": [150.0],  # 50% jump in one candle
            "low": [90.0],
            "close": [140.0],
            "tick_volume": [1000],
        }
    )

    with caplog.at_level(logging.WARNING):
        validator.validate_silver_candles(df.lazy(), "WIN$")

    assert "spikes anormais (>20% intrabarra)" in caplog.text
