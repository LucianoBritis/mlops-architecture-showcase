import polars as pl

from src.lakehouse.silver_cleansing import SilverCleanser


def test_backward_difference_splicing():
    """
    Test to ensure the Backward Difference Splicing correctly flattens artificial rollover gaps in the Silver Layer.
    """
    # Mock raw data with an artificial gap of 1000 points on 2026-08-20 due to contract rollover (WINQ26 -> WINU26)
    raw_data = pl.DataFrame(
        {
            "time": [
                "2026-08-19 10:00:00",
                "2026-08-20 10:00:00",
                "2026-08-21 10:00:00",
            ],
            "open": [130000.0, 131200.0, 131500.0],
            "high": [130000.0, 131200.0, 131500.0],
            "low": [130000.0, 131200.0, 131500.0],
            "close": [
                130000.0,
                131200.0,
                131500.0,
            ],  # 1200 point rollover gap on the 20th
            "tick_volume": [100, 100, 100],
            "spread": [1, 1, 1],
            "real_volume": [1000, 1000, 1000],
            "original_contract": [
                "WINQ26",
                "WINU26",
                "WINU26",
            ],  # Triggers the rollover logic
        }
    )

    cleanser = SilverCleanser()
    # Apply the cleansing for the continuous WIN$ symbol
    spliced_lf = cleanser.clean_candles(raw_data, symbol="WIN$")
    spliced_df = spliced_lf.collect()

    # The prices before the rollover (2026-08-19) should be shifted UP by 1200 points
    # to maintain continuity with the new contract price level (backward splicing).
    expected_previous_close = 130000.0 + 1200.0  # 131200.0

    assert (
        spliced_df.filter(pl.col("time").cast(pl.String).str.contains("2026-08-19"))
        .select("close")
        .item()
        == expected_previous_close
    )

    # The days on or after rollover remain at their new absolute level
    assert (
        spliced_df.filter(pl.col("time").cast(pl.String).str.contains("2026-08-20"))
        .select("close")
        .item()
        == 131200.0
    )
    assert (
        spliced_df.filter(pl.col("time").cast(pl.String).str.contains("2026-08-21"))
        .select("close")
        .item()
        == 131500.0
    )
