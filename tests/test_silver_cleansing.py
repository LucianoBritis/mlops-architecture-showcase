import polars as pl

from src.lakehouse.silver_cleansing import SilverCleanser


def test_backward_difference_splicing():
    """
    Teste para garantir que o algoritmo de Backward Difference Splicing remove
    perfeitamente os gaps artificiais de rolagem de contrato na Camada Silver.
    """
    # Mock de dados brutos com um gap artificial de 1200 pontos em 20/08/2026 devido à rolagem (WINQ26 -> WINU26)
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
            ],  # Gap de rolagem de 1200 pontos no dia 20
            "tick_volume": [100, 100, 100],
            "spread": [1, 1, 1],
            "real_volume": [1000, 1000, 1000],
            "original_contract": [
                "WINQ26",
                "WINU26",
                "WINU26",
            ],  # Aciona a lógica de rolagem
        }
    )

    cleanser = SilverCleanser()
    # Aplica a limpeza para o símbolo contínuo WIN$
    spliced_lf = cleanser.clean_candles(raw_data, symbol="WIN$")
    spliced_df = spliced_lf.collect()

    # Os preços anteriores à rolagem (19/08/2026) devem ser ajustados PARA CIMA em 1200 pontos
    # para manter a continuidade com o nível de preço do novo contrato (backward splicing).
    expected_previous_close = 130000.0 + 1200.0  # 131200.0

    assert (
        spliced_df.filter(pl.col("time").cast(pl.String).str.contains("2026-08-19"))
        .select("close")
        .item()
        == expected_previous_close
    )

    # Os dias a partir da rolagem permanecem no seu novo nível absoluto
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
