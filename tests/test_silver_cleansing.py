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


def test_fractional_differentiation():
    """
    Teste de Diferenciação Fracionária (López de Prado 2018/2020) para d*=0.4.
    Garante a geração de séries temporalmente preservadas e estacionárias para PyTorch/DRL.
    """
    cleanser = SilverCleanser()
    df = pl.DataFrame(
        {
            "time": ["2026-08-19", "2026-08-20", "2026-08-21"],
            "close": [100.0, 102.0, 105.0],
        }
    )

    lf = cleanser.apply_fractional_differentiation(df.lazy(), column="close", d=0.4)
    result = lf.collect()

    assert "frac_diff_close" in result.columns
    # Verifica que a convolução fracionária gerou valores válidos
    assert result["frac_diff_close"].is_null().sum() == 0
    # Primeiro valor é 100.0 (w0 = 1.0)
    assert result["frac_diff_close"][0] == 100.0
    # Segundo valor é 102.0 - 0.4 * 100.0 = 62.0
    assert abs(result["frac_diff_close"][1] - 62.0) < 1e-4
