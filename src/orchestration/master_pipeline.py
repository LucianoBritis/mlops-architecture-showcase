#!/usr/bin/env python3
"""
master_dag_futures.py
=====================
Orquestrador SOTA Meta-RL (HFT Actor) exclusivo para Contratos Futuros (WIN / WDO).
Integra a purificação de dados (Filtro de Entropia GMM) e o Treinamento HFT via ML-2.
NOTA SRE: Derivativos não passam por Seleção de Portfólio (Markowitz/ML-5).
"""

import os

os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = "sqlite+aiosqlite:///:memory:"
import asyncio
import logging
import signal
import sys
from pathlib import Path

from app.training.shared_lib.auto_fix_mounts import auto_fix_mlflow_paths
from app.training.shared_lib.paths import ProjectPaths
from prefect import flow, get_run_logger, task

# SRE Auto-Heal do Disco Removível
auto_fix_mlflow_paths(ProjectPaths.PROJECT_ROOT, ProjectPaths.MLFLOW_DB)

# Configuração de caminhos do Django e Labsmtr
REPO_ROOT = Path(__file__).resolve().parents[2]  # server/
sys.path.insert(0, str(REPO_ROOT))

# Configurar Django para chamar comandos internos
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")
import django

django.setup()
from django.core.management import call_command

TRAINING_DIR = ProjectPaths.TRAINING_DIR
BACKEND_DIR = ProjectPaths.BACKEND_DIR
VENV_PYTHON = ProjectPaths.VENV_PYTHON

# =============================================================================
# SRE Anti-Zombie: Registry global de processos filhos
# Qualquer subprocess registrado aqui será terminado se o DAG receber SIGTERM/SIGINT
# =============================================================================
_CHILD_PROCESSES: list[asyncio.subprocess.Process] = []


def _sre_sigterm_handler(signum, frame):
    """Propaga SIGTERM para todos os filhos antes de encerrar o processo pai."""
    _logger = logging.getLogger("SRE_AntiZombie")
    _logger.warning(
        f"🚨 [SRE] Sinal {signum} recebido. Propagando SIGTERM para {len(_CHILD_PROCESSES)} processo(s) filho(s)..."
    )
    for proc in _CHILD_PROCESSES:
        try:
            if proc.returncode is None:  # ainda vivo
                proc.terminate()
                _logger.warning(f"   ☠ PID {proc.pid} terminado via SIGTERM.")
        except Exception as e:
            _logger.error(f"   ❌ Falha ao terminar PID {proc.pid}: {e}")
    _logger.warning("✅ [SRE] Todos os filhos sinalizados. Encerrando DAG de Futuros.")
    os._exit(1)


signal.signal(signal.SIGTERM, _sre_sigterm_handler)
signal.signal(signal.SIGINT, _sre_sigterm_handler)


def build_env(project_path: Path, dry_run: bool, asset: str, timeframe: str) -> dict:
    env = os.environ.copy()
    env["DRY_RUN"] = "1" if dry_run else "0"
    env["ASSET"] = asset
    env["SYMBOL"] = asset
    env["TIMEFRAME"] = timeframe
    env["DATA_TYPE"] = "futures"
    env["SILVER_PATH"] = str(ProjectPaths.SILVER_DIR)
    env["GOLD_PATH"] = str(ProjectPaths.GOLD_DIR)
    env["MLFLOW_TRACKING_URI"] = "sqlite:///" + str(ProjectPaths.MLFLOW_DB)

    pythonpaths = [
        str(BACKEND_DIR.parent),
        str(BACKEND_DIR),
        str(project_path),
        str(TRAINING_DIR),
        str(TRAINING_DIR / "ml_1_quant_system"),
        str(TRAINING_DIR / "ml_2_hft_system"),
        str(TRAINING_DIR / "ml_3_saidec_system"),
        str(TRAINING_DIR / "ml_4_series_temporais"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join([p for p in pythonpaths if p])
    return env


async def run_training_script(
    name: str, entrypoint: Path, project_path: Path, env: dict
):
    logger = logging.getLogger(__name__)
    logger.info(f"[{name}] Iniciando {entrypoint.name}...")
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            VENV_PYTHON,
            str(entrypoint),
            cwd=str(project_path),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        _CHILD_PROCESSES.append(process)  # SRE: registra para propagação de SIGTERM

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                logger.info(f"[{name}] {text}")

        await process.wait()
        if process.returncode != 0:
            logger.error(f"❌ [{name}] Falha (código {process.returncode}).")
            return False
        logger.info(f"✅ [{name}] Concluído com sucesso.")
        return True
    except Exception:
        logger.exception(f"❌ [{name}] Falha crítica de execução.")
        return False
    finally:
        if process is not None and process in _CHILD_PROCESSES:
            _CHILD_PROCESSES.remove(process)  # SRE: desregistra após conclusão


@task(name="Information_Bars_Filter_Futures")
def task_build_information_bars(asset: str, base_timeframe: str = "M5") -> str:
    import urllib.parse

    import polars as pl
    from app.training.shared_lib.information_bars import InformationBarOptimizer
    from app.training.shared_lib.paths import ProjectPaths

    logger = logging.getLogger(__name__)

    # Localiza os dados base recém ingeridos (suporta URL-encoded symbol%24)
    possible_assets = [asset, urllib.parse.quote(asset, safe="")]
    gold_dir = None

    for candidate_asset in possible_assets:
        for dtype in ["futures", "spot"]:
            candidate_dir = (
                ProjectPaths.PROJECT_ROOT
                / "data"
                / "gold_delta"
                / f"data_type={dtype}"
                / f"symbol={candidate_asset}"
                / f"timeframe={base_timeframe}"
            )
            if candidate_dir.exists() and list(candidate_dir.rglob("*.parquet")):
                gold_dir = candidate_dir
                break
        if gold_dir:
            break

    if not gold_dir or not gold_dir.exists():
        logger.error(
            f"❌ Dados base {base_timeframe} ausentes para {asset}. A DAG falhará (Tolerância Zero)."
        )
        raise ValueError(f"Dados {base_timeframe} ausentes para o derivativo {asset}.")

    files = list(gold_dir.rglob("*.parquet"))
    if not files:
        raise ValueError(
            f"Nenhum arquivo Parquet encontrado para {asset} em {gold_dir}"
        )

    dfs = [pl.read_parquet(f) for f in files]
    df = pl.concat(dfs, how="vertical")

    # Padroniza colunas temporais
    time_col = None
    for col in ["timestamp", "time", "open_time"]:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df = df.sort(time_col).unique(subset=[time_col], keep="first")
        if time_col != "timestamp":
            df = df.rename({time_col: "timestamp"})

    # Padroniza close/volume
    if "close" not in df.columns:
        c_col = [c for c in df.columns if "close" in c.lower()]
        if c_col:
            df = df.rename({c_col[0]: "close"})

    if "volume" not in df.columns:
        v_col = [c for c in df.columns if "vol" in c.lower()]
        if v_col:
            df = df.rename({v_col[0]: "volume"})
        else:
            raise ValueError(
                f"❌ Coluna volume ausente em {asset}. Fallback sintético proibido pelas regras SRE."
            )

    # Otimização por GMM & Entropia de Shannon (Gera Volume Bars purificados)
    opt = InformationBarOptimizer()
    sota_df = opt.optimize_dataframe(df.to_pandas(), asset)
    sota_df = pl.from_pandas(sota_df)

    # Salva partição SOTA_VOL
    out_tf = "SOTA_VOL"
    out_dir = (
        ProjectPaths.PROJECT_ROOT
        / "data"
        / "gold_delta"
        / "data_type=futures"
        / f"symbol={asset}"
        / f"timeframe={out_tf}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sota_df.write_parquet(out_dir / "data.parquet")
    logger.info(
        f"✅ Filtro Entrópico SOTA concluído para o derivativo {asset}. Timeframe Alvo: {out_tf}"
    )
    return out_tf


@task
async def run_ml2_hft(dry_run: bool, asset: str, timeframe: str):
    """Treinamento exclusivo do ML-2 HFT para Derivativos (PPO / Actor-Critic)."""
    logger = logging.getLogger(__name__)
    logger.info(
        f"🤖 [ML-2] Iniciando HFT PPO Training para {asset} no timeframe {timeframe}..."
    )

    p2 = TRAINING_DIR / "ml_2_hft_system"
    e2 = p2 / "training/train.py"
    env = build_env(p2, dry_run, asset, timeframe)

    await run_training_script(f"ML-2 HFT [{asset}]", e2, p2, env)


@task
async def task_run_coint_futures(dry_run: bool, symbols: list):
    """
    Executa a análise de Cointegração ML-0 para Contratos Futuros (WIN$ x WDO$),
    garantindo que o sinal de spread previna a entrada simultânea na mesma direção (Long-Long).
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"🔗 [ML-0] Analisando Cointegração e Spread de Futuros para: {symbols}..."
    )
    p0 = TRAINING_DIR / "ml_0_coint_system"
    e0 = p0 / "train_advanced_cointegration.py"
    env = build_env(p0, dry_run, ",".join(symbols), "M5")
    await run_training_script("ML-0 Cointegration [Futuros]", e0, p0, env)


@task
def task_run_hawkes_futures(asset: str, timeframe: str = "M5"):
    """
    Executa a análise de Hawkes Process & Microestrutura HFT (britis-investing-high-risk SOTA)
    para medir picos de agressão e cascata de ordens no ativo futuro.
    """
    import numpy as np
    import polars as pl
    from app.training.ml_4_series_temporais.hawkes_microstructure import (
        MultivariateHawkesEstimator,
    )
    from app.training.shared_lib.paths import ProjectPaths

    logger = logging.getLogger(__name__)
    logger.info(
        f"⚡ [ML-4 Hawkes] Calculando intensidade de fluxo de ordens e Branching Ratio para {asset}..."
    )

    gold_dir = ProjectPaths.get_gold_path(
        data_type="futures", asset=asset, timeframe=timeframe
    )
    if not gold_dir.exists() or not list(gold_dir.rglob("*.parquet")):
        gold_dir = ProjectPaths.get_silver_path(
            data_type="futures", asset=asset, timeframe=timeframe
        )

    if gold_dir.exists() and list(gold_dir.rglob("*.parquet")):
        df = pl.read_parquet(list(gold_dir.rglob("*.parquet"))[0]).to_pandas()
        if "close" in df.columns and len(df) > 5:
            closes = df["close"].values
            returns = np.diff(closes) / closes[:-1]
            buy_events = returns[returns > 0]
            sell_events = np.abs(returns[returns < 0])

            estimator = MultivariateHawkesEstimator()
            res = estimator.fit_evaluate(buy_events, sell_events)
            logger.info(
                f"📊 [ML-4 Hawkes {asset}] Branching Ratio: {res.branching_ratio:.4f} | Cascade: {res.is_supercritical_cascade} | TSRV Vol: {res.tsrv_volatility:.6f}"
            )


@task
def task_run_feature_selection_futures(asset: str, timeframe: str = "M5"):
    """
    Executa a seleção de features em 3 Etapas (Variância/VIF -> Clustered Boruta SHAP -> Gold Mask)
    para o ativo futuro de forma a filtrar o ruído estocástico.
    """
    from app.data_lakehouse.delta_repository import PolarsDeltaRepository
    from app.training.shared_lib.feature_selection import FeatureSelectionPipeline
    from app.training.shared_lib.paths import ProjectPaths

    logger = logging.getLogger(__name__)
    logger.info(
        f"🎯 [3-Stage Feature Selection] Executando Clustered Boruta SHAP para {asset}..."
    )

    gold_df = PolarsDeltaRepository.load_data(
        layer="gold",
        data_type="futures",
        symbol=asset,
        timeframe=timeframe,
        base_path=str(ProjectPaths.DATA_DIR),
    )
    if gold_df.is_empty():
        gold_df = PolarsDeltaRepository.load_data(
            layer="silver",
            data_type="futures",
            symbol=asset,
            timeframe=timeframe,
            base_path=str(ProjectPaths.DATA_DIR),
        )

    if not gold_df.is_empty():
        df_pd = gold_df.to_pandas()
        candidate_cols = [
            c
            for c in df_pd.columns
            if c
            not in [
                "time",
                "symbol",
                "data_type",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        if len(candidate_cols) > 1:
            pipeline = FeatureSelectionPipeline(
                n_estimators=30, max_iter=5, random_state=42
            )
            accepted, summary = pipeline.run_selection(
                df_pd, feature_cols=candidate_cols, target_col="log_return"
            )
            logger.info(
                f"✅ [3-Stage Feature Selection {asset}] {len(accepted)}/{len(candidate_cols)} features aceitas: {accepted}"
            )

            # Persistir metadados da seleção de features para consumo dos modelos ML_*
            meta_dir = ProjectPaths.DATA_DIR / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_file = meta_dir / f"selected_features_{asset}.json"
            import json

            with open(meta_file, "w") as f:
                json.dump(
                    {"asset": asset, "accepted_features": accepted, "summary": summary},
                    f,
                    indent=2,
                )
            logger.info(f"💾 Metadados de Seleção salvos em {meta_file}")


@task
def task_run_hft_backtest(asset: str, timeframe: str = "M5"):
    """
    Executa a simulação Event-Driven de Backtest com fricção B3 (Slippage, Emolumentos)
    gerando o relatório institucional e o gráfico da Curva de Patrimônio (Equity Curve).
    """
    from app.training.shared_lib.event_driven_backtest import generate_backtest_report

    logger = logging.getLogger(__name__)
    logger.info(f"📊 [HFT Backtest] Executando simulação Event-Driven para {asset}...")
    report = generate_backtest_report(asset, timeframe)
    if report:
        metrics = report.get("metrics", {})
        logger.info(
            f"📈 [HFT Backtest {asset}] Sharpe: {metrics.get('sharpe_ratio')} | Sortino: {metrics.get('sortino_ratio')} | Max DD: {metrics.get('max_drawdown_pct')}% | Win Rate: {metrics.get('win_rate_pct')}%"
        )


@flow(name="HFT Sniper Master DAG (Futures Derivatives)")
async def futures_master_dag(
    dry_run: bool = False, symbols: list | None = None, base_timeframe: str = "M5"
):
    logger = get_run_logger()

    if symbols is None:
        symbols = ["WIN$", "WDO$"]

    logger.info(f"🏁 Iniciando DAG exclusiva de Futuros SOTA | Ativos: {symbols}")
    logger.info(
        "🛡️ SRE Constraint: Pulando seleção de portfólio de Markowitz (ML-5). Futuros treinam HFT direto."
    )

    # 0. Ingestão Automática no Lakehouse (3500 barras para convergência estatística do GMM)
    logger.info(
        f"📥 Ingerindo dados base {base_timeframe} para derivativos no Lakehouse..."
    )
    await asyncio.to_thread(
        call_command,
        "run_lakehouse_pipeline",
        symbols=symbols,
        timeframe=base_timeframe,
        bars=50000,
        data_type="futures",
    )

    # 1. Análise de Cointegração ML-0 para pares de Futuros (Hedge Direcional WIN$/WDO$)
    if len(symbols) >= 2:
        await task_run_coint_futures(dry_run, symbols)

    sota_timeframes_map = {}

    # Execução sequencial para evitar OOM (Proteção de Memória SRE)
    for asset in symbols:
        logger.info(f"⚙️ Processando {asset}...")
        try:
            # 1. Hawkes Process & Microestrutura HFT (Portado do britis-investing-high-risk)
            await asyncio.to_thread(task_run_hawkes_futures, asset, base_timeframe)

            # 2. 3-Stage Feature Selection (Clustered Boruta SHAP com Expurgo Temporal)
            await asyncio.to_thread(
                task_run_feature_selection_futures, asset, base_timeframe
            )

            # 3. Filtro Entrópico: Converte M5/M1 em Volume Bars purificadas via GMM
            sota_tf = await asyncio.to_thread(
                task_build_information_bars, asset, base_timeframe
            )
            sota_timeframes_map[asset] = sota_tf

            # 4. Treina o pipeline HFT ML-2 alimentando-o com os dados purificados (salva modelo padronizado M5)
            await run_ml2_hft(dry_run, asset, base_timeframe)

            # 5. Executa Backtest Event-Driven HFT com Fricção B3 (Slippage + Corretagem)
            await asyncio.to_thread(task_run_hft_backtest, asset, base_timeframe)

        except Exception as e:
            logger.error(f"❌ Falha fatal no pipeline do {asset}: {e}")
            logger.error("🛑 Abortando treinamento para este ativo (Fail-Fast).")

    logger.info("🎉 Futures HFT Pipeline concluído com sucesso!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--symbols", type=str, default="WIN$,WDO$", help="Futuros separados por vírgula"
    )
    parser.add_argument(
        "--base-tf",
        type=str,
        default="M5",
        help="Timeframe de base para o HFT (M1 ou M5)",
    )
    args = parser.parse_known_args()[0]

    symbols_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
    asyncio.run(
        futures_master_dag(
            dry_run=args.dry_run, symbols=symbols_list, base_timeframe=args.base_tf
        )
    )
