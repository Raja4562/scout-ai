"""
ScoutAI - Data Freshness Pipeline
==================================
One command to pull the latest season stats and keep the app current.

What it does (in order):
  1. Detect the current football season automatically from today's date
  2. Back up the existing players_all_leagues.csv
  3. Fetch fresh FBref stats for all 5 leagues  (slow: ~30-45 min)
  4. Overlay real Transfermarkt contract dates   (fast: ~2 min)
  5. Rebuild the embedding index                 (fast: ~1 min)
  6. Hot-reload the running server without restart

Usage:
  Full refresh (new season stats + contracts + embeddings):
      python pipeline/refresh.py

  Contracts + embeddings only (skip slow FBref scrape):
      python pipeline/refresh.py --skip-fbref

  Specific leagues or seasons:
      python pipeline/refresh.py --leagues EPL "La Liga" --seasons 2526 2425

  Reload server data only (after manual CSV edits):
      python pipeline/refresh.py --reload-only

Scheduling (Windows Task Scheduler):
  Set trigger: monthly, first day of July (new season) and January (winter window)
  Action: python D:\\project\\ScoutAI\\pipeline\\refresh.py --skip-fbref
  Full scrape: run manually at start of season or set weekly overnight job
"""

import argparse
import logging
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scoutai.refresh")


# ---------------------------------------------------------------------------
# Season detection
# ---------------------------------------------------------------------------

def current_season() -> tuple[str, str]:
    """
    Return (current_season_code, previous_season_code) based on today's date.
    Football seasons run Aug-May. After July 1 = new season has started.

    Examples:
      May  2026  ->  ("2526", "2425")   # in-progress season 2025-26
      Aug  2026  ->  ("2627", "2526")   # new season 2026-27 just started
    """
    today = date.today()
    if today.month >= 7:
        yr = today.year          # e.g., Aug 2026 → yr=2026 → season 2026-27
    else:
        yr = today.year - 1      # e.g., May 2026 → yr=2025 → season 2025-26

    def _code(y):
        return f"{str(y)[2:]}{str(y+1)[2:]}"

    return _code(yr), _code(yr - 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess command, stream its output, return True on success."""
    logger.info("━━━ %s ━━━", label)
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        # Don't capture — let logs stream to the terminal in real time
    )
    elapsed = time.time() - start
    if result.returncode == 0:
        logger.info("✓ %s completed in %.0fs", label, elapsed)
        return True
    else:
        logger.error("✗ %s FAILED (exit=%d) in %.0fs", label, result.returncode, elapsed)
        return False


def _backup_csv(csv_path: Path) -> Path | None:
    """Copy players_all_leagues.csv to a timestamped backup. Returns backup path."""
    if not csv_path.exists():
        logger.info("No existing CSV to back up — first run.")
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = csv_path.parent / f"players_all_leagues_backup_{ts}.csv"
    shutil.copy2(csv_path, backup)
    logger.info("Backed up existing data to %s", backup.name)
    return backup


def _renormalize_csv(csv_path: Path) -> bool:
    """
    Re-run normalizer on the existing CSV so market values reflect the
    latest trained model. Called after contract updates and model retraining.
    """
    logger.info("━━━ Re-normalize CSV (apply model valuations) ━━━")
    start = time.time()
    try:
        import pandas as pd
        from normalizer import normalise

        df = pd.read_csv(csv_path, low_memory=False)
        # Reset valuations so the model always overwrites them cleanly
        df["market_value"] = 0.0
        df["future_value"] = 0.0
        df = normalise(df)
        df.to_csv(csv_path, index=False)
        elapsed = time.time() - start

        mv = df["market_value"]
        logger.info(
            "✓ Re-normalized %d players in %.0fs  (median MV=%.1fM, max=%.1fM)",
            len(df), elapsed, mv.median(), mv.max()
        )
        return True
    except Exception as exc:
        logger.error("✗ Re-normalization failed: %s", exc)
        return False


def _rebuild_embeddings() -> bool:
    """Rebuild the embedding index from the current CSV."""
    logger.info("━━━ Rebuild embeddings ━━━")
    start = time.time()
    try:
        # Delete stale cache so build_or_load_embeddings forces a rebuild
        emb_cache = ROOT / "data" / "player_embeddings.npy"
        if emb_cache.exists():
            emb_cache.unlink()
            logger.info("Deleted stale embedding cache")

        from data_loader import reload_players
        from embeddings import build_or_load_embeddings
        df = reload_players()
        build_or_load_embeddings(df, force_rebuild=True)
        elapsed = time.time() - start
        logger.info("✓ Embeddings rebuilt in %.0fs (%d players)", elapsed, len(df))
        return True
    except Exception as exc:
        logger.error("✗ Embedding rebuild failed: %s", exc)
        return False


def _hot_reload_server(server_url: str = "http://localhost:8001") -> bool:
    """
    Tell the running server to reload its data from disk.
    Returns True if the server confirmed the reload.
    """
    url = f"{server_url}/api/admin/reload"
    try:
        r = requests.post(url, timeout=120)
        if r.status_code == 200:
            data = r.json()
            logger.info(
                "✓ Server hot-reloaded: %d players across %d league(s)",
                data.get("player_count", "?"),
                data.get("league_count", "?"),
            )
            return True
        else:
            logger.warning("Server reload returned HTTP %d — restart may be needed", r.status_code)
            return False
    except requests.ConnectionError:
        logger.info("Server not running at %s — skipping hot-reload", server_url)
        return False
    except Exception as exc:
        logger.warning("Hot-reload failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    start_total = time.time()
    csv_path = ROOT / "data" / "players_all_leagues.csv"
    py = sys.executable

    logger.info("═" * 60)
    logger.info("ScoutAI Data Refresh  —  %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("═" * 60)

    if args.reload_only:
        logger.info("Mode: reload-only (no scraping, just reload server data)")
        _hot_reload_server(args.server)
        return

    # ── Season detection ──────────────────────────────────────────────────
    if args.seasons:
        seasons = args.seasons
        logger.info("Seasons (from args): %s", seasons)
    else:
        cur, prev = current_season()
        seasons = [cur, prev]
        logger.info("Auto-detected seasons: %s (current) + %s (prev)", cur, prev)

    leagues = args.leagues or ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    logger.info("Leagues: %s", ", ".join(leagues))
    logger.info("")

    # ── Backup ───────────────────────────────────────────────────────────
    _backup_csv(csv_path)
    logger.info("")

    ok = True

    # ── Step 1: FBref stats ───────────────────────────────────────────────
    if args.skip_fbref:
        logger.info("Skipping FBref scrape (--skip-fbref)")
    else:
        cmd = [py, "pipeline/fetch_leagues.py", "--seasons"] + seasons + ["--leagues"] + leagues
        if not _run(cmd, "FBref stats (soccerdata)"):
            logger.error("FBref fetch failed. Keeping existing player data.")
            logger.error("Run with --skip-fbref to refresh contracts only.")
            ok = False
        logger.info("")

    # ── Step 2: Contract dates ────────────────────────────────────────────
    if not _run([py, "pipeline/fetch_contracts.py"], "Transfermarkt contract dates"):
        logger.warning("Contract fetch failed — existing contract_years kept as-is")
    logger.info("")

    # ── Step 3: Retrain valuation model ──────────────────────────────────
    if args.skip_train:
        logger.info("Skipping model retraining (--skip-train)")
    else:
        if not _run([py, "pipeline/train_valuation.py"], "LightGBM valuation model"):
            logger.warning("Model training failed — existing model kept as-is")
    logger.info("")

    # ── Step 4: Re-normalize CSV with latest model ────────────────────────
    _renormalize_csv(csv_path)
    logger.info("")

    # ── Step 5: Style clustering ──────────────────────────────────────────
    logger.info("━━━ Style Clustering (K-Means + UMAP) ━━━")
    cluster_rebuild = not args.skip_train   # retrain clusters when model is retrained
    cluster_cmd = [py, "pipeline/cluster_styles.py"]
    if cluster_rebuild:
        cluster_cmd.append("--rebuild")
    if not _run(cluster_cmd, "Playing style clustering"):
        logger.warning("Style clustering failed — labels may be stale")
    logger.info("")

    # ── Step 6: Embeddings ────────────────────────────────────────────────
    _rebuild_embeddings()
    logger.info("")

    # ── Step 7: Hot-reload server ─────────────────────────────────────────
    _hot_reload_server(args.server)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed_total = time.time() - start_total
    logger.info("")
    logger.info("═" * 60)
    if ok:
        logger.info("✓ Refresh complete in %.0fs (%.1f min)", elapsed_total, elapsed_total / 60)
    else:
        logger.info("⚠ Refresh done with warnings in %.0fs", elapsed_total)

    # Print current distribution
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        by_league = df.groupby("league").size().sort_values(ascending=False)
        logger.info("Players: %d total", len(df))
        for league, count in by_league.items():
            logger.info("  %-12s %d", league, count)
        dist = df["contract_years"].value_counts().sort_index().to_dict()
        logger.info("Contract distribution: %s", dist)
    except Exception:
        pass

    logger.info("═" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ScoutAI data freshness pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/refresh.py                          # full refresh
  python pipeline/refresh.py --skip-fbref             # contracts only (~5 min)
  python pipeline/refresh.py --leagues EPL "La Liga"  # two leagues only
  python pipeline/refresh.py --reload-only            # just tell server to reload
        """,
    )
    p.add_argument(
        "--seasons", nargs="+", default=None,
        help="Season codes e.g. 2526 2425 (default: auto-detected)",
    )
    p.add_argument(
        "--leagues", nargs="+", default=None,
        choices=["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
        help="Leagues to refresh (default: all 5)",
    )
    p.add_argument(
        "--skip-fbref", action="store_true",
        help="Skip FBref scraping — only refresh contract dates + model + embeddings (~5 min)",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip LightGBM retraining — use existing model/valuation_lgbm.pkl",
    )
    p.add_argument(
        "--reload-only", action="store_true",
        help="Skip all scraping, just tell the running server to reload its data from disk",
    )
    p.add_argument(
        "--server", default="http://localhost:8001",
        help="Server URL for hot-reload (default: http://localhost:8001)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
