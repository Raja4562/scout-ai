"""
Merge two players_all_leagues.csv files into one.
Used when pipeline runs are split across sessions due to FBref rate limiting.

Usage:
    python pipeline/merge_csv.py --base data/players_all_leagues.csv
                                  --add  data/players_extra.csv
                                  --out  data/players_all_leagues.csv

The merged file deduplicates by (player, team, league), keeping the most
recent season_year. Then re-runs normalization so _pct and _xpct columns
are correct across the combined dataset.
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from normalizer import normalise

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("scoutai.merge")


def merge(base_path: str, add_path: str, out_path: str) -> None:
    logger.info("Loading base: %s", base_path)
    base = pd.read_csv(base_path, low_memory=False)

    logger.info("Loading additional: %s", add_path)
    add = pd.read_csv(add_path, low_memory=False)

    logger.info("Base: %d rows, Add: %d rows", len(base), len(add))
    combined = pd.concat([base, add], ignore_index=True, sort=False)

    # Deduplicate - keep latest season per (player, team, league)
    if "season_year" in combined.columns:
        combined = combined.sort_values("season_year", ascending=False)
    combined = combined.drop_duplicates(subset=["player", "team", "league"], keep="first")
    combined = combined.reset_index(drop=True)
    combined["_idx"] = combined.index

    logger.info("After dedup: %d players across %d leagues",
                len(combined), combined["league"].nunique())

    # Re-normalize so cross-league percentiles are consistent
    logger.info("Re-running normalization ...")
    combined = normalise(combined)

    Path(out_path).parent.mkdir(exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info("Saved %d players to %s", len(combined), out_path)

    by_league = combined.groupby("league").size().sort_values(ascending=False)
    logger.info("Players by league:\n%s", by_league.to_string())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--add",  required=True)
    p.add_argument("--out",  required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(args.base, args.add, args.out)
