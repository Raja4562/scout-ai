"""
ScoutAI - Multi-League Data Pipeline
Fetches player statistics from FBref via soccerdata for 5 major leagues
across up to 3 seasons, merges stat categories, normalises cross-league,
and writes data/players_all_leagues.csv ready for the app to load.

Usage:
    python pipeline/fetch_leagues.py [--seasons 2324 2223] [--leagues EPL "La Liga"]

Data flow:
    FBref (soccerdata) -> raw DataFrames -> merge stat categories
    -> column mapping -> cross-league normalization (normalizer.py)
    -> data/players_all_leagues.csv

Valid soccerdata FBref stat_types:
    standard, keeper, shooting, playing_time, misc
    (passing, defense, possession are NOT supported by soccerdata v1.9)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from normalizer import normalise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("scoutai.pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FBREF_LEAGUE_IDS = {
    "EPL":        "ENG-Premier League",
    "La Liga":    "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie A":    "ITA-Serie A",
    "Ligue 1":    "FRA-Ligue 1",
}

DEFAULT_SEASONS = ["2324", "2223"]
DEFAULT_LEAGUES = list(FBREF_LEAGUE_IDS.keys())

# Only the stat_types actually supported by soccerdata v1.9 FBref scraper
STAT_TYPES = ["standard", "shooting", "misc", "keeper"]

# ---------------------------------------------------------------------------
# Column mapping: exact soccerdata flattened name -> internal name
#
# soccerdata flattens MultiIndex columns by joining with "_":
#   ("Playing Time", "Min") -> "playing time_min"
#   ("Performance", "Gls") -> "performance_gls"
#   ("Expected", "xG")     -> "expected_xg"
# ---------------------------------------------------------------------------

COLUMN_REMAP = {
    # Index columns (available after reset_index)
    "player":                    "player",
    "team":                      "team",
    "pos":                       "position",
    "age":                       "age",
    "born":                      "born",
    "nation":                    "nation",
    "league":                    "league",
    "season":                    "season",

    # Standard - Playing Time
    "playing time_mp":           "playing_time_mp",
    "playing time_starts":       "playing_time_starts",
    "playing time_min":          "minutes",
    "playing time_90s":          "nineties",

    # Standard - Performance
    "performance_gls":           "goals",
    "performance_ast":           "assists",
    "performance_g+a":           "g+a",
    "performance_g-pk":          "g-pk",
    "performance_pk":            "pk",
    "performance_pkatt":         "pkatt",
    "performance_crdy":          "crdy",
    "performance_crdr":          "crdr",

    # Standard - Expected (xG on FBref standard page)
    "expected_xg":               "xg",
    "expected_npxg":             "npxg",
    "expected_xag":              "xag",
    "expected_npxg+xag":         "npxg+xag",

    # Standard - Progression
    "progression_prgc":          "progression_prgc",
    "progression_prgp":          "progression_prgp",
    "progression_prgr":          "progression_prgr",

    # Shooting stat type columns (prefixed "standard_" in FBref shooting page)
    "standard_sh":               "sh",
    "standard_sot":              "sot",
    "standard_sh/90":            "sh_per90",
    "standard_sot/90":           "sot_per90",
    "standard_g/sh":             "g_per_shot",
    "standard_g/sot":            "g_per_sot",
    "standard_dist":             "shot_dist",

    # Shooting - Expected (on shooting page)
    "expected_xg.1":             "xg_shooting",   # duplicate xg from shooting page, ignore
    "expected_npxg.1":           "npxg_shooting",

    # Misc - Performance (yellow/red cards shared with standard - will be dropped as dupes)
    # Unique to misc:
    "performance_fls":           "fouls_committed",
    "performance_fld":           "fouls_drawn",
    "performance_off":           "offsides",
    "performance_crs":           "crs",
    "performance_int":           "int",
    "performance_tklw":          "tackles_tkl",
    "performance_pkwon":         "pkwon",
    "performance_pkcon":         "pkcon",
    "performance_og":            "og",
    "performance_2crdy":         "second_yellow",
    "performance_clr":           "clr",
    "performance_err":           "err",

    # Misc - Aerial Duels
    "aerial duels_won":          "aerial_duels_won",
    "aerial duels_lost":         "aerial_duels_lost",
    "aerial duels_won%":         "aerial_duels_wonpct",

    # Keeper - Performance (for GKs only, outfield players will have NaN)
    "performance_ga":            "gk_ga",
    "performance_ga90":          "gk_ga90",
    "performance_sota":          "gk_sota",
    "performance_saves":         "gk_saves",
    "performance_save%":         "gk_savepct",
    "performance_cs":            "gk_cs",
    "performance_cs%":           "gk_cspct",
    "performance_w":             "gk_wins",
    "performance_d":             "gk_draws",
    "performance_l":             "gk_losses",

    # Keeper - Penalty Kicks
    "penalty kicks_pkatt":       "gk_pk_att",
    "penalty kicks_pksv":        "gk_pk_saved",
    "penalty kicks_pkm":         "gk_pk_missed",
}

OUT_FILE = ROOT / "data" / "players_all_leagues.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    soccerdata returns MultiIndex columns.
    Flatten to single-level lower-case strings joined by '_'.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c).strip().lower() for c in col if c and str(c).strip() != "nan")
            for col in df.columns
        ]
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply COLUMN_REMAP to standardise column names using exact matching."""
    rename = {}
    for col in df.columns:
        if col in COLUMN_REMAP:
            rename[col] = COLUMN_REMAP[col]
    if rename:
        logger.debug("Renaming %d columns", len(rename))
    return df.rename(columns=rename)


def _log_columns(df: pd.DataFrame, tag: str) -> None:
    """Log first 60 column names for debugging."""
    cols = list(df.columns)
    logger.info("%s columns (%d total): %s", tag, len(cols), cols[:60])


def _fetch_league_season(
    fbref,
    league_name: str,
    season: str,
) -> pd.DataFrame | None:
    """
    Fetch and merge all stat categories for one (league, season).
    Returns a merged DataFrame or None if all fetches fail.
    """
    frames = {}
    for stat_type in STAT_TYPES:
        try:
            logger.info("  Fetching %s | %s | %s ...", league_name, season, stat_type)
            raw = fbref.read_player_season_stats(stat_type=stat_type)
            raw = _flatten_columns(raw)
            raw = raw.reset_index()
            # Drop completely empty columns
            raw = raw.dropna(axis=1, how="all")
            logger.info("    OK: %d rows, %d columns", len(raw), len(raw.columns))
            if stat_type == "standard":
                _log_columns(raw, f"{league_name}/{season}/standard")
            frames[stat_type] = raw
        except Exception as exc:
            logger.warning("    SKIP %s/%s/%s: %s", stat_type, league_name, season, exc)

    if not frames:
        return None

    # Standard stats as base - required
    base = frames.get("standard")
    if base is None:
        logger.error("No standard stats for %s %s - skipping this combination", league_name, season)
        return None

    # Merge each additional stat type onto the base
    merge_keys = ["player", "team"]
    # Verify merge keys exist
    for k in merge_keys:
        if k not in base.columns:
            logger.warning("Missing merge key '%s' in standard stats for %s %s", k, league_name, season)
            return None

    for stat_type in STAT_TYPES:
        if stat_type == "standard":
            continue
        df = frames.get(stat_type)
        if df is None:
            continue

        # Verify merge keys present in joining frame
        if not all(k in df.columns for k in merge_keys):
            logger.warning("  %s: missing player/team columns, skipping merge", stat_type)
            continue

        # Drop columns already in base (avoids value duplication on merge)
        drop_cols = [c for c in df.columns if c in base.columns and c not in merge_keys]
        df = df.drop(columns=drop_cols, errors="ignore")

        base = pd.merge(base, df, on=merge_keys, how="left", suffixes=("", f"_{stat_type}"))
        logger.info("  After merging %s: %d columns", stat_type, len(base.columns))

    base = _remap_columns(base)
    base["league"] = league_name
    base["season"] = season

    return base


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns that normalizer also uses."""
    # long_cmp_rate: fraction of long passes completed (may not exist without passing stats)
    if "long_cmp" in df.columns and "long_att" in df.columns:
        att = df["long_att"].fillna(0).astype(float)
        df["long_cmp_rate"] = np.where(
            att > 0, df["long_cmp"].fillna(0).astype(float) / att, 0.0
        )
    return df


def _add_position_primary(df: pd.DataFrame) -> pd.DataFrame:
    pos_col = "position" if "position" in df.columns else None
    if pos_col is None:
        df["position_primary"] = "MF"
        return df
    if "position_primary" not in df.columns:
        df["position_primary"] = (
            df[pos_col].fillna("MF").astype(str)
            .str.split(",").str[0].str.strip().str.upper()
        )
    mapping = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}
    df["position_primary"] = df["position_primary"].map(mapping).fillna("MF")
    return df


def _add_nationality(df: pd.DataFrame) -> pd.DataFrame:
    nation_col = next((c for c in df.columns if "nation" in c.lower() and c != "nationality"), None)
    if nation_col:
        df["nationality"] = (
            df[nation_col].astype(str).str.split().str[-1].str.upper()
        )
    elif "nationality" not in df.columns:
        df["nationality"] = ""
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(leagues: list[str], seasons: list[str]) -> None:
    try:
        import soccerdata as sd
    except ImportError:
        logger.error("soccerdata not installed. Run: pip install soccerdata")
        sys.exit(1)

    logger.info("soccerdata version: %s", getattr(sd, "__version__", "unknown"))

    all_frames = []

    for league_name in leagues:
        fbref_id = FBREF_LEAGUE_IDS.get(league_name)
        if not fbref_id:
            logger.warning("Unknown league: %s - skipping", league_name)
            continue

        for season in seasons:
            logger.info("=== %s | Season %s ===", league_name, season)
            try:
                fbref = sd.FBref(leagues=[fbref_id], seasons=[season])
                df = _fetch_league_season(fbref, league_name, season)
                if df is not None and len(df) > 0:
                    all_frames.append(df)
                    logger.info("  Added %d rows for %s %s", len(df), league_name, season)
                else:
                    logger.warning("  No data returned for %s %s", league_name, season)
            except Exception as exc:
                logger.error("  FBref init failed for %s %s: %s", league_name, season, exc)

        # Pause between leagues to respect FBref rate limits
        import time
        logger.info("Pausing 30s before next league to avoid FBref rate-limiting ...")
        time.sleep(30)

    if not all_frames:
        logger.error("No data fetched. Check internet connection and soccerdata version.")
        sys.exit(1)

    logger.info("Concatenating %d frames ...", len(all_frames))
    combined = pd.concat(all_frames, ignore_index=True, sort=False)

    # Post-process
    combined = _compute_derived(combined)
    combined = _add_position_primary(combined)
    combined = _add_nationality(combined)

    # Parse season_year
    if "season" in combined.columns:
        try:
            combined["season_year"] = combined["season"].astype(str).str[:2].astype(int) + 2000
        except Exception:
            combined["season_year"] = 2024

    # Filter minimum minutes
    if "minutes" in combined.columns:
        combined = combined[combined["minutes"].fillna(0).astype(float) >= 300].copy()
    else:
        logger.warning("'minutes' column not found - skipping minutes filter")

    # Deduplicate: keep latest season per (player, team, league)
    combined = combined.sort_values("season_year", ascending=False)
    combined = combined.drop_duplicates(subset=["player", "team", "league"], keep="first")
    combined = combined.reset_index(drop=True)
    combined["_idx"] = combined.index

    logger.info("After deduplication: %d players", len(combined))
    by_league = combined.groupby("league").size().sort_values(ascending=False)
    logger.info("Raw counts by league:\n%s", by_league.to_string())

    # Cross-league normalization
    logger.info("Running normalization on %d players ...", len(combined))
    combined = normalise(combined)

    # Save
    OUT_FILE.parent.mkdir(exist_ok=True)
    combined.to_csv(str(OUT_FILE), index=False)
    logger.info("Saved %d players to %s", len(combined), OUT_FILE)

    by_league_final = combined.groupby("league").size().sort_values(ascending=False)
    logger.info("Final players by league:\n%s", by_league_final.to_string())
    logger.info("Total columns in output: %d", len(combined.columns))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ScoutAI multi-league data pipeline")
    p.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help=f"FBref season codes (default: {DEFAULT_SEASONS})",
    )
    p.add_argument(
        "--leagues", nargs="+", default=DEFAULT_LEAGUES,
        choices=list(FBREF_LEAGUE_IDS.keys()),
        help="Leagues to fetch (default: all 5)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting pipeline: leagues=%s  seasons=%s", args.leagues, args.seasons)
    run(args.leagues, args.seasons)
