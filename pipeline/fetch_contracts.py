"""
ScoutAI - Contract Data Fetcher
Pulls real contract expiry dates from transfermarkt-datasets (Cloudflare CDN)
and merges them into data/players_all_leagues.csv.

Source: https://github.com/dcaribou/transfermarkt-datasets
  - Auto-updated from Transfermarkt
  - 47k+ players, current season data
  - contract_expiration_date column

Match strategy:
  1. Exact name match (lowercase, stripped)
  2. Fuzzy name match via rapidfuzz (score >= 88)
  3. Fallback: keep existing synthetic contract_years

Run:
    python pipeline/fetch_contracts.py
"""

import gzip
import io
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("scoutai.contracts")

TM_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
CSV_PATH = ROOT / "data" / "players_all_leagues.csv"
TODAY = datetime(2026, 5, 8)          # current date for contract_years calculation
MAX_CONTRACT_YEARS = 5                 # cap display at 5 years


def _years_remaining(expiry_date) -> int | None:
    """Compute integer years remaining from today. Returns None if no date."""
    if pd.isna(expiry_date):
        return None
    try:
        exp = pd.Timestamp(expiry_date)
        days = (exp - pd.Timestamp(TODAY)).days
        if days <= 0:
            return 0          # already expired or expires this summer
        years = days / 365.25
        # Round to nearest integer, minimum 1
        return min(MAX_CONTRACT_YEARS, max(1, round(years)))
    except Exception:
        return None


def fetch_tm_data() -> pd.DataFrame:
    """Download and parse the transfermarkt-datasets players CSV."""
    logger.info("Downloading Transfermarkt dataset from CDN ...")
    r = requests.get(TM_URL, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    logger.info("Downloaded %d KB", len(r.content) // 1024)

    df = pd.read_csv(gzip.open(io.BytesIO(r.content)), low_memory=False)
    df["contract_expiration_date"] = pd.to_datetime(df["contract_expiration_date"], errors="coerce")

    # Keep most recent season record per player name
    df = df[df["last_season"] >= 2023].copy()
    df = df.sort_values("last_season", ascending=False).drop_duplicates(subset=["name"], keep="first")
    df["name_key"] = df["name"].str.lower().str.strip()

    logger.info(
        "TM dataset: %d players (2023+), %d with contract date",
        len(df), df["contract_expiration_date"].notna().sum()
    )
    return df


def match_players(our: pd.DataFrame, tm: pd.DataFrame) -> pd.Series:
    """
    Return a Series of contract_expiration_date indexed by our df's index.
    Pass 1: exact name match.
    Pass 2: fuzzy match on unmatched rows (rapidfuzz token_sort_ratio >= 88).
    """
    from rapidfuzz import process, fuzz

    our_keys = our["player"].str.lower().str.strip()
    tm_dict = dict(zip(tm["name_key"], tm["contract_expiration_date"]))
    tm_keys = list(tm_dict.keys())

    result = pd.Series(pd.NaT, index=our.index, dtype="datetime64[ns]")

    # Pass 1: exact
    exact_hits = 0
    fuzzy_hits = 0
    for idx, name_key in our_keys.items():
        if name_key in tm_dict:
            result[idx] = tm_dict[name_key]
            exact_hits += 1

    # Pass 2: fuzzy for remaining
    unmatched_idx = our.index[result.isna()]
    for idx in unmatched_idx:
        name_key = our_keys[idx]
        match = process.extractOne(
            name_key, tm_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=88
        )
        if match:
            result[idx] = tm_dict[match[0]]
            fuzzy_hits += 1

    total = len(our)
    matched = result.notna().sum()
    logger.info(
        "Match results: %d exact + %d fuzzy = %d/%d (%.1f%%) matched",
        exact_hits, fuzzy_hits, matched, total, matched / total * 100
    )
    return result


def run():
    tm = fetch_tm_data()

    logger.info("Loading %s ...", CSV_PATH)
    our = pd.read_csv(CSV_PATH, low_memory=False)
    logger.info("Our dataset: %d players", len(our))

    # Match contract dates
    contract_dates = match_players(our, tm)

    # Compute contract_years from real dates; fall back to existing synthetic
    real_count = 0
    synthetic_count = 0
    new_years = []
    new_expires = []

    for idx in our.index:
        real_yrs = _years_remaining(contract_dates[idx])
        if real_yrs is not None:
            new_years.append(real_yrs)
            new_expires.append(TODAY.year + real_yrs)
            real_count += 1
        else:
            # Keep existing synthetic value
            existing = int(our.at[idx, "contract_years"]) if "contract_years" in our.columns else 2
            new_years.append(existing)
            new_expires.append(TODAY.year + existing)
            synthetic_count += 1

    our["contract_years"] = new_years
    our["contract_expires"] = new_expires

    logger.info(
        "contract_years: %d real / %d synthetic fallback",
        real_count, synthetic_count
    )
    logger.info("Distribution: %s", dict(
        zip(*[x.tolist() for x in __import__("numpy").unique(new_years, return_counts=True)])
    ))

    # Spot check
    for name in ["Harry Kane", "Erling Haaland", "Rodrygo", "Olivier Giroud"]:
        rows = our[our["player"] == name]
        if len(rows):
            row = rows.iloc[0]
            logger.info(
                "  %s: contract_years=%d expires=%d",
                row["player"], row["contract_years"], row["contract_expires"]
            )

    our.to_csv(CSV_PATH, index=False)
    logger.info("Saved updated CSV to %s", CSV_PATH)


if __name__ == "__main__":
    run()
