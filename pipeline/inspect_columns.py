"""
Quick diagnostic: show exact column names soccerdata produces for each stat type.
Run: python pipeline/inspect_columns.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import soccerdata as sd
import pandas as pd

LEAGUE = "ENG-Premier League"
SEASON = "2324"
STAT_TYPES = ["standard", "shooting", "misc", "keeper"]

fbref = sd.FBref(leagues=[LEAGUE], seasons=[SEASON])

for stat_type in STAT_TYPES:
    try:
        raw = fbref.read_player_season_stats(stat_type=stat_type)

        # Flatten MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            flat_cols = [
                "_".join(str(c).strip().lower() for c in col if c and str(c).strip() != "nan")
                for col in raw.columns
            ]
        else:
            flat_cols = [str(c).strip().lower() for c in raw.columns]

        raw = raw.reset_index()
        # Re-flatten after reset_index
        if isinstance(raw.columns, pd.MultiIndex):
            all_cols = [
                "_".join(str(c).strip().lower() for c in col if c and str(c).strip() != "nan")
                for col in raw.columns
            ]
        else:
            all_cols = [str(c).strip().lower() for c in raw.columns]

        print(f"\n{'='*60}")
        print(f"STAT TYPE: {stat_type}  ({len(raw)} rows, {len(all_cols)} cols)")
        print(f"{'='*60}")
        for i, c in enumerate(all_cols):
            print(f"  [{i:02d}] {c}")
    except Exception as e:
        print(f"\nFAILED {stat_type}: {e}")
