"""
ScoutAI - Data Loader
Loads and prepares the player dataset. Auto-detects multi-league CSV if
available (data/players_all_leagues.csv), otherwise falls back to the
EPL-only capstone dataset (data/players.csv).

Deduplicates to the most recent season per (player, team, league).
"""

import logging
import os
import pandas as pd
import numpy as np
from config import DATA_FILE, DATA_FILE_MULTI, MIN_MINUTES, RADAR_FEATURES, LEAGUE_DIFFICULTY
from transfer_confidence import compute_confidence

logger = logging.getLogger("scoutai.data")

_df: pd.DataFrame | None = None


def _active_data_file() -> str:
    """Return multi-league file if it exists, else EPL-only fallback."""
    if os.path.exists(DATA_FILE_MULTI):
        logger.info("Multi-league dataset detected: %s", DATA_FILE_MULTI)
        return DATA_FILE_MULTI
    return DATA_FILE


def load_players() -> pd.DataFrame:
    """
    Load player CSV, clean nulls, deduplicate to latest season per player,
    assign a stable integer index used by the embedding matrix.
    """
    global _df
    if _df is not None:
        return _df

    filepath = _active_data_file()
    logger.info("Loading player dataset from %s", filepath)
    df = pd.read_csv(filepath, low_memory=False)

    # Keep only players with meaningful playing time
    df = df[df["minutes"].fillna(0) >= MIN_MINUTES].copy()

    # Ensure season_year exists
    if "season_year" not in df.columns:
        df["season_year"] = (
            df["season"].astype(str).str.split("-").str[0]
        )
        df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").fillna(2023).astype(int)

    # Add league column (EPL for legacy single-league data)
    if "league" not in df.columns:
        df["league"] = "EPL"

    # Deduplicate: keep the most recent season per (player, team, league)
    df = df.sort_values("season_year", ascending=False)
    df = df.drop_duplicates(subset=["player", "team", "league"], keep="first")
    df = df.reset_index(drop=True)
    df["_idx"] = df.index

    # Normalize nationality string: "eng ENG" -> "ENG"
    nation_col = next((c for c in df.columns if "nation" in c.lower()), None)
    if nation_col:
        df["nationality"] = df[nation_col].astype(str).str.split().str[-1].str.upper()
    elif "nationality" not in df.columns:
        df["nationality"] = ""

    # Fill numeric nulls with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)

    # Ensure position_primary exists
    if "position_primary" not in df.columns:
        df["position_primary"] = (
            df["position"].astype(str).str.split(",").str[0].str.strip()
        )

    # Map multi-character positions to primary code (FW,MF -> FW)
    pos_map = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}
    df["position_primary"] = (
        df["position_primary"]
        .str.upper()
        .map(pos_map)
        .fillna("MF")
    )

    n_leagues = df["league"].nunique()
    league_counts = df.groupby("league").size().to_dict()
    logger.info(
        "Loaded %d players across %d league(s): %s",
        len(df), n_leagues, league_counts,
    )
    _df = df
    return _df


def reload_players() -> pd.DataFrame:
    """Force a reload (call when CSV changes at runtime)."""
    global _df
    _df = None
    return load_players()


def get_player_seasons(player: str, team: str = "") -> list[dict]:
    """
    Return every season row for a player from the raw CSV (no dedup).
    Used by Feature 24 — Season Progression Chart.

    Returns a list of dicts sorted oldest→newest, each containing:
      season, season_year, team, league, age, minutes,
      psi, market_value, goals_per90, assists_per90, xg_per90,
      pass_acc, tackles_per90, interceptions, gk_savepct, gk_cspct, gk_ga90
    """
    filepath = _active_data_file()
    try:
        raw = pd.read_csv(filepath, low_memory=False)
    except Exception as exc:
        logger.warning("get_player_seasons: could not read CSV — %s", exc)
        return []

    mask = raw["player"].astype(str) == player
    if team:
        mask &= raw["team"].astype(str) == team
    rows = raw[mask].copy()
    if rows.empty:
        return []

    # Ensure season_year
    if "season_year" not in rows.columns:
        rows["season_year"] = (
            rows["season"].astype(str).str.split("-").str[0]
        )
    rows["season_year"] = pd.to_numeric(rows["season_year"], errors="coerce").fillna(2023).astype(int)

    # Ensure league
    if "league" not in rows.columns:
        rows["league"] = "EPL"

    rows = rows.sort_values("season_year")

    def _f(d: dict, key: str, default: float = 0.0) -> float:
        try:
            v = d.get(key, default)
            return round(float(v) if v is not None else default, 3)
        except (TypeError, ValueError):
            return default

    def _i(d: dict, key: str, default: int = 0) -> int:
        try:
            v = d.get(key, default)
            return int(float(v)) if v is not None else default
        except (TypeError, ValueError):
            return default

    seasons = []
    for _, row in rows.iterrows():
        d = row.to_dict()
        mins     = _f(d, "minutes", 0)
        nineties = max(mins / 90, 0.01)

        goals_p90   = _f(d, "goals_per90")  or round(_f(d, "goals")   / nineties, 2)
        assists_p90 = _f(d, "assists_per90") or round(_f(d, "assists") / nineties, 2)
        tackles_p90 = round(_f(d, "tackles_tkl") / nineties, 2)

        seasons.append({
            "season":        str(d.get("season", "")),
            "season_year":   _i(d, "season_year", 2023),
            "team":          str(d.get("team", "")),
            "league":        str(d.get("league", "EPL")),
            "age":           _i(d, "age"),
            "minutes":       _i(d, "minutes"),
            # overall
            "psi":           round(_f(d, "psi", 0) * 100, 1),   # 0-100
            "market_value":  _f(d, "market_value"),
            # attacking
            "goals_per90":   goals_p90,
            "assists_per90": assists_p90,
            "xg_per90":      _f(d, "xg_per90"),
            "xag_per90":     _f(d, "xag_per90"),
            # passing
            "pass_acc":      _f(d, "total_cmppct"),
            # defensive
            "tackles_per90": tackles_p90,
            "interceptions": _f(d, "int"),
            "aerial_pct":    _f(d, "aerial_duels_wonpct"),
            # keeper
            "gk_savepct":    _f(d, "gk_savepct"),
            "gk_cspct":      _f(d, "gk_cspct"),
            "gk_ga90":       _f(d, "gk_ga90"),
        })

    return seasons


def player_to_dict(row, score: float = 0.0) -> dict:
    """
    Serialise a DataFrame row into a clean dict for the API and frontend.
    Includes both in-league (radar) and cross-league (radar_x) percentile
    values for Chart.js dual-radar display.
    """
    pos = str(row.get("position_primary", "MF"))
    if pos not in RADAR_FEATURES:
        pos = "MF"

    radar_cfg = RADAR_FEATURES[pos]

    def _read_pct(cols):
        vals = []
        for col in cols:
            v = row.get(col, 0)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = 0.0
            # Values may already be 0-1 (from normalizer) or 0-100 (from capstone)
            if v <= 1.0:
                v = v * 100
            vals.append(round(v, 1))
        return vals

    radar_vals   = _read_pct(radar_cfg["cols"])
    # Cross-league percentile: use xcols if available, else fall back to in-league
    xcols        = radar_cfg.get("xcols", radar_cfg["cols"])
    radar_x_vals = _read_pct(xcols)
    if all(v == 0.0 for v in radar_x_vals):
        radar_x_vals = list(radar_vals)   # fall back to in-league data

    def _f(key, default=0.0):
        try:
            return round(float(row.get(key, default)), 2)
        except (TypeError, ValueError):
            return default

    def _i(key, default=0):
        try:
            return int(row.get(key, default))
        except (TypeError, ValueError):
            return default

    mv   = _f("market_value")
    pv   = _f("predicted_value_new") or _f("predicted_value")
    fv   = _f("future_value")
    gap  = round(fv - mv, 1)
    league = str(row.get("league", "EPL"))

    nineties = max(_i("minutes") / 90, 0.01)

    # Transfer confidence — computed once from core fields (Feature 18)
    _tc = compute_confidence({
        "age":            _i("age"),
        "contract_years": _i("contract_years") or 2,
        "minutes":        _i("minutes"),
        "market_value":   mv,
        "future_value":   fv,
        "peak_age":       _i("peak_age") or 27,
    })

    # Core per-90 stats - derive on the fly if normalizer didn't pre-compute
    goals_p90   = _f("goals_per90")  or round(_f("goals")   / nineties, 2)
    assists_p90 = _f("assists_per90") or round(_f("assists") / nineties, 2)
    sh_p90      = _f("sh_per90")     or round(_f("sh")       / nineties, 2)
    sot_p90     = _f("sot_per90")    or round(_f("sot")      / nineties, 2)

    # Shot efficiency
    sh_total  = _f("sh")
    g_per_sh  = _f("g_per_shot")  or (round(_f("goals") / sh_total, 3) if sh_total > 0 else 0.0)
    g_per_sot_val = _f("g_per_sot") or (
        round(_f("goals") / _f("sot"), 3) if _f("sot") > 0 else 0.0
    )

    # Keeper stats
    gk_savepct = _f("gk_savepct")
    gk_cspct   = _f("gk_cspct")
    gk_ga90    = _f("gk_ga90")

    return {
        "player":           str(row.get("player", "")),
        "team":             str(row.get("team", "")),
        "league":           league,
        "position":         pos,
        "age":              _i("age"),
        "nationality":      str(row.get("nationality", "")),
        "season":           str(row.get("season", "")),
        "minutes":          _i("minutes"),
        "market_value":     round(mv, 1),
        "predicted_value":  round(pv, 1),
        "future_value":     round(fv, 1),
        "value_gap":        gap,
        "psi":              _f("psi", 0.0),

        # Attacking - always computed even if xG unavailable
        "goals_per90":      goals_p90,
        "assists_per90":    assists_p90,
        "sh_per90":         sh_p90,
        "sot_per90":        sot_p90,
        "g_per_shot":       g_per_sh,
        "g_per_sot":        g_per_sot_val,

        # Legacy / EPL-capstone stats (will be 0 for multi-league data)
        "xg_per90":         _f("xg_per90"),
        "xag_per90":        _f("xag_per90"),
        "pass_acc":         _f("total_cmppct"),
        "aerial_pct":       _f("aerial_duels_wonpct"),
        "take_on_pct":      _f("take-ons_succpct"),

        # Defensive stats
        "tackles_per90":    round(_f("tackles_tkl") / nineties, 2),
        "interceptions":    _f("int"),
        "fouls_drawn":      _f("fouls_drawn"),

        # Goalkeeper stats
        "gk_savepct":       gk_savepct,
        "gk_cspct":         gk_cspct,
        "gk_ga90":          gk_ga90,

        # Contract data
        "contract_years":   _i("contract_years") or 2,
        "contract_expires": _i("contract_expires") or 2026,

        # Valuation confidence interval (80% prediction interval from quantile LightGBM)
        "mv_low":           round(_f("mv_low")  or max(0.1, mv * 0.70), 1),
        "mv_high":          round(_f("mv_high") or mv * 1.30, 1),
        "mv_uncertainty":   round(_f("mv_uncertainty") or mv * 0.30, 1),

        # Age curve — position-specific peak and current arc phase
        "peak_age":         _i("peak_age") or 27,
        "arc_phase":        str(row.get("arc_phase", "peak")),

        # Playing style cluster (Feature 11)
        "style_cluster":    _i("style_cluster"),
        "style_label":      str(row.get("style_label", "")),
        "umap_x":           round(_f("umap_x"), 4),
        "umap_y":           round(_f("umap_y"), 4),

        # Injury risk score (Feature 12) — usage-pattern flag, not medical
        "injury_risk":       _i("injury_risk"),
        "injury_risk_label": str(row.get("injury_risk_label", "Moderate")),
        "injury_risk_age":   _i("injury_risk_age"),
        "injury_risk_load":  _i("injury_risk_load"),

        "score":            round(score, 3),
        "league_difficulty": LEAGUE_DIFFICULTY.get(league, 1.0),
        "radar": {
            "labels": radar_cfg["labels"],
            "values": radar_vals,
        },
        "radar_x": {
            "labels": radar_cfg["labels"],
            "values": radar_x_vals,
        },

        # Transfer Confidence Score (Feature 18)
        "transfer_confidence": _tc["score"],
        "transfer_label":      _tc["label"],
        "transfer_color":      _tc["color"],
        "transfer_reasons":    _tc["reasons"],
        "transfer_factors":    _tc["factors"],
    }


def get_all_players() -> list[dict]:
    """Return all players as dicts (used for status/debug)."""
    df = load_players()
    return [player_to_dict(row) for _, row in df.iterrows()]
