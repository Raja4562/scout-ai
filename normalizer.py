"""
ScoutAI - Cross-League Normalization
Computes within-league and cross-league percentile ranks for all key stat
columns. Adds _pct (within-league) and _xpct (cross-league adjusted) columns
so the radar chart and search scoring remain fair across EPL, La Liga,
Bundesliga, Serie A, and Ligue 1.

Also computes PSI (Player Strength Index) from scratch using a per-position
weighted formula, and estimates market/future value for players where no
external valuation is available.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("scoutai.normalizer")

# ---------------------------------------------------------------------------
# League difficulty coefficients
# Derived from historical UEFA coefficient data and cross-league xG studies.
# EPL is the reference league (1.000). A coefficient of 0.925 means a player
# scoring 0.50 g/90 in Bundesliga is equivalent to 0.50 * 0.925 = 0.463 g/90
# in the EPL for cross-league comparison purposes.
# ---------------------------------------------------------------------------

LEAGUE_DIFFICULTY = {
    "EPL":        1.000,
    "La Liga":    0.955,
    "Serie A":    0.940,
    "Bundesliga": 0.925,
    "Ligue 1":    0.875,
}

# ---------------------------------------------------------------------------
# Stat columns to normalise per position group.
# Mapping: column_name -> (higher_is_better, used_for_positions)
# ---------------------------------------------------------------------------

_STAT_META = {
    # --- Core attacking (available: goals/assists from standard, shots from shooting) ---
    "goals_per90":      (True,  {"FW", "MF", "DF"}),
    "assists_per90":    (True,  {"FW", "MF", "DF"}),
    "sh_per90":         (True,  {"FW", "MF"}),
    "sot_per90":        (True,  {"FW", "MF"}),
    "g_per_shot":       (True,  {"FW", "MF"}),
    "g_per_sot":        (True,  {"FW", "MF"}),
    "sh":               (True,  {"FW", "MF"}),         # raw shots (capstone compat)

    # --- Defense (from misc stat type) ---
    "tackles_tkl":      (True,  {"DF", "MF"}),
    "int":              (True,  {"DF", "MF"}),
    "fouls_drawn":      (True,  {"DF", "MF", "FW"}),   # winning fouls = positive
    "fouls_committed":  (False, {"DF", "MF"}),          # fewer = more disciplined

    # --- Legacy columns (EPL capstone only - absent in soccerdata scrape) ---
    # Kept so _add_percentile_columns skips them gracefully when col not present.
    "xg_per90":             (True,  {"FW", "MF"}),
    "xag_per90":            (True,  {"FW", "MF", "DF"}),
    "total_cmppct":         (True,  {"GK", "DF", "MF", "FW"}),
    "short_cmppct":         (True,  {"GK", "DF", "MF"}),
    "medium_cmppct":        (True,  {"GK", "DF", "MF"}),
    "long_cmppct":          (True,  {"GK", "DF", "MF"}),
    "long_cmp_rate":        (True,  {"GK"}),
    "progression_prgc":     (True,  {"DF", "MF", "FW"}),
    "progression_prgp":     (True,  {"GK", "DF", "MF"}),
    "progression_prgr":     (True,  {"MF", "FW"}),
    "carries_prgdist":      (True,  {"DF", "MF", "FW"}),
    "blocks_blocks":        (True,  {"DF", "MF"}),
    "aerial_duels_wonpct":  (True,  {"GK", "DF", "FW"}),
    "take-ons_succpct":     (True,  {"MF", "FW"}),
    "touches_def_pen":      (True,  {"GK", "DF"}),

    # --- Goalkeeper-specific (from keeper stat type) ---
    "gk_savepct":       (True,  {"GK"}),
    "gk_cspct":         (True,  {"GK"}),
    "gk_ga90":          (False, {"GK"}),   # lower GA/90 = higher rank
    "gk_sota":          (True,  {"GK"}),   # shots faced (workload)
    "gk_saves":         (True,  {"GK"}),   # saves count
    "gk_wins":          (True,  {"GK"}),
}

# All unique stat columns (used for percentile computation)
ALL_STAT_COLS = list(_STAT_META.keys())

# ---------------------------------------------------------------------------
# PSI weights per position
# PSI = weighted sum of normalised per-90 stats, scaled 0-1
# ---------------------------------------------------------------------------

_PSI_WEIGHTS = {
    # Weights use only columns confirmed available from soccerdata v1.9.
    # Legacy columns (xg, xag, aerial, progression) are kept in _STAT_META
    # for backward compat with the EPL capstone but excluded from PSI here.
    "FW": {
        "goals_per90": 0.35,
        "sh_per90":    0.20,
        "assists_per90": 0.15,
        "g_per_shot":  0.15,
        "sot_per90":   0.10,
        "g_per_sot":   0.05,
    },
    "MF": {
        "goals_per90":   0.25,
        "assists_per90": 0.25,
        "tackles_tkl":   0.20,
        "int":           0.15,
        "sh_per90":      0.10,
        "g_per_shot":    0.05,
    },
    "DF": {
        "tackles_tkl":   0.35,
        "int":           0.30,
        "assists_per90": 0.15,
        "goals_per90":   0.10,
        "fouls_drawn":   0.10,
    },
    "GK": {
        "gk_savepct":  0.35,
        "gk_cspct":    0.25,
        "gk_ga90":     0.20,
        "gk_saves":    0.10,
        "gk_wins":     0.10,
    },
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Given a raw multi-league player DataFrame, adds:
      {col}_pct   - within-(league, position) percentile rank  [0-1]
      {col}_xpct  - cross-league percentile using difficulty-adjusted stats [0-1]
      psi         - recalculated Player Strength Index  [0-1]
      market_value_est - estimated market value when market_value is 0
      injury_risk      - usage-pattern injury risk score 0-100 (Feature 12)

    Modifies a copy; does not mutate input.
    """
    logger.info("Running cross-league normalization on %d players ...", len(df))
    df = df.copy()

    # Ensure league column exists
    if "league" not in df.columns:
        df["league"] = "EPL"

    # Ensure position_primary exists
    if "position_primary" not in df.columns:
        df["position_primary"] = (
            df["position"].astype(str).str.split(",").str[0].str.strip()
        )

    # Fill numeric nulls
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)

    df = _add_derived_per90(df)
    df = _add_percentile_columns(df)
    df = _add_psi(df)
    df = _estimate_values(df)
    df = _add_contract_data(df)
    df = _compute_injury_risk(df)

    logger.info("Normalization complete. Columns: %d", len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_derived_per90(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-90 columns where they don't already exist."""
    nineties = (df["minutes"].fillna(0).astype(float) / 90).clip(lower=0.01)

    for raw_col, per90_col in [
        ("goals",   "goals_per90"),
        ("assists", "assists_per90"),
        ("xg",      "xg_per90"),
        ("xag",     "xag_per90"),
        ("sh",      "sh_per90"),
        ("sot",     "sot_per90"),
    ]:
        if raw_col in df.columns and per90_col not in df.columns:
            df[per90_col] = df[raw_col].fillna(0).astype(float) / nineties

    # Shot accuracy: goals per shot (finishing efficiency)
    if "g_per_shot" not in df.columns:
        if "goals" in df.columns and "sh" in df.columns:
            shots = df["sh"].fillna(0).astype(float)
            df["g_per_shot"] = np.where(
                shots > 0,
                df["goals"].fillna(0).astype(float) / shots,
                0.0,
            )

    # Goals per shot on target (clinical finishing)
    if "g_per_sot" not in df.columns:
        if "goals" in df.columns and "sot" in df.columns:
            sot = df["sot"].fillna(0).astype(float)
            df["g_per_sot"] = np.where(
                sot > 0,
                df["goals"].fillna(0).astype(float) / sot,
                0.0,
            )

    # GK: goals-against per 90 (if gk_ga90 absent but gk_ga present)
    if "gk_ga90" not in df.columns and "gk_ga" in df.columns:
        df["gk_ga90"] = df["gk_ga"].fillna(0).astype(float) / nineties

    # GK: saves per 90 (workload-normalised)
    if "gk_saves" in df.columns and "gk_saves_per90" not in df.columns:
        df["gk_saves_per90"] = df["gk_saves"].fillna(0).astype(float) / nineties

    # long_cmp_rate: fraction of long passes completed (EPL capstone compat)
    if "long_cmppct" in df.columns and "long_cmp_rate" not in df.columns:
        df["long_cmp_rate"] = df["long_cmppct"].fillna(0) / 100.0

    return df


def _rank_pct(series: pd.Series) -> pd.Series:
    """Percentile rank 0-1, ties averaged, NaN -> 0."""
    return series.rank(method="average", pct=True).fillna(0.0)


def _add_percentile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every stat in ALL_STAT_COLS:
      col_pct  - percentile within (league, position_primary)
      col_xpct - percentile within position_primary, using league-difficulty-adjusted values
    """
    diff_map = df["league"].map(LEAGUE_DIFFICULTY).fillna(1.0)

    for col in ALL_STAT_COLS:
        if col not in df.columns:
            continue

        raw = df[col].fillna(0).astype(float)
        higher_better, _ = _STAT_META[col]

        # For "lower is better" stats we flip the rank. All current stats are higher=better.
        if not higher_better:
            raw = -raw

        # --- Within-league-position percentile ---
        pct_col = f"{col}_pct"
        df[pct_col] = (
            df.groupby(["league", "position_primary"])[col]
            .transform(_rank_pct)
        )

        # --- Cross-league percentile (difficulty-adjusted) ---
        adjusted = raw * diff_map
        temp = pd.DataFrame(
            {"_adj": adjusted, "position_primary": df["position_primary"]},
            index=df.index,
        )
        xpct_col = f"{col}_xpct"
        df[xpct_col] = temp.groupby("position_primary")["_adj"].transform(_rank_pct)

    return df


def _add_psi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PSI from within-league-position percentile columns.
    Each position has its own weighted formula.
    Result is in [0, 1].
    """
    psi_vals = np.zeros(len(df), dtype=np.float64)

    for pos, weights in _PSI_WEIGHTS.items():
        mask = df["position_primary"] == pos
        if not mask.any():
            continue

        total_w = sum(weights.values())
        pos_psi = np.zeros(mask.sum(), dtype=np.float64)

        for col, w in weights.items():
            pct_col = f"{col}_pct"
            if pct_col in df.columns:
                pos_psi += (df.loc[mask, pct_col].fillna(0).values * w)
            # If column missing, that weight contributes 0

        psi_vals[mask.values] = pos_psi / total_w

    df["psi"] = np.round(psi_vals, 4)
    return df


# ---------------------------------------------------------------------------
# Position-specific age curves
# ---------------------------------------------------------------------------
#
# Each position has a different career arc. The multiplier is the ratio of
# a player's *expected peak value* to their current market value.
#
#   multiplier > 1 → player has upside / still growing toward peak
#   multiplier = 1 → player is at their peak window
#   multiplier < 1 → player has declined past peak
#
# Grounded in transfer-market research:
#   GK:  peak 28-33 — last to peak (positional read + reflexes), very slow decline
#   DF:  peak 26-30 — leadership + anticipation compensate for physical fade
#   MF:  peak 25-29 — technique sustains value; holding MFs peak later, #10s earlier
#   FW:  peak 24-27 — pace-dependent; sharpest rise AND sharpest decline
#
# The arc is modelled as three piecewise-linear segments:
#   1. Growth  (age 18 → peak_start): multiplier falls from mult_at_18 to 1.0
#   2. Plateau (peak_start → peak_end): multiplier = 1.0
#   3. Decline (age > peak_end): multiplier = 1 - decline_rate * years, ≥ floor
# ---------------------------------------------------------------------------

POSITION_AGE_CURVES: dict[str, dict] = {
    "FW": {
        "mult_at_18":    1.75,   # 75% upside from 18yo to peak
        "peak_start":    24,     # peak window opens
        "peak_end":      27,     # peak window closes
        "decline_rate":  0.090,  # 9% per year after peak_end
        "floor":         0.22,   # minimum floor (age ~35+)
        "label":         "Forwards peak at 24–27",
    },
    "MF": {
        "mult_at_18":    1.60,
        "peak_start":    25,
        "peak_end":      29,
        "decline_rate":  0.075,
        "floor":         0.25,
        "label":         "Midfielders peak at 25–29",
    },
    "DF": {
        "mult_at_18":    1.50,
        "peak_start":    26,
        "peak_end":      30,
        "decline_rate":  0.070,
        "floor":         0.28,
        "label":         "Defenders peak at 26–30",
    },
    "GK": {
        "mult_at_18":    1.45,
        "peak_start":    28,
        "peak_end":      33,
        "decline_rate":  0.050,
        "floor":         0.40,
        "label":         "Goalkeepers peak at 28–33",
    },
}


def _pos_multiplier(age: float, pos: str) -> float:
    """
    Compute the future-value multiplier for a single player given age + position.
    Returns the ratio of expected peak value to current market value.
    """
    c = POSITION_AGE_CURVES.get(pos, POSITION_AGE_CURVES["MF"])
    peak_start   = c["peak_start"]
    peak_end     = c["peak_end"]
    mult_at_18   = c["mult_at_18"]
    decline_rate = c["decline_rate"]
    floor        = c["floor"]

    if age <= peak_start:
        # Linear growth: mult_at_18 at age 18 → 1.0 at peak_start
        span = max(peak_start - 18, 1)
        return mult_at_18 - (mult_at_18 - 1.0) * (age - 18) / span
    elif age <= peak_end:
        # Plateau
        return 1.0
    else:
        years_past = age - peak_end
        return max(floor, 1.0 - decline_rate * years_past)


def _arc_phase(age: float, pos: str) -> str:
    """Return the arc phase label for a player: pre-peak / peak / post-peak."""
    c = POSITION_AGE_CURVES.get(pos, POSITION_AGE_CURVES["MF"])
    if age < c["peak_start"]:
        return "pre-peak"
    elif age <= c["peak_end"]:
        return "peak"
    else:
        return "post-peak"


def _position_future_factors(
    age: "pd.Series", pos: "pd.Series"
) -> tuple["pd.Series", "pd.Series", "pd.Series"]:
    """
    Vectorised application of position-specific age curves.

    Returns:
        future_factors  – Series of multipliers aligned to df index
        peak_ages       – Series of int midpoint peak age per position
        arc_phases      – Series of 'pre-peak' / 'peak' / 'post-peak'
    """
    import pandas as _pd

    factors = _pd.Series(index=age.index, dtype="float64")
    peaks   = _pd.Series(index=age.index, dtype="int32")
    phases  = _pd.Series(index=age.index, dtype="object")

    for p, curve in POSITION_AGE_CURVES.items():
        mask = pos == p
        if not mask.any():
            continue
        mid_peak = (curve["peak_start"] + curve["peak_end"]) // 2
        factors[mask] = age[mask].apply(lambda a: _pos_multiplier(a, p))
        peaks[mask]   = mid_peak
        phases[mask]  = age[mask].apply(lambda a: _arc_phase(a, p))

    # Fill any unrecognised positions with the MF curve
    unset = factors.isna()
    if unset.any():
        factors[unset] = age[unset].apply(lambda a: _pos_multiplier(a, "MF"))
        peaks[unset]   = 27
        phases[unset]  = age[unset].apply(lambda a: _arc_phase(a, "MF"))

    return factors, peaks, phases


def _load_valuation_models():
    """
    Load the three LightGBM valuation models (q10, q50, q90) plus the
    conformal calibration constant from training metadata.

    Returns (model_low, model_mid, model_high, calib_constant).
    Any component may be None if the file is missing.
    """
    import json as _json
    from pathlib import Path
    import joblib

    model_dir = Path(__file__).resolve().parent / "models"
    paths = {
        "mid":  model_dir / "valuation_lgbm.pkl",
        "low":  model_dir / "valuation_lgbm_low.pkl",
        "high": model_dir / "valuation_lgbm_high.pkl",
        "meta": model_dir / "valuation_meta.json",
    }

    if not paths["mid"].exists():
        return None, None, None, 0.0

    try:
        model_mid  = joblib.load(paths["mid"])
        model_low  = joblib.load(paths["low"])  if paths["low"].exists()  else None
        model_high = joblib.load(paths["high"]) if paths["high"].exists() else None

        # Load conformal calibration constant from meta file
        calib_const = 0.0
        if paths["meta"].exists():
            try:
                meta = _json.loads(paths["meta"].read_text())
                calib_const = float(
                    meta.get("interval", {}).get("calib_constant_m", 0.0)
                )
            except Exception:
                pass

        have_pi = model_low is not None and model_high is not None
        logger.info(
            "Loaded LightGBM valuation models (median%s, calib=±%.1fM)",
            " + q10/q90 PI" if have_pi else " only",
            calib_const,
        )
        return model_low, model_mid, model_high, calib_const
    except Exception as exc:
        logger.warning("Could not load valuation models: %s — using formula fallback", exc)
        return None, None, None, 0.0


def _build_model_features(df: pd.DataFrame) -> "pd.DataFrame":
    """Build the feature DataFrame expected by the LightGBM valuation model."""
    nineties = (df["minutes"].fillna(0).astype(float) / 90).clip(lower=0.01)

    def _col(c, default=0.0):
        if c in df.columns:
            return df[c].fillna(default).astype(float)
        return pd.Series(default, index=df.index, dtype="float64")

    pos = df["position_primary"].fillna("MF")

    feat = pd.DataFrame({
        "age":           _col("age", 25),
        "minutes":       _col("minutes"),
        "psi":           _col("psi"),
        "league_diff":   df["league"].map(LEAGUE_DIFFICULTY).fillna(0.95),
        "goals_per90":   _col("goals_per90"),
        "assists_per90": _col("assists_per90"),
        "sh_per90":      _col("sh_per90"),
        "sot_per90":     _col("sot_per90"),
        "g_per_shot":    _col("g_per_shot"),
        "g_per_sot":     _col("g_per_sot"),
        "tackles_per90": _col("tackles_tkl") / nineties,
        "int_per90":     _col("int") / nineties,
        "xg_per90":      _col("xg_per90"),
        "xag_per90":     _col("xag_per90"),
        "gk_savepct":    _col("gk_savepct"),
        "gk_cspct":      _col("gk_cspct"),
        "gk_ga90":       _col("gk_ga90"),
        "pos_GK":        (pos == "GK").astype(float),
        "pos_DF":        (pos == "DF").astype(float),
        "pos_MF":        (pos == "MF").astype(float),
        "pos_FW":        (pos == "FW").astype(float),
    }, index=df.index)

    return feat


def _estimate_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate market_value for all players in the pipeline.

    Priority:
      1. LightGBM model (models/valuation_lgbm.pkl) — trained on real
         Transfermarkt values for 1,959 players across 5 leagues.
         Run `python pipeline/train_valuation.py` to build/update it.
      2. Hand-coded formula fallback — position median × PSI × league × minutes.
         Used when the model file is absent (first run before training).

    future_value = market_value × future_multiplier(age)
    This represents the expected trajectory: young players appreciate,
    veterans decline. The gap (future_value - market_value) drives the
    BUY / SELL / HOLD verdicts in the scouting reports.
    """
    if "market_value" not in df.columns:
        df["market_value"] = 0.0
    if "future_value" not in df.columns:
        df["future_value"] = 0.0
    if "predicted_value" not in df.columns:
        df["predicted_value"] = df["market_value"]

    age = df["age"].fillna(25).astype(float)
    pos = df["position_primary"].fillna("MF")

    future_factors, peak_ages, arc_phases = _position_future_factors(age, pos)

    # ── Try LightGBM models first ──────────────────────────────────────────
    model_low, model_mid, model_high, calib_const = _load_valuation_models()
    if model_mid is not None:
        try:
            import warnings
            X_feat = _build_model_features(df)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mv_col = np.clip(np.round(np.expm1(model_mid.predict(X_feat)),  1), 0.1, 250.0)

                # Confidence interval columns (if quantile models available)
                if model_low is not None and model_high is not None:
                    mv_low_raw  = np.expm1(model_low.predict(X_feat))
                    mv_high_raw = np.expm1(model_high.predict(X_feat))
                    # Apply conformal calibration: expand each bound by calib_const
                    mv_low_cal  = mv_low_raw  - calib_const
                    mv_high_cal = mv_high_raw + calib_const
                    # Clip and enforce low <= mid <= high
                    mv_low_col  = np.clip(np.round(mv_low_cal,  1), 0.1, mv_col)
                    mv_high_col = np.clip(np.round(mv_high_cal, 1), mv_col, 300.0)
                else:
                    # Fallback: symmetric ±30% interval
                    mv_low_col  = np.round(mv_col * 0.70, 1)
                    mv_high_col = np.round(mv_col * 1.30, 1)

            df["market_value"]    = mv_col
            df["predicted_value"] = mv_col
            df["future_value"]    = np.round(mv_col * future_factors.values, 1)
            df["mv_low"]          = mv_low_col
            df["mv_high"]         = mv_high_col
            df["mv_uncertainty"]  = np.round((mv_high_col - mv_low_col) / 2, 1)
            df["peak_age"]        = peak_ages.values
            df["arc_phase"]       = arc_phases.values

            logger.info(
                "LightGBM valuations applied to %d players  "
                "(median=%.1fM  80%% PI avg width=%.1fM)",
                len(df),
                float(np.median(mv_col)),
                float((mv_high_col - mv_low_col).mean()),
            )
            return df
        except Exception as exc:
            logger.warning("LightGBM prediction failed (%s) — using formula fallback", exc)

    # ── Formula fallback (no model file yet) ──────────────────────────────
    logger.info("Using hand-coded formula for market value estimation")

    _base = {"GK": 8.0, "DF": 12.0, "MF": 18.0, "FW": 22.0}
    diff_map = df["league"].map(LEAGUE_DIFFICULTY).fillna(1.0)
    psi  = df["psi"].fillna(0.3).astype(float)
    mins = df["minutes"].fillna(0).astype(float)

    base_vals   = df["position_primary"].map(_base).fillna(15.0)
    psi_factor  = 0.3 + psi * 1.4
    mins_factor = np.clip(mins / 2500, 0.3, 1.0)

    def age_curve(a):
        if a < 22:   return 0.7 + (a - 18) * 0.075
        elif a < 27: return 1.0 + (a - 22) * 0.03
        elif a < 30: return 1.15 - (a - 27) * 0.06
        else:        return max(0.3, 0.97 - (a - 30) * 0.08)

    age_factors  = age.apply(age_curve)
    estimated_mv = np.round(base_vals * psi_factor * diff_map * mins_factor * age_factors, 1)

    needs_mv = df["market_value"].fillna(0) == 0
    if needs_mv.any():
        df.loc[needs_mv, "market_value"]    = estimated_mv[needs_mv]
        df.loc[needs_mv, "predicted_value"] = estimated_mv[needs_mv]
        logger.info("Formula: estimated market values for %d players", needs_mv.sum())

    mv_col = df["market_value"].fillna(0).astype(float).values
    df["future_value"]   = np.round(mv_col * future_factors.values, 1)
    df["mv_low"]         = np.round(mv_col * 0.70, 1)
    df["mv_high"]        = np.round(mv_col * 1.30, 1)
    df["mv_uncertainty"] = np.round(mv_col * 0.30, 1)
    df["peak_age"]       = peak_ages.values
    df["arc_phase"]      = arc_phases.values
    logger.info("Applied future multiplier to %d players", len(df))

    return df


def _compute_injury_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usage-pattern injury risk score (0–100) — Feature 12.

    This is NOT a medical model.  It is a scouting flag that synthesises
    publicly observable usage patterns known to correlate with injury
    frequency in sports science literature:

      1. Age risk      (35%) — risk accelerates post-peak; young players are
                               resilient, 30+ show slower recovery.
      2. Load risk     (35%) — total season minutes as a proxy for cumulative
                               fatigue.  Extreme loads (>3000 min) spike risk.
      3. Position risk (15%) — physical demand by role; wide forwards and
                               defensive midfielders face the most contact/sprint
                               load; GKs the least.
      4. Intensity risk(15%) — high-PSI × high-minutes players running at
                               maximum physical output for extended periods.

    Output columns added:
      injury_risk          int  0-100
      injury_risk_label    str  "Low" / "Moderate" / "Elevated" / "High" / "Very High"
      injury_risk_age      int  0-100  (age sub-score)
      injury_risk_load     int  0-100  (load sub-score)
    """
    df = df.copy()

    age  = df["age"].fillna(25).astype(float)
    mins = df["minutes"].fillna(0).astype(float)
    pos  = df["position_primary"].fillna("MF").astype(str)
    psi  = df["psi"].fillna(0.5).astype(float)

    # ── 1. Age risk ────────────────────────────────────────────────────────
    # Piecewise: low before 24, gradual rise to 28, steeper beyond 30
    def _age_risk(a: float) -> float:
        if a <= 21:   return 0.10
        if a <= 24:   return 0.15 + (a - 21) * 0.025   # 0.15 → 0.225
        if a <= 27:   return 0.225 + (a - 24) * 0.050  # 0.225 → 0.375
        if a <= 30:   return 0.375 + (a - 27) * 0.075  # 0.375 → 0.600
        if a <= 33:   return 0.600 + (a - 30) * 0.090  # 0.600 → 0.870
        return min(0.95, 0.870 + (a - 33) * 0.040)

    age_risk = age.apply(_age_risk)

    # ── 2. Load risk ───────────────────────────────────────────────────────
    # 38-game season max = 3420 min.  Thresholds based on top-flight norms.
    def _load_risk(m: float) -> float:
        if m < 900:   return 0.10   # managed / part-time / recovering
        if m < 1500:  return 0.20
        if m < 2000:  return 0.35
        if m < 2500:  return 0.50
        if m < 2800:  return 0.65
        if m < 3100:  return 0.78
        if m < 3300:  return 0.88
        return 0.95                  # marathon-man loads (>3300 min)

    load_risk = mins.apply(_load_risk)

    # ── 3. Position base risk ──────────────────────────────────────────────
    # FW/wide players: explosive sprints, collisions
    # DM / pressing MF: high running volume + duels
    # CB: aerial duels, physical battles
    # GK: minimal sprint/contact compared to outfielders
    pos_base = pos.map({
        "FW": 0.65,
        "MF": 0.58,
        "DF": 0.55,
        "GK": 0.28,
    }).fillna(0.58)

    # Boost for high-pressing forwards (high tackles_tkl rate signals press work)
    nineties    = (mins / 90).clip(lower=0.1)
    tackle_p90  = df["tackles_tkl"].fillna(0).astype(float) / nineties
    fouls_p90   = df["fouls_drawn"].fillna(0).astype(float) / nineties

    # Pressing FWs and physical DMs attract more contact; boost their pos_base
    # (capped so it can't push base above 0.80)
    contact_boost = np.where(
        pos.isin(["FW", "MF"]),
        (tackle_p90.clip(upper=5) / 5 * 0.10 + fouls_p90.clip(upper=5) / 5 * 0.05),
        0.0,
    )
    pos_risk = (pos_base + contact_boost).clip(upper=0.80)

    # ── 4. Intensity risk ─────────────────────────────────────────────────
    # A high-PSI player burning minutes at extreme output is at elevated risk.
    # Scale: PSI * load fraction (minutes / 3420)
    load_frac      = (mins / 3420).clip(upper=1.0)
    intensity_risk = (psi * load_frac).clip(upper=1.0)

    # ── Weighted composite ─────────────────────────────────────────────────
    risk_raw = (
        0.35 * age_risk
        + 0.35 * load_risk
        + 0.15 * pos_risk
        + 0.15 * intensity_risk
    )

    # Clip to [0,1] and scale to 0-100
    risk_score = (risk_raw.clip(0.0, 1.0) * 100).round().astype(int)

    # ── Tier labels ────────────────────────────────────────────────────────
    def _label(s: int) -> str:
        if s <= 24:  return "Low"
        if s <= 44:  return "Moderate"
        if s <= 59:  return "Elevated"
        if s <= 74:  return "High"
        return "Very High"

    risk_label = risk_score.apply(_label)

    df["injury_risk"]       = risk_score.values
    df["injury_risk_label"] = risk_label.values
    df["injury_risk_age"]   = (age_risk  * 100).round().astype(int).values
    df["injury_risk_load"]  = (load_risk * 100).round().astype(int).values

    # Summarise distribution for the log
    dist = risk_label.value_counts().to_dict()
    logger.info("Injury risk distribution: %s", dist)
    return df


def _add_contract_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate realistic, deterministic contract years for each player.
    Uses md5(player+team) as a repeatable seed so the same player always
    gets the same contract length across pipeline re-runs.

    Distribution is age-weighted and market-value adjusted:
      Under 21  : mostly 3-5 years (clubs lock in young talent)
      21-23     : 2-4 years typical
      24-26     : 1-3 years typical
      27-29     : 1-2 years typical
      30-31     : 1-2 years, weighted toward 1
      32+       : overwhelmingly 1 year

    High-market-value players (>25M) are shifted toward longer contracts
    because clubs protect their most expensive assets.

    Always regenerates so a fresh pipeline run stays consistent.
    """
    import hashlib

    ages = df["age"].fillna(25).astype(int).values
    mvs  = df["market_value"].fillna(0).astype(float).values
    players = df["player"].astype(str).values
    teams   = df["team"].astype(str).values

    contract_yrs = []
    for player, team, age, mv in zip(players, teams, ages, mvs):
        seed = int(hashlib.md5(f"{player}|{team}".encode()).hexdigest()[:8], 16)
        r    = seed % 100  # 0..99

        # Cumulative probability breakpoints for 1, 2, 3, 4, 5 years
        if age < 21:
            cuts = [5,  18, 40, 72, 100]   # 5%/13%/22%/32%/28%
        elif age < 24:
            cuts = [12, 30, 58, 84, 100]   # 12%/18%/28%/26%/16%
        elif age < 27:
            cuts = [22, 48, 72, 92, 100]   # 22%/26%/24%/20%/8%
        elif age < 30:
            cuts = [36, 64, 84, 97, 100]   # 36%/28%/20%/13%/3%
        elif age < 32:
            cuts = [50, 78, 94, 100, 100]  # 50%/28%/16%/6%/0%
        else:
            cuts = [62, 88, 98, 100, 100]  # 62%/26%/10%/2%/0%

        # High-MV players secured with longer deals - shift breakpoints left
        if mv > 40:
            cuts = [max(0, c - 22) for c in cuts]
        elif mv > 25:
            cuts = [max(0, c - 14) for c in cuts]
        elif mv > 12:
            cuts = [max(0, c - 7)  for c in cuts]

        # Ensure the last breakpoint always reaches 100 so no player
        # falls through to the default 5-year value incorrectly
        cuts[-1] = 100

        years = next((i + 1 for i, t in enumerate(cuts) if r < t), 5)

        # Hard age caps - no club gives long deals to older players
        if age >= 34:
            years = min(years, 1)
        elif age >= 32:
            years = min(years, 2)
        elif age >= 30:
            years = min(years, 3)

        contract_yrs.append(years)

    # Only fill in players that don't already have real contract data.
    # fetch_contracts.py populates real TM values first; normalizer provides
    # synthetic fallback only for the unmatched remainder.
    # Only fill genuinely missing values. contract_years == 0 is valid (free agent).
    if "contract_years" in df.columns:
        needs_contract = df["contract_years"].isna()
    else:
        needs_contract = pd.Series(True, index=df.index)

    if needs_contract.any():
        df.loc[needs_contract, "contract_years"]   = [contract_yrs[i] for i in np.where(needs_contract)[0]]
        logger.info(
            "Contract data: filled %d synthetic values (existing real data preserved)",
            needs_contract.sum(),
        )
    else:
        logger.info("Contract data: all players already have values (no synthetic fallback needed)")

    # Always (re)compute contract_expires from contract_years
    if "contract_expires" not in df.columns or df["contract_expires"].isna().any():
        df["contract_expires"] = 2026 + df["contract_years"].fillna(2)

    dist = dict(zip(*np.unique(df["contract_years"].dropna().astype(int), return_counts=True)))
    logger.info("Contract distribution: %s", dist)
    return df
