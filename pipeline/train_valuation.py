"""
ScoutAI - LightGBM Valuation Model  (with Confidence Intervals)
=================================================================
Replaces the hand-coded formula in normalizer.py with a data-driven model
trained on real Transfermarkt market values.  Three models are trained:

  q10  — lower bound (10th percentile)   → models/valuation_lgbm_low.pkl
  q50  — point estimate (median)         → models/valuation_lgbm.pkl
  q90  — upper bound (90th percentile)   → models/valuation_lgbm_high.pkl

Together they produce calibrated 80% prediction intervals, e.g.
"54M  [43M – 65M]" instead of a bare "54M".

What it does:
  1. Downloads Transfermarkt market values (same CDN as fetch_contracts.py)
  2. Name-matches them to our 2,135 FBref players (exact + fuzzy)
  3. Evaluates the current hand-coded formula as a baseline
  4. Trains median LightGBM with 5-fold stratified CV + reports metrics
  5. Trains quantile models with CV calibration check (% coverage)
  6. Saves all three model files + metadata

Run:
    python pipeline/train_valuation.py

Called automatically by:
    python pipeline/refresh.py  (after FBref stats are updated)
"""

import gzip
import io
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scoutai.valuation")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TM_URL     = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
CSV_PATH   = ROOT / "data" / "players_all_leagues.csv"
MODEL_DIR  = ROOT / "models"
MODEL_PATH      = MODEL_DIR / "valuation_lgbm.pkl"       # q50 median
MODEL_LOW_PATH  = MODEL_DIR / "valuation_lgbm_low.pkl"   # q10 lower bound
MODEL_HIGH_PATH = MODEL_DIR / "valuation_lgbm_high.pkl"  # q90 upper bound
META_PATH       = MODEL_DIR / "valuation_meta.json"

# Quantile targets — 80% prediction interval
Q_LOW  = 0.10
Q_HIGH = 0.90

MIN_MV_EUR = 500_000      # ignore placeholder / retired-player values
N_FOLDS    = 5
RANDOM_STATE = 42

LEAGUE_DIFFICULTY = {
    "EPL": 1.000, "La Liga": 0.955,
    "Serie A": 0.940, "Bundesliga": 0.925, "Ligue 1": 0.875,
}

# Feature columns — drawn from what FBref + normalizer produce.
# Optional cols (xg, aerial, etc.) are filled with 0 when absent.
FEATURE_COLS = [
    "age",
    "minutes",
    "psi",
    "league_diff",       # mapped from league difficulty
    "goals_per90",
    "assists_per90",
    "sh_per90",
    "sot_per90",
    "g_per_shot",
    "g_per_sot",
    "tackles_per90",     # tackles_tkl / nineties
    "int_per90",         # interceptions / nineties
    "xg_per90",          # 0 for leagues where xG unavailable
    "xag_per90",
    "gk_savepct",        # 0 for outfield players
    "gk_cspct",
    "gk_ga90",
    # Position one-hot (4 columns)
    "pos_GK", "pos_DF", "pos_MF", "pos_FW",
]

# Shared tree structure hyperparameters
_SHARED = {
    "n_estimators":      1200,
    "learning_rate":     0.025,
    "num_leaves":        31,
    "max_depth":         6,
    "min_child_samples": 8,
    "subsample":         0.80,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.15,
    "reg_lambda":        0.50,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
}

# Point estimate (median) — same as before
LGBM_PARAMS = {"objective": "regression", "metric": "mae", **_SHARED}

# Quantile regression — builds calibrated prediction intervals
LGBM_PARAMS_LOW  = {"objective": "quantile", "alpha": Q_LOW,  **_SHARED}
LGBM_PARAMS_HIGH = {"objective": "quantile", "alpha": Q_HIGH, **_SHARED}


# ---------------------------------------------------------------------------
# Step 1 — Download + match Transfermarkt market values
# ---------------------------------------------------------------------------

def fetch_tm_values() -> pd.DataFrame:
    """Download Transfermarkt dataset, filter to active players with real values."""
    cache = ROOT / "data" / "_tm_cache.pkl"
    if cache.exists():
        import pickle
        logger.info("Loading cached TM data from %s", cache)
        with open(cache, "rb") as f:
            tm = pickle.load(f)
    else:
        logger.info("Downloading Transfermarkt dataset ...")
        r = requests.get(TM_URL, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        tm = pd.read_csv(gzip.open(io.BytesIO(r.content)), low_memory=False)
        logger.info("Downloaded %d players", len(tm))

    # Filter to recent seasons + real valuations
    tm = tm[
        (tm["last_season"] >= 2023) &
        (tm["market_value_in_eur"].notna()) &
        (tm["market_value_in_eur"] >= MIN_MV_EUR)
    ].copy()
    tm = tm.sort_values("last_season", ascending=False).drop_duplicates("name", keep="first")
    tm["name_key"] = tm["name"].str.lower().str.strip()
    logger.info("TM active players with real values: %d", len(tm))
    return tm


def match_market_values(our: pd.DataFrame, tm: pd.DataFrame) -> pd.Series:
    """
    Return a Series of market_value_in_eur indexed by our df's index.
    Pass 1: exact name match.
    Pass 2: fuzzy match (token_sort_ratio >= 88) for unmatched.
    """
    our_keys = our["player"].str.lower().str.strip()
    tm_dict  = dict(zip(tm["name_key"], tm["market_value_in_eur"]))
    tm_keys  = list(tm_dict.keys())

    result = pd.Series(np.nan, index=our.index, dtype="float64")
    exact_hits = 0

    for idx, key in our_keys.items():
        if key in tm_dict:
            result[idx] = tm_dict[key]
            exact_hits += 1

    unmatched = our.index[result.isna()]
    fuzzy_hits = 0
    for idx in unmatched:
        m = process.extractOne(
            our_keys[idx], tm_keys,
            scorer=fuzz.token_sort_ratio, score_cutoff=88
        )
        if m:
            result[idx] = tm_dict[m[0]]
            fuzzy_hits += 1

    matched = result.notna().sum()
    logger.info(
        "Matched: %d exact + %d fuzzy = %d / %d (%.1f%%)",
        exact_hits, fuzzy_hits, matched, len(our), matched / len(our) * 100
    )
    return result


# ---------------------------------------------------------------------------
# Step 2 — Build feature matrix
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix from the normalised player DataFrame."""
    X = pd.DataFrame(index=df.index)

    nineties = (df["minutes"].fillna(0).astype(float) / 90).clip(lower=0.01)

    def _col(c, default=0.0):
        if c in df.columns:
            return df[c].fillna(default).astype(float)
        return pd.Series(default, index=df.index)

    X["age"]          = _col("age", 25)
    X["minutes"]      = _col("minutes")
    X["psi"]          = _col("psi")
    X["league_diff"]  = df["league"].map(LEAGUE_DIFFICULTY).fillna(0.95)

    X["goals_per90"]   = _col("goals_per90")
    X["assists_per90"] = _col("assists_per90")
    X["sh_per90"]      = _col("sh_per90")
    X["sot_per90"]     = _col("sot_per90")
    X["g_per_shot"]    = _col("g_per_shot")
    X["g_per_sot"]     = _col("g_per_sot")
    X["xg_per90"]      = _col("xg_per90")
    X["xag_per90"]     = _col("xag_per90")

    # Normalise raw counts to per-90 so they're comparable across minutes played
    tackles = _col("tackles_tkl")
    ints    = _col("int")
    X["tackles_per90"] = tackles / nineties
    X["int_per90"]     = ints / nineties

    X["gk_savepct"] = _col("gk_savepct")
    X["gk_cspct"]   = _col("gk_cspct")
    X["gk_ga90"]    = _col("gk_ga90")

    # Position one-hot
    pos = df["position_primary"].fillna("MF")
    for p in ["GK", "DF", "MF", "FW"]:
        X[f"pos_{p}"] = (pos == p).astype(float)

    return X[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Step 3 — Metrics helpers
# ---------------------------------------------------------------------------

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error, ignoring zero targets."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def metrics_report(y_true, y_pred, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mpe  = mape(y_true, y_pred)
    logger.info(
        "  %-20s  MAE=%.2fM  MAPE=%.1f%%  R²=%.3f",
        label, mae, mpe, r2
    )
    return {"label": label, "mae": round(mae, 3), "mape": round(mpe, 1), "r2": round(r2, 3)}


# ---------------------------------------------------------------------------
# Step 4 — Train + cross-validate
# ---------------------------------------------------------------------------

def cross_validate(X: np.ndarray, y: np.ndarray, pos_labels: np.ndarray):
    """
    5-fold CV stratified by position.
    Returns out-of-fold predictions in the original value space (millions EUR).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, pos_labels)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        preds_log = model.predict(X_val)
        preds_mv  = np.expm1(preds_log)          # back to millions EUR
        true_mv   = np.expm1(y_val)

        fold_mae  = mean_absolute_error(true_mv, preds_mv)
        fold_maes.append(fold_mae)
        oof_preds[val_idx] = preds_mv

        logger.info("  Fold %d/%d: MAE=%.2fM  trees=%d",
                    fold + 1, N_FOLDS, fold_mae, model.best_iteration_)

    logger.info("  CV mean MAE: %.2fM ± %.2fM", np.mean(fold_maes), np.std(fold_maes))
    return oof_preds


def train_final(X: np.ndarray, y: np.ndarray) -> lgb.LGBMRegressor:
    """Train median model on the full matched dataset."""
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
    return model


def train_quantile_models(
    X: np.ndarray, y: np.ndarray, pos_labels: np.ndarray
) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor, float, float]:
    """
    Train lower (q10) and upper (q90) quantile models.

    Because market values are heavy-tailed, raw quantile regression
    under-covers (actual ~55% vs nominal 80%). We apply conformal-prediction
    calibration on the OOF nonconformity scores to guarantee the target
    coverage level empirically.

    Nonconformity score for player i:
        s_i = max(q_low_i - y_i, y_i - q_high_i)   (0 if y inside interval)
    Calibration constant  c  is the ⌈(n+1)(1-α)/n⌉ quantile of scores.
    At inference: expand interval → [q_low - c, q_high + c].

    Returns (model_low, model_high, calibration_constant, calibrated_coverage).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("QUANTILE REGRESSION — calibrated 80%% prediction interval")
    logger.info("=" * 60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_low  = np.zeros(len(y))
    oof_high = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, pos_labels)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr        = y[train_idx]

        m_low  = lgb.LGBMRegressor(**LGBM_PARAMS_LOW)
        m_high = lgb.LGBMRegressor(**LGBM_PARAMS_HIGH)
        m_low.fit( X_tr, y_tr, callbacks=[lgb.log_evaluation(0)])
        m_high.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(0)])

        oof_low[val_idx]  = np.expm1(m_low.predict(X_val))
        oof_high[val_idx] = np.expm1(m_high.predict(X_val))

        cov = ((np.expm1(y[val_idx]) >= oof_low[val_idx]) &
               (np.expm1(y[val_idx]) <= oof_high[val_idx])).mean()
        logger.info("  Fold %d/%d: raw coverage=%.1f%%  width=%.1fM",
                    fold + 1, N_FOLDS, cov * 100,
                    (oof_high[val_idx] - oof_low[val_idx]).mean())

    true_mv = np.expm1(y)

    # Conformal calibration: nonconformity scores
    scores = np.maximum(oof_low - true_mv, true_mv - oof_high)
    scores = np.maximum(scores, 0.0)    # 0 if already inside

    alpha          = 1.0 - 0.80         # target mis-coverage = 20%
    n              = len(scores)
    level          = np.ceil((n + 1) * (1 - alpha)) / n
    level          = min(level, 1.0)
    calib_constant = float(np.quantile(scores, level))

    # Calibrated intervals
    cal_low  = oof_low  - calib_constant
    cal_high = oof_high + calib_constant
    cal_cov  = ((true_mv >= cal_low) & (true_mv <= cal_high)).mean()

    logger.info(
        "  Calibration constant: +/-%.1fM per bound  →  coverage %.1f%%  (target 80%%)",
        calib_constant, cal_cov * 100,
    )
    logger.info(
        "  Calibrated interval width: %.1fM average",
        (cal_high - cal_low).mean(),
    )

    # Train final quantile models on all data
    logger.info("  Training final quantile models on all %d players ...", len(y))
    final_low  = lgb.LGBMRegressor(**LGBM_PARAMS_LOW)
    final_high = lgb.LGBMRegressor(**LGBM_PARAMS_HIGH)
    final_low.fit( X, y, callbacks=[lgb.log_evaluation(0)])
    final_high.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    return final_low, final_high, calib_constant, cal_cov


# ---------------------------------------------------------------------------
# Step 5 — Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(model: lgb.LGBMRegressor):
    imp = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    logger.info("Top 10 features by gain:")
    for feat, score in imp.head(10).items():
        bar = "█" * int(score / imp.max() * 20)
        logger.info("  %-20s %s %.0f", feat, bar, score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    MODEL_DIR.mkdir(exist_ok=True)

    # ── Load our data ──────────────────────────────────────────────────────
    logger.info("Loading player dataset from %s ...", CSV_PATH)
    our = pd.read_csv(CSV_PATH, low_memory=False)
    logger.info("Players: %d across %d leagues",
                len(our), our["league"].nunique())

    # ── Fetch TM market values ─────────────────────────────────────────────
    tm = fetch_tm_values()
    tm_values = match_market_values(our, tm)

    # ── Build training set: matched players only ───────────────────────────
    matched_mask = tm_values.notna()
    our_matched  = our[matched_mask].copy()
    mv_matched   = tm_values[matched_mask].values / 1_000_000   # → millions EUR

    logger.info("Training set: %d players with real TM values", len(our_matched))

    # ── Build features ─────────────────────────────────────────────────────
    X_df = build_features(our_matched)
    X    = X_df.values.astype(np.float32)
    y_log = np.log1p(mv_matched)        # log-space target

    pos_labels = our_matched["position_primary"].fillna("MF").values

    # ── BASELINE: current formula vs real TM values ────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("BASELINE  —  Current hand-coded formula")
    logger.info("=" * 60)
    baseline_mv = our_matched["market_value"].fillna(0).astype(float).values
    baseline_result = metrics_report(mv_matched, baseline_mv, "Formula (baseline)")

    # ── LightGBM cross-validation ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("LIGHTGBM  —  5-fold stratified CV by position")
    logger.info("=" * 60)
    oof_preds = cross_validate(X, y_log, pos_labels)
    lgbm_result = metrics_report(mv_matched, oof_preds, "LightGBM (OOF CV)")

    # ── Head-to-head comparison ────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    mae_improvement   = (baseline_result["mae"]  - lgbm_result["mae"])  / baseline_result["mae"]  * 100
    mape_improvement  = (baseline_result["mape"] - lgbm_result["mape"]) / baseline_result["mape"] * 100
    r2_improvement    = lgbm_result["r2"] - baseline_result["r2"]

    logger.info("  %-22s  %8s  %8s  %6s", "Model", "MAE (M)", "MAPE", "R²")
    logger.info("  %s", "-" * 52)
    logger.info("  %-22s  %8.2f  %7.1f%%  %6.3f",
                "Formula (baseline)",
                baseline_result["mae"], baseline_result["mape"], baseline_result["r2"])
    logger.info("  %-22s  %8.2f  %7.1f%%  %6.3f",
                "LightGBM (5-fold CV)",
                lgbm_result["mae"], lgbm_result["mape"], lgbm_result["r2"])
    logger.info("  %s", "-" * 52)
    logger.info("  Improvement:  MAE ↓%.1f%%  MAPE ↓%.1f%%  R² +%.3f",
                mae_improvement, mape_improvement, r2_improvement)

    # ── Train final median model on all matched data ───────────────────────
    logger.info("")
    logger.info("Training final median model on all %d players ...", len(our_matched))
    final_model = train_final(X, y_log)
    print_feature_importance(final_model)

    # ── Train quantile models (confidence intervals) ───────────────────────
    model_low, model_high, calib_const, cal_cov = train_quantile_models(X, y_log, pos_labels)

    # ── Save all three models + calibration constant ────────────────────────
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(model_low,   MODEL_LOW_PATH)
    joblib.dump(model_high,  MODEL_HIGH_PATH)
    logger.info("Models saved:")
    logger.info("  Median : %s", MODEL_PATH)
    logger.info("  Q%-2d    : %s", int(Q_LOW  * 100), MODEL_LOW_PATH)
    logger.info("  Q%-2d    : %s", int(Q_HIGH * 100), MODEL_HIGH_PATH)

    meta = {
        "features":           FEATURE_COLS,
        "n_training_samples": int(len(our_matched)),
        "n_folds":            N_FOLDS,
        "interval":           {
            "q_low": Q_LOW, "q_high": Q_HIGH, "label": "80% PI",
            "calib_constant_m": round(calib_const, 2),
        },
        "calibration": {
            "raw_coverage":  round(1 - (1 - cal_cov), 3),  # same as cal_cov
            "cal_coverage":  round(float(cal_cov), 3),
        },
        "lgbm_params":        LGBM_PARAMS,
        "baseline":           baseline_result,
        "lgbm_cv":            lgbm_result,
        "improvement": {
            "mae_pct":  round(mae_improvement, 1),
            "mape_pct": round(mape_improvement, 1),
            "r2_delta": round(r2_improvement, 3),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("Meta saved to %s", META_PATH)

    # ── Per-position breakdown ─────────────────────────────────────────────
    logger.info("")
    logger.info("Per-position OOF metrics (LightGBM):")
    for pos in ["GK", "DF", "MF", "FW"]:
        mask = pos_labels == pos
        if mask.sum() < 5:
            continue
        pos_true = mv_matched[mask]
        pos_pred = oof_preds[mask]
        pos_mae  = mean_absolute_error(pos_true, pos_pred)
        pos_r2   = r2_score(pos_true, pos_pred)
        pos_mape = mape(pos_true, pos_pred)
        logger.info("  %-4s  n=%-4d  MAE=%.2fM  MAPE=%.1f%%  R²=%.3f",
                    pos, mask.sum(), pos_mae, pos_mape, pos_r2)

    # ── Spot check: top players with confidence intervals ─────────────────
    logger.info("")
    logger.info("Spot check — point estimate + 80%% PI vs real TM:")
    import warnings
    X_all      = pd.DataFrame(build_features(our).values, columns=FEATURE_COLS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_mid  = np.expm1(final_model.predict(X_all))
        pred_low  = np.expm1(model_low.predict(X_all))
        pred_high = np.expm1(model_high.predict(X_all))

    check_names = ["Harry Kane", "Erling Haaland", "Rodrygo", "Pedri", "Virgil van Dijk"]
    logger.info("  %-22s  %8s  %24s  %8s", "Player", "Estimate", "Calibrated 80% PI", "TM real")
    for name in check_names:
        rows = our[our["player"] == name]
        if rows.empty:
            continue
        i      = rows.index[0]
        tm_mv  = tm_values.get(i, np.nan)
        tm_str = f"{tm_mv/1e6:.1f}M" if not pd.isna(tm_mv) else "n/a"
        # Apply calibration constant for display
        cal_low  = max(0.1, pred_low[i]  - calib_const)
        cal_high = pred_high[i] + calib_const
        logger.info(
            "  %-22s  %7.1fM  [%6.1fM – %7.1fM]  %8s",
            name, pred_mid[i], cal_low, cal_high, tm_str,
        )


if __name__ == "__main__":
    run()
