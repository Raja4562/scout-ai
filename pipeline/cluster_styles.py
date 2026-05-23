"""
ScoutAI – Playing Style Clustering  (Feature 11)
=================================================
Assigns each player a named style archetype via per-position K-Means on
performance features, then projects all players into 2-D UMAP space for the
scatter-plot UI.

Outputs:
  models/style_clusters.pkl   – per-position {scaler, kmeans, label_map, features}
  data/players_all_leagues.csv (in-place update) – adds:
    style_cluster   int    e.g. 2
    style_label     str    e.g. "Press-Resistant 8"
    umap_x / umap_y float  2-D UMAP coordinates

Usage:
  python pipeline/cluster_styles.py           # run (or skip if pkl exists)
  python pipeline/cluster_styles.py --rebuild # force retrain
"""

import os, sys, pickle, argparse, logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import DATA_FILE_MULTI, DATA_FILE, MIN_MINUTES

MODEL_DIR   = os.path.join(ROOT, "models")
CLUSTER_PKL = os.path.join(MODEL_DIR, "style_clusters.pkl")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("scoutai.cluster")

# ---------------------------------------------------------------------------
# Per-position config
# ---------------------------------------------------------------------------
# Features are raw per-90 values derived inside this script.
# Archetype rules: first matching rule (any required feature in top-k highest
# z-score centroids) wins; last entry is the default.

POSITION_CONFIG = {
    "FW": {
        "k": 5,
        "features": [
            "goals_per90", "assists_per90", "sh_per90", "sot_per90",
            "g_per_shot",  "g_per_sot",    "tackles_per90", "fouls_drawn_per90",
        ],
        "archetypes": [
            # (required_top_features,  label)
            (["g_per_shot",  "goals_per90"],   "Poacher"),
            (["assists_per90"],                 "Creative Forward"),
            (["tackles_per90"],                 "Pressing Forward"),
            (["sh_per90",    "sot_per90"],      "Target Man"),
            ([],                                "Inverted Winger"),    # default
        ],
    },
    "MF": {
        "k": 6,
        "features": [
            "goals_per90", "assists_per90", "sh_per90",
            "tackles_per90", "int_per90",  "fouls_drawn_per90",
        ],
        "archetypes": [
            (["tackles_per90", "int_per90"],    "Ball Winner"),
            (["goals_per90",   "sh_per90"],     "Box-to-Box"),
            (["assists_per90", "goals_per90"],  "Attacking Midfielder"),
            (["assists_per90"],                 "Playmaker"),
            (["tackles_per90"],                 "Press-Resistant 8"),
            ([],                                "Holding Midfielder"),  # default
        ],
    },
    "DF": {
        "k": 5,
        "features": [
            "tackles_per90", "int_per90",
            "goals_per90",   "assists_per90",
            "fouls_drawn_per90",
        ],
        "archetypes": [
            (["goals_per90",   "assists_per90"], "Attacking Fullback"),
            (["assists_per90"],                  "Ball-Playing CB"),
            (["tackles_per90", "int_per90"],     "Defensive Anchor"),
            (["int_per90"],                      "Sweeper CB"),
            ([],                                 "Stopper"),            # default
        ],
    },
    "GK": {
        "k": 3,
        "features": ["gk_savepct", "gk_cspct", "gk_ga90_inv"],
        "archetypes": [
            (["gk_savepct"],  "Shot-Stopper"),
            (["gk_cspct"],    "Sweeper Keeper"),
            ([],              "Distribution GK"),                       # default
        ],
    },
}

# Features used for the global UMAP projection (all positions together).
# These are the normalizer-generated percentile columns (0–1, fully populated).
UMAP_FEATURES = [
    "goals_per90_pct", "assists_per90_pct", "sh_per90_pct",
    "sot_per90_pct",   "g_per_shot_pct",    "g_per_sot_pct",
    "int_pct",         "tackles_tkl_pct",   "fouls_drawn_pct",
    "gk_savepct_pct",  "gk_cspct_pct",      "gk_ga90_pct",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-90 rate columns needed for clustering."""
    nineties = (df["minutes"].fillna(0) / 90).clip(lower=0.1)
    df = df.copy()
    df["tackles_per90"]     = df["tackles_tkl"].fillna(0) / nineties
    df["int_per90"]         = df["int"].fillna(0)         / nineties
    df["fouls_drawn_per90"] = df["fouls_drawn"].fillna(0) / nineties
    df["gk_ga90_inv"]       = 1.0 / (df["gk_ga90"].fillna(2.0).clip(lower=0.1))
    return df


# ---------------------------------------------------------------------------
# Archetype naming
# ---------------------------------------------------------------------------
def _name_clusters(centroids: np.ndarray, features: list[str],
                   rules: list[tuple]) -> dict[int, str]:
    """
    For each cluster centroid, compute z-scores across clusters and find
    which features are highest.  Apply the first archetype rule whose
    required features are all in the top-3 highest z-score features.
    """
    if len(centroids) == 1:
        return {0: rules[-1][1]}

    # z-score each feature across centroids
    std = centroids.std(axis=0).clip(min=1e-6)
    z   = (centroids - centroids.mean(axis=0)) / std   # (k, n_features)

    label_map: dict[int, str] = {}
    used_labels: set[str]     = set()

    for cid in range(len(centroids)):
        top_idx  = np.argsort(z[cid])[::-1]           # features sorted by z desc
        top_feat = [features[i] for i in top_idx[:3]]  # top-3 features

        chosen = rules[-1][1]   # default
        for req_features, lbl in rules[:-1]:
            if lbl in used_labels:
                continue
            if not req_features:               # shouldn't be a non-default rule
                chosen = lbl
                break
            if all(f in top_feat for f in req_features):
                chosen = lbl
                break
            # Relax: just the first required feature must appear in top-3
            if req_features and req_features[0] in top_feat:
                chosen = lbl
                break
        used_labels.add(chosen)
        label_map[cid] = chosen

    return label_map


# ---------------------------------------------------------------------------
# Per-position clustering
# ---------------------------------------------------------------------------
def _cluster_position(df_pos: pd.DataFrame, pos: str, cfg: dict,
                      seed: int = 42) -> tuple[pd.Series, pd.Series, object, StandardScaler]:
    """
    Fit K-Means for one position group.
    Returns (cluster_ids, style_labels, kmeans, scaler).
    """
    feats    = cfg["features"]
    k        = cfg["k"]
    rules    = cfg["archetypes"]

    X = df_pos[feats].fillna(0).values

    scaler   = StandardScaler()
    Xs       = scaler.fit_transform(X)

    km       = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    labels   = km.fit_predict(Xs)

    label_map = _name_clusters(km.cluster_centers_, feats, rules)
    style_labels = pd.Series([label_map[c] for c in labels],
                              index=df_pos.index)
    cluster_ids  = pd.Series(labels, index=df_pos.index)

    logger.info(
        "  %s: k=%d  dist=%s",
        pos, k,
        {label_map[i]: int((labels == i).sum()) for i in range(k)},
    )
    return cluster_ids, style_labels, km, scaler


# ---------------------------------------------------------------------------
# Global UMAP projection
# ---------------------------------------------------------------------------
def _compute_umap(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """UMAP 2-D projection across all players using percentile features."""
    try:
        import umap as umap_lib
    except ImportError:
        logger.warning("umap-learn not installed; falling back to PCA")
        from sklearn.decomposition import PCA
        X = df[UMAP_FEATURES].fillna(0.5).values
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
        return coords[:, 0], coords[:, 1]

    X = df[UMAP_FEATURES].fillna(0.5).values
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        low_memory=False,
    )
    coords = reducer.fit_transform(X)
    return coords[:, 0], coords[:, 1]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_clustering(csv_path: str, rebuild: bool = False) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not rebuild and os.path.exists(CLUSTER_PKL):
        logger.info("Style cluster model already exists (%s). Use --rebuild to retrain.", CLUSTER_PKL)
        # Still apply the saved model to the CSV so new players get labels
        _apply_saved_model(csv_path)
        return

    logger.info("Loading %s …", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    df = _build_features(df)

    # Ensure position_primary exists
    if "position_primary" not in df.columns:
        df["position_primary"] = (
            df["position"].astype(str).str.split(",").str[0].str.strip()
            .str.upper().map({"GK":"GK","DF":"DF","MF":"MF","FW":"FW"}).fillna("MF")
        )

    all_clusters = pd.Series(index=df.index, dtype=int)
    all_labels   = pd.Series(index=df.index, dtype=str)
    saved_models: dict = {}

    for pos, cfg in POSITION_CONFIG.items():
        mask = df["position_primary"] == pos
        if mask.sum() < cfg["k"] * 3:
            logger.warning("Too few %s players (%d) for k=%d; skipping", pos, mask.sum(), cfg["k"])
            all_clusters[mask] = 0
            all_labels[mask]   = "Unknown"
            continue

        logger.info("Clustering %s (%d players) …", pos, mask.sum())
        df_pos  = df[mask].copy()
        ids, lbls, km, scaler = _cluster_position(df_pos, pos, cfg)
        all_clusters[mask] = ids.values
        all_labels[mask]   = lbls.values
        saved_models[pos]  = {
            "kmeans":    km,
            "scaler":    scaler,
            "features":  cfg["features"],
            "label_map": {i: lbls[df_pos.index[ids.values == i][0]]
                          for i in range(cfg["k"])
                          if (ids.values == i).any()},
        }

    # ── UMAP 2-D ──────────────────────────────────────────────────────────
    logger.info("Computing UMAP projection …")
    ux, uy = _compute_umap(df)

    # ── Write back to CSV ──────────────────────────────────────────────────
    df_orig = pd.read_csv(csv_path, low_memory=False)   # clean copy
    df_orig["style_cluster"] = all_clusters.values.astype(int)
    df_orig["style_label"]   = all_labels.values.astype(str)
    df_orig["umap_x"]        = np.round(ux, 4)
    df_orig["umap_y"]        = np.round(uy, 4)
    df_orig.to_csv(csv_path, index=False)
    logger.info("✓ Wrote style_cluster / style_label / umap_x / umap_y to %s", csv_path)

    # ── Save model ────────────────────────────────────────────────────────
    with open(CLUSTER_PKL, "wb") as f:
        pickle.dump(saved_models, f)
    logger.info("✓ Saved cluster models → %s", CLUSTER_PKL)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("\nStyle label distribution:")
    for lbl, cnt in all_labels.value_counts().items():
        logger.info("  %-28s %d", lbl, cnt)


def _apply_saved_model(csv_path: str) -> None:
    """Apply an already-trained model to re-label the current CSV."""
    if not os.path.exists(CLUSTER_PKL):
        logger.error("No cluster model found at %s; run without --rebuild first", CLUSTER_PKL)
        return
    with open(CLUSTER_PKL, "rb") as f:
        models = pickle.load(f)

    df = pd.read_csv(csv_path, low_memory=False)
    df = _build_features(df)

    if "position_primary" not in df.columns:
        df["position_primary"] = (
            df["position"].astype(str).str.split(",").str[0].str.strip()
            .str.upper().map({"GK":"GK","DF":"DF","MF":"MF","FW":"FW"}).fillna("MF")
        )

    all_clusters = pd.Series(index=df.index, dtype=int)
    all_labels   = pd.Series(index=df.index, dtype=str)

    for pos, m in models.items():
        mask = df["position_primary"] == pos
        if not mask.any():
            continue
        feats = m["features"]
        X  = df.loc[mask, feats].fillna(0).values
        Xs = m["scaler"].transform(X)
        c  = m["kmeans"].predict(Xs)
        all_clusters[mask] = c
        all_labels[mask]   = [m["label_map"].get(ci, "Unknown") for ci in c]

    ux, uy = _compute_umap(df)

    df_orig = pd.read_csv(csv_path, low_memory=False)
    df_orig["style_cluster"] = all_clusters.values.astype(int)
    df_orig["style_label"]   = all_labels.values.astype(str)
    df_orig["umap_x"]        = np.round(ux, 4)
    df_orig["umap_y"]        = np.round(uy, 4)
    df_orig.to_csv(csv_path, index=False)
    logger.info("✓ Re-applied cluster model → %s", csv_path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Force retrain even if model already exists")
    args = ap.parse_args()

    csv_path = DATA_FILE_MULTI if os.path.exists(DATA_FILE_MULTI) else DATA_FILE
    run_clustering(csv_path, rebuild=args.rebuild)
