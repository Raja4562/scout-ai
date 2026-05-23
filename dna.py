"""
ScoutAI — Player DNA Matching  (Feature 28)
=============================================
Embeds a team's squad as a single "DNA vector" by taking a PSI-weighted
average of every player's sentence-transformer embedding, then L2-normalising.

Because the embeddings encode style archetypes (e.g. "Ball Winner",
"Sweeper Keeper"), percentile-based performance descriptors, arc phase, and
contract context, the DNA vector captures tactical identity — not just
position counts.  Cosine similarity against external players' vectors then
finds the best cultural/stylistic fits.

Match score interpretation (cosine sim × 100)
----------------------------------------------
  ≥ 82  — Excellent DNA fit: very similar style profile
  72–81 — Strong fit: broadly aligned playing identity
  62–71 — Decent fit: some stylistic overlap
  < 62  — Modest / tactical stretch

These thresholds are empirical for all-MiniLM-L6-v2 on football profiles.
"""

from __future__ import annotations
from collections import Counter
import logging
import numpy as np

logger = logging.getLogger("scoutai.dna")


# ---------------------------------------------------------------------------
# Build team DNA vector
# ---------------------------------------------------------------------------

def build_team_dna(
    df,
    matrix:        np.ndarray,
    team:          str,
    squad_filter:  str  = "all",    # "all" | "starters" (top 11 by PSI)
    psi_weighted:  bool = True,     # weight each vec by PSI; False = uniform
) -> dict | None:
    """
    Compute the team's DNA embedding vector and a human-readable style profile.

    Returns
    -------
    dict with keys:
        dna_vec   np.ndarray (384,) — L2-normalised
        profile   dict              — identity summary
        squad     list[dict]        — players used to build the DNA
    or None if the team has no players in the dataset.
    """
    from data_loader import player_to_dict

    team_rows = df[df["team"] == team].copy()
    if team_rows.empty:
        return None

    if squad_filter == "starters":
        team_rows = team_rows.nlargest(11, "psi")

    vecs, weights, squad_players = [], [], []

    for _, row in team_rows.iterrows():
        idx = int(row["_idx"])
        if idx >= len(matrix):
            continue
        psi    = float(row.get("psi", 0.5) or 0.5)
        weight = psi if psi_weighted else 1.0
        vecs.append(matrix[idx])
        weights.append(weight)
        squad_players.append(player_to_dict(row))

    if not vecs:
        return None

    # PSI-weighted centroid
    w = np.array(weights, dtype=np.float32)
    w /= w.sum()
    dna_vec = np.average(np.stack(vecs), axis=0, weights=w).astype(np.float32)

    # L2-normalise so downstream dot product == cosine similarity
    norm = float(np.linalg.norm(dna_vec))
    if norm > 0:
        dna_vec /= norm

    # Style profile ─ dominant archetypes, positional balance, aggregate stats
    style_counter = Counter(
        p["style_label"] for p in squad_players if p.get("style_label")
    )
    pos_counter = Counter(p.get("position", "MF") for p in squad_players)

    ages  = [p["age"]          for p in squad_players if p.get("age")]
    psis  = [p["psi"]          for p in squad_players if p.get("psi")]
    mvs   = [p["market_value"] for p in squad_players if p.get("market_value")]

    profile = {
        "team":            team,
        "player_count":    len(squad_players),
        "squad_filter":    squad_filter,
        "top_styles":      style_counter.most_common(5),     # [(label, count), …]
        "position_counts": dict(pos_counter),                # {GK:1, DF:4, …}
        "avg_age":         round(float(np.mean(ages)), 1)  if ages  else None,
        "avg_psi":         round(float(np.mean(psis)), 3)  if psis  else None,
        "avg_mv":          round(float(np.mean(mvs)),  1)  if mvs   else None,
        "total_mv":        round(float(np.sum(mvs)),   0)  if mvs   else None,
        "arc_mix":         _arc_mix(squad_players),
    }

    logger.info(
        "DNA built for %s: %d players, top styles: %s",
        team, len(squad_players),
        [s for s, _ in style_counter.most_common(3)],
    )
    return {"dna_vec": dna_vec, "profile": profile, "squad": squad_players}


def _arc_mix(players: list[dict]) -> dict:
    """Return proportion of squad in each arc phase."""
    total = len(players) or 1
    cnt   = Counter(p.get("arc_phase", "peak") for p in players)
    return {
        phase: round(cnt.get(phase, 0) / total * 100)
        for phase in ("pre-peak", "peak", "post-peak")
    }


# ---------------------------------------------------------------------------
# Find DNA matches
# ---------------------------------------------------------------------------

def find_dna_matches(
    df,
    matrix:       np.ndarray,
    dna_vec:      np.ndarray,
    exclude_team: str,
    position:     str   = "all",   # "all" | "GK" | "DF" | "MF" | "FW"
    league:       str   = "all",
    max_age:      int   = 35,
    max_price:    float = 9999,
    min_minutes:  int   = 500,
    arc_phase:    str   = "all",   # "all" | "pre-peak" | "peak" | "post-peak"
    limit:        int   = 40,
) -> list[dict]:
    """
    Rank all external players by cosine similarity to the team DNA vector.

    Each result includes:
        dna_score  float  raw cosine sim (0-1)
        dna_pct    float  × 100, displayed as "XX.X%"
        dna_tier   str    "excellent" | "strong" | "decent" | "modest"
    """
    from data_loader import player_to_dict

    sims  = (matrix @ dna_vec).astype(np.float64)
    order = np.argsort(sims)[::-1]

    results = []
    for i in order:
        row = df.iloc[i]

        if str(row.get("team", "")) == exclude_team:
            continue
        pos = str(row.get("position_primary", "MF"))
        if position != "all" and pos != position.upper():
            continue
        if league != "all" and str(row.get("league", "")) != league:
            continue
        if int(row.get("age", 99)) > max_age:
            continue
        mv = float(row.get("market_value", 0) or 0)
        if mv > max_price:
            continue
        if float(row.get("minutes", 0)) < min_minutes:
            continue
        if arc_phase != "all" and str(row.get("arc_phase", "")) != arc_phase:
            continue

        score = float(sims[i])
        p = player_to_dict(row)
        p["dna_score"] = round(score, 4)
        p["dna_pct"]   = round(score * 100, 1)
        p["dna_tier"]  = (
            "excellent" if score >= 0.82 else
            "strong"    if score >= 0.72 else
            "decent"    if score >= 0.62 else
            "modest"
        )
        results.append(p)
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Per-position DNA breakdown
# ---------------------------------------------------------------------------

def positional_dna_breakdown(
    df,
    matrix:   np.ndarray,
    team:     str,
    psi_weighted: bool = True,
) -> dict:
    """
    Compute a separate DNA vector per position group so callers can ask:
    "Which midfielders fit our midfield identity specifically?"

    Returns { "GK": dna_result_or_None, "DF": …, "MF": …, "FW": … }
    """
    breakdown = {}
    for pos in ("GK", "DF", "MF", "FW"):
        sub_df = df[(df["team"] == team) & (df["position_primary"] == pos)].copy()
        if sub_df.empty:
            breakdown[pos] = None
            continue
        sub_df["_idx"] = sub_df.index   # ensure _idx is current
        result = build_team_dna(sub_df, matrix, team, psi_weighted=psi_weighted)
        breakdown[pos] = result
    return breakdown
