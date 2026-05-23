"""
ScoutAI - Transfer Confidence Score  (Feature 18)
===================================================
Computes a 0-100 transfer likelihood signal for each player based on
four independent signals:

  1. Contract urgency   (0-40 pts)  — shorter contract = higher urgency
  2. Age-stage window   (0-25 pts)  — inflection points drive movement
  3. Playing time gap   (0-20 pts)  — under-used players want out
  4. Market value gap   (0-15 pts)  — undervalued players attract interest

Score thresholds → label:
  0–24   → Low
  25–44  → Moderate
  45–64  → High
  65–100 → Very High

Usage:
    from transfer_confidence import compute_confidence
    result = compute_confidence(player_row_dict)
    # {score, label, color, reasons, factors}
"""

from __future__ import annotations
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_confidence(row: dict[str, Any]) -> dict[str, Any]:
    """
    Compute transfer confidence for a single player dict.

    Parameters
    ----------
    row : dict
        Player record as returned by player_to_dict / search_players.
        Expected keys: age, contract_years, minutes, market_value,
                       future_value, peak_age  (all optional with defaults).

    Returns
    -------
    dict with keys:
        score   (int, 0-100)
        label   (str)
        color   (str, CSS hex)
        reasons (list[str], 2-4 human-readable bullets)
        factors (dict, raw sub-scores for debug/tooltip)
    """
    age            = float(row.get("age",            25))
    contract_years = float(row.get("contract_years",  3))
    minutes        = float(row.get("minutes",       1800))
    market_value   = float(row.get("market_value",    10))
    future_value   = float(row.get("future_value",    10))
    peak_age       = float(row.get("peak_age",        27))

    value_gap = future_value - market_value   # positive = undervalued

    # ── 1. Contract urgency (0–40) ──────────────────────────────────────────
    if contract_years <= 0:
        contract_pts = 40
        contract_reason = "Free agent — available at no transfer fee"
    elif contract_years <= 1:
        contract_pts = 35
        contract_reason = f"Contract expires in {int(contract_years * 12 + 0.5)} months — renewal pressure imminent"
    elif contract_years < 2:
        contract_pts = 28
        contract_reason = "Under 2 years remaining — entering critical renewal window"
    elif contract_years <= 2:
        contract_pts = 20
        contract_reason = "2 years remaining — clubs will seek fee before value drops"
    elif contract_years <= 3:
        contract_pts = 10
        contract_reason = "3 years remaining — low but non-zero contract pressure"
    else:
        contract_pts = 0
        contract_reason = None

    # ── 2. Age-stage window (0–25) ─────────────────────────────────────────
    years_to_peak = peak_age - age

    if age <= 20:
        age_pts = 18
        age_reason = f"Young talent ({int(age)}y) — clubs often sell for profit before peak"
    elif age <= 23 and years_to_peak > 2:
        age_pts = 22
        age_reason = f"Pre-peak ({int(age)}y) — premium buying window, high market activity"
    elif 24 <= age <= 27 and years_to_peak >= 0:
        age_pts = 25
        age_reason = f"Peak age ({int(age)}y) — maximum transfer fee potential drives activity"
    elif 28 <= age <= 31:
        age_pts = 12
        age_reason = f"Post-peak ({int(age)}y) — clubs hold assets; loans more likely than sales"
    elif age >= 32:
        age_pts = 20
        age_reason = f"Veteran ({int(age)}y) — likely seeking final contract, free transfer possible"
    else:
        age_pts = 8
        age_reason = None

    # ── 3. Playing time gap (0–20) ─────────────────────────────────────────
    if minutes < 400:
        playing_pts = 20
        playing_reason = f"Barely playing ({int(minutes)} mins) — likely seeking a move for regular football"
    elif minutes < 900:
        playing_pts = 15
        playing_reason = f"Limited role ({int(minutes)} mins) — squad player likely open to new opportunity"
    elif minutes < 1500:
        playing_pts = 8
        playing_reason = f"Rotation player ({int(minutes)} mins) — some interest in a starting role elsewhere"
    else:
        playing_pts = 0
        playing_reason = None

    # ── 4. Value gap (0–15) ────────────────────────────────────────────────
    if value_gap > 20:
        value_pts = 15
        value_reason = f"Undervalued by ~{value_gap:.0f}M — high external interest expected"
    elif value_gap > 10:
        value_pts = 10
        value_reason = f"Undervalued by ~{value_gap:.0f}M — attracting scouting attention"
    elif value_gap > 4:
        value_pts = 5
        value_reason = f"Slight undervaluation (~{value_gap:.0f}M) — moderate transfer appeal"
    elif value_gap < -10:
        value_pts = 0
        value_reason = "Overvalued — clubs unlikely to meet asking price"
    else:
        value_pts = 2
        value_reason = None

    # ── Total ──────────────────────────────────────────────────────────────
    score = min(100, contract_pts + age_pts + playing_pts + value_pts)

    if score >= 65:
        label = "Very High"
        color = "#ef4444"   # red
    elif score >= 45:
        label = "High"
        color = "#f97316"   # orange
    elif score >= 25:
        label = "Moderate"
        color = "#eab308"   # yellow
    else:
        label = "Low"
        color = "#22c55e"   # green (low = stable, good for clubs hunting longevity)

    # Collect top reasons (skip None)
    candidate_reasons = [
        (contract_pts, contract_reason),
        (age_pts,      age_reason),
        (playing_pts,  playing_reason),
        (value_pts,    value_reason),
    ]
    reasons = [
        r for _, r in sorted(candidate_reasons, key=lambda x: -x[0])
        if r is not None
    ][:3]   # top 3 most impactful

    return {
        "score":   score,
        "label":   label,
        "color":   color,
        "reasons": reasons,
        "factors": {
            "contract": contract_pts,
            "age_stage": age_pts,
            "playing_time": playing_pts,
            "value_gap": value_pts,
        },
    }


# ---------------------------------------------------------------------------
# Batch helper (for enriching a full DataFrame column)
# ---------------------------------------------------------------------------

def enrich_dataframe(df):
    """
    Add transfer_confidence, transfer_label, transfer_color columns to df.
    Operates in-place and returns df for chaining.
    """
    scores  = []
    labels  = []
    colors  = []
    for _, row in df.iterrows():
        c = compute_confidence(row.to_dict())
        scores.append(c["score"])
        labels.append(c["label"])
        colors.append(c["color"])
    df["transfer_confidence"] = scores
    df["transfer_label"]      = labels
    df["transfer_color"]      = colors
    return df
