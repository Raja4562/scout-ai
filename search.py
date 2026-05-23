"""
ScoutAI - Search Engine
Hybrid semantic + PSI + value-gap scoring with natural language filter
extraction. Returns ranked player dicts ready to send to the frontend.
"""

import re
import logging
import numpy as np
import pandas as pd

from config import (
    TOP_K_DEFAULT, TOP_K_SIMILAR,
    SEMANTIC_WEIGHT, PSI_WEIGHT, VALUE_WEIGHT,
    POSITION_KEYWORDS, LEAGUE_KEYWORDS,
    NATIONALITY_KEYWORDS, STAT_BOOST_KEYWORDS,
)
from data_loader import load_players, player_to_dict
from embeddings import build_or_load_embeddings, encode_query

logger = logging.getLogger("scoutai.search")

_df:         pd.DataFrame | None = None
_embeddings: np.ndarray   | None = None


def _ensure_loaded():
    global _df, _embeddings
    if _df is None:
        _df = load_players()
    if _embeddings is None:
        _embeddings = build_or_load_embeddings(_df)


# ---------------------------------------------------------------------------
# Natural language filter extraction
# ---------------------------------------------------------------------------

def parse_filters(query: str) -> dict:
    """
    Extract structured filters from free-text natural language queries.

    Recognises:
      position    - "left back", "no.10", "striker", "GK", …
      max_age     - "under 25", "younger than 23", "wonderkid"
      min_age     - "veteran", "experienced", "over 28"
      age_range   - "between 22 and 26", "aged 22–26"
      max_price   - "under 30M", "within 50M", "budget 20M"
      min_minutes - "regular starter", "first choice"
      nationality - "Brazilian", "French", "Dutch", …
      league      - "Premier League", "Bundesliga", …
      contract    - "free agent", "expiring contract", "out of contract"
      arc_phase   - "pre-peak", "emerging", "peak", "veteran", "past prime"
      style       - passed directly (from UI style-filter clicks)
      stat_boosts - {"col": weight, …} dict for soft stat scoring nudges
    """
    q = query.lower()
    # Pad with spaces so boundary checks on keyword phrases work uniformly
    qs = f" {q} "
    filters: dict = {}

    # ── Position ──────────────────────────────────────────────────────────────
    # Scan longest keywords first (POSITION_KEYWORDS is ordered by insertion, but
    # we sort by length descending here to avoid "back" matching "left back")
    for kw, pos in sorted(POSITION_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in qs:
            filters["position"] = pos
            break

    # Explicit bare position codes (GK/DF/MF/FW as standalone word)
    if "position" not in filters:
        for code in ("gk", "df", "mf", "fw"):
            if re.search(rf"\b{code}\b", q):
                filters["position"] = code.upper()
                break

    # ── Age ───────────────────────────────────────────────────────────────────
    # Age range: "between 22 and 26" / "aged 22 to 26"
    # (avoid bare "22-26" pattern — too easy to conflict with prices/stats)
    m_range = re.search(
        r"(?:between|aged?)\s+(\d+)\s+(?:and|to|-|–)\s+(\d+)", q
    )
    if m_range:
        lo, hi = int(m_range.group(1)), int(m_range.group(2))
        if 15 <= lo < hi <= 45:       # sanity-check: plausible football ages
            filters["min_age"] = lo
            filters["max_age"] = hi

    # Max age (only if not already set by range)
    # \b prevents backtracking into partial numbers (e.g. "25M" → 2)
    # (?!\s?[mM](?:illion)?\b) blocks "25M" / "25 M" / "25 million" but NOT "28 midfielder"
    # Range check 15–45 guards against any residual false positives
    if "max_age" not in filters:
        m = re.search(
            r"(?:under|younger than|below|aged?)\s+(\d+)\b(?!\s?[mM](?:illion)?\b)", q
        )
        if m:
            val = int(m.group(1))
            if 15 <= val <= 45:
                filters["max_age"] = val
        elif re.search(r"\byoung\b|\byouth\b|\bprospect\b|\bwonderki?d\b|\bemerging\b", q):
            filters["max_age"] = 23

    # Min age (only if not already set by range)
    # Explicit "over N" beats keyword heuristics like "veteran"
    if "min_age" not in filters:
        m2 = re.search(r"(?:over|older than|at least)\s+(\d+)(?!\s*[mM])", q)
        if m2:
            filters["min_age"] = int(m2.group(1))
        elif re.search(r"\bveteran\b|\bexperienced\b|\bsenior\b", q):
            filters["min_age"] = 29

    # ── Budget / max price ────────────────────────────────────────────────────
    m3 = re.search(
        r"(?:under|below|within|max(?:imum)?|budget|less than|no more than|up to|cost(?:s|ing)?)"
        r"\s*[\$€£]?\s*(\d+(?:\.\d+)?)\s*[mM](?:illion)?",
        q,
    )
    if m3:
        filters["max_price"] = float(m3.group(1))

    # ── Min minutes / playing time ────────────────────────────────────────────
    if re.search(r"\bregular starter\b|\bfirst.?choice\b|\bfirst.?team\b|\bstarting\b", q):
        filters["min_minutes"] = 1800    # ~20 full games
    m_mins = re.search(r"(\d+)\s*(?:\+)?\s*minutes?", q)
    if m_mins:
        filters["min_minutes"] = int(m_mins.group(1))

    # ── Nationality ───────────────────────────────────────────────────────────
    for kw, iso in sorted(NATIONALITY_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in qs:
            # Don't let "english" match if it was already consumed as a league
            if not (kw == "english" and filters.get("league") == "EPL"):
                filters["nationality"] = iso
                break

    # ── League ────────────────────────────────────────────────────────────────
    for kw, league in sorted(LEAGUE_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in qs and "league" not in filters:
            filters["league"] = league
            break

    # ── Contract situation ────────────────────────────────────────────────────
    if re.search(
        r"\bfree agent\b|\bout of contract\b|\bno contract\b|\bcontractless\b", q
    ):
        filters["free_agent"] = True
        filters["max_contract_years"] = 0
    elif re.search(
        r"\bexpir(?:ing|ed)\b|\bpre.?contract\b|\bending contract\b"
        r"|\bfinal year\b|\blast year of contract\b", q
    ):
        filters["max_contract_years"] = 1

    # ── Career arc phase ──────────────────────────────────────────────────────
    if re.search(r"\bpre.?peak\b|\bemerging\b|\bupside\b|\bdevelop\b|\bpotential\b", q):
        filters["arc_phase"] = "pre-peak"
    elif re.search(r"\bpost.?peak\b|\bpast (?:his |their )?prime\b|\bdeclining\b", q):
        filters["arc_phase"] = "post-peak"
    elif re.search(r"\bin (?:their |his )?prime\b|\bpeak years\b", q):
        filters["arc_phase"] = "peak"

    # ── Stat boosts (soft scoring signals) ────────────────────────────────────
    # Scan longest phrases first (STAT_BOOST_KEYWORDS is already ordered that way)
    accumulated: dict[str, float] = {}
    for phrase, boosts in STAT_BOOST_KEYWORDS:
        if phrase in q:
            for col, w in boosts.items():
                accumulated[col] = accumulated.get(col, 0.0) + w
    if accumulated:
        filters["stat_boosts"] = accumulated

    return filters


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search_players(
    query:     str,
    filters:   dict | None = None,
    top_k:     int = TOP_K_DEFAULT,
) -> list[dict]:
    """
    Hybrid search:
      score = SEMANTIC_WEIGHT * cosine_sim
            + PSI_WEIGHT      * psi_norm
            + VALUE_WEIGHT    * value_gap_norm

    Returns a list of player dicts sorted by score descending.
    """
    _ensure_loaded()
    if filters is None:
        filters = {}

    # Merge natural-language filters (only add if not already set by caller)
    parsed = parse_filters(query)
    for k, v in parsed.items():
        if k not in filters:
            filters[k] = v

    # Encode query and compute cosine similarities
    q_vec = encode_query(query)
    sims  = (_embeddings @ q_vec).astype(np.float64)   # shape (N,)

    # PSI normalisation (0-1)
    psi_vals = _df["psi"].values.astype(np.float64)
    psi_max  = psi_vals.max() or 1.0
    psi_norm = psi_vals / psi_max

    # Value gap normalisation (future_value - market_value, clipped & scaled)
    mv   = _df["market_value"].values.astype(np.float64)
    fv   = _df["future_value"].values.astype(np.float64)
    gaps = np.clip(fv - mv, -50, 100)
    gap_range = gaps.max() - gaps.min() or 1.0
    gap_norm  = (gaps - gaps.min()) / gap_range

    # ── Stat boosts (Feature 14) ────────────────────────────────────────────
    stat_boosts  = filters.get("stat_boosts", {})
    extra_value  = stat_boosts.get("__value_boost", 0.0)
    extra_psi    = stat_boosts.get("__psi_boost",   0.0)

    boost_score = np.zeros(len(_df), dtype=np.float64)
    for col, w in stat_boosts.items():
        if col.startswith("__"):
            continue
        if col in _df.columns:
            vals = _df[col].fillna(0.0).values.astype(np.float64)
            boost_score += w * vals

    # Hybrid score with dynamic weight adjustments
    hybrid = (
        SEMANTIC_WEIGHT              * sims
        + (PSI_WEIGHT   + extra_psi) * psi_norm
        + (VALUE_WEIGHT + extra_value) * gap_norm
        + boost_score
    )

    # Candidate pool: 10× top_k so style/league/age filters don't starve results
    pool_size = min(max(top_k * 10, 200), len(hybrid))
    top_idx   = np.argpartition(hybrid, -pool_size)[-pool_size:]
    top_idx   = top_idx[np.argsort(hybrid[top_idx])[::-1]]

    results = []
    for idx in top_idx:
        row = _df.iloc[idx]

        # Apply filters
        pos = str(row.get("position_primary", ""))
        if "position" in filters and pos != filters["position"]:
            continue
        age = float(row.get("age", 0))
        if "max_age" in filters and age > filters["max_age"]:
            continue
        if "min_age" in filters and age < filters["min_age"]:
            continue
        mv_val = float(row.get("market_value", 0))
        if "max_price" in filters and mv_val > filters["max_price"]:
            continue
        row_league = str(row.get("league", "EPL"))
        if "league" in filters and row_league != filters["league"]:
            continue
        # Legacy expiring flag
        if filters.get("expiring") and float(row.get("contract_years", 5)) > 1:
            continue
        # Contract years (Feature 14)
        c_years = float(row.get("contract_years", 5))
        if "max_contract_years" in filters and c_years > filters["max_contract_years"]:
            continue
        if filters.get("free_agent") and c_years > 0:
            continue
        # Min minutes played (Feature 14)
        if "min_minutes" in filters and float(row.get("minutes", 0)) < filters["min_minutes"]:
            continue
        # Nationality (Feature 14)
        if "nationality" in filters:
            row_nat = str(row.get("nationality", ""))
            if row_nat != filters["nationality"]:
                continue
        # Career arc phase (Feature 14)
        if "arc_phase" in filters and str(row.get("arc_phase", "")) != filters["arc_phase"]:
            continue
        # Style cluster filter (Feature 11)
        if "style" in filters and str(row.get("style_label", "")) != filters["style"]:
            continue

        results.append(player_to_dict(row, score=float(hybrid[idx])))
        if len(results) >= top_k:
            break

    logger.info(
        "Search '%s' -> %d results (filters: %s, pool: %d players)",
        query[:50], len(results), filters, len(_df),
    )
    return results


def find_similar(player_name: str, top_n: int = TOP_K_SIMILAR) -> list[dict]:
    """
    Find the most similar players to a given player by embedding similarity.
    Restricts to same position group. Excludes the player themselves.
    """
    _ensure_loaded()

    matches = _df[_df["player"] == player_name]
    if matches.empty:
        return []

    row  = matches.sort_values("minutes", ascending=False).iloc[0]
    idx  = int(row["_idx"])
    pos  = str(row.get("position_primary", "MF"))

    anchor_vec = _embeddings[idx]

    # Restrict to same position
    pos_mask = (_df["position_primary"] == pos).values
    pos_idx  = np.where(pos_mask)[0]

    sims    = (_embeddings[pos_idx] @ anchor_vec)
    order   = np.argsort(sims)[::-1]
    ordered = pos_idx[order]

    results = []
    for i in ordered:
        if i == idx:
            continue
        r = _df.iloc[i]
        results.append(player_to_dict(r, score=float(sims[order[len(results)]])))
        if len(results) >= top_n:
            break

    return results


# ---------------------------------------------------------------------------
# Summary generator (template-based for Phase 1, replaced by LLM in Phase 2)
# ---------------------------------------------------------------------------

def generate_summary(query: str, results: list[dict], filters: dict, context: str = "") -> str:
    if not results:
        return (
            "No players matched your search. "
            "Try broadening the position, age, or budget filters."
        )

    top  = results[0]
    n    = len(results)
    pos  = filters.get("position", top["position"])
    pos_label = {"FW": "forwards", "MF": "midfielders",
                 "DF": "defenders", "GK": "goalkeepers"}.get(pos, "players")

    gap = top["value_gap"]
    if gap > 5:
        val_str = f"undervalued by {gap:.0f}M euros"
    elif gap < -5:
        val_str = f"overvalued by {abs(gap):.0f}M euros"
    else:
        val_str = "fairly priced by the model"

    lines = [
        f"Found {n} {pos_label} matching your search.",
        f"Best match: {top['player']} ({top['team']}, {top['age']}y) - "
        f"{top['market_value']:.0f}M market value, {val_str}.",
    ]

    if top["psi"] > 0.65:
        lines.append(
            f"PSI of {top['psi']:.2f} places {top['player']} "
            f"in the top percentile for their position."
        )

    budget_hits = [r for r in results if "max_price" in filters
                   and r["market_value"] <= filters["max_price"]]
    if budget_hits:
        lines.append(
            f"{len(budget_hits)} players found within "
            f"your {filters['max_price']:.0f}M budget."
        )

    future_gems = [r for r in results if r["value_gap"] > 15 and r["age"] <= 24]
    if future_gems:
        gem = future_gems[0]
        lines.append(
            f"Hidden gem: {gem['player']} ({gem['team']}, {gem['age']}y) "
            f"has a projected future value {gem['value_gap']:.0f}M above current market."
        )

    if context:
        lines.append(f"Context: {context}.")

    return "  ".join(lines)
