"""
ScoutAI — Transfer Window Simulator  (Feature 26)
===================================================
Given a team, budget, formation, and a list of needed positions, produces
a structured transfer plan:

  • Top candidate per position + 2 alternatives
  • Fee negotiation range  (opening bid → expected → walk-away max)
  • Urgency-based signing order
  • Sell recommendations to fund the window
  • LLM prompt ready for streaming narrative

Fee negotiation model
---------------------
Transfer confidence adjusts the fee baseline:
  Very High (≥65)  → club desperate; discount 20 %
  High     (45-64) → open to offers; discount 10 %
  Moderate (25-44) → needs persuading; market rate
  Low      (<25)   → settled player; 10 % premium

  opening_bid  = baseline × 0.80
  expected_fee = baseline × 0.95
  max_fee      = max(baseline × 1.12, future_value × 0.90)
"""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any
import logging

logger = logging.getLogger("scoutai.transfer_window")

# ---------------------------------------------------------------------------
# Position metadata
# ---------------------------------------------------------------------------

# Maps UI position slug → internal DB position code + display label
POSITION_META: dict[str, dict] = {
    "GK":  {"code": "GK", "label": "Goalkeeper",       "icon": "🧤"},
    "LB":  {"code": "DF", "label": "Left Back",         "icon": "⬅"},
    "CB":  {"code": "DF", "label": "Centre Back",       "icon": "🛡"},
    "RB":  {"code": "DF", "label": "Right Back",        "icon": "➡"},
    "DM":  {"code": "MF", "label": "Defensive Mid",     "icon": "⚙"},
    "CM":  {"code": "MF", "label": "Central Mid",       "icon": "⚡"},
    "CAM": {"code": "MF", "label": "Attacking Mid",     "icon": "🎯"},
    "LW":  {"code": "MF", "label": "Left Wing",         "icon": "💨"},
    "RW":  {"code": "MF", "label": "Right Wing",        "icon": "💨"},
    "SS":  {"code": "FW", "label": "Second Striker",    "icon": "🔥"},
    "ST":  {"code": "FW", "label": "Striker",           "icon": "⚽"},
    "CF":  {"code": "FW", "label": "Centre Forward",    "icon": "🎯"},
}

# ---------------------------------------------------------------------------
# Fee negotiation
# ---------------------------------------------------------------------------

def _fee_baseline(player: dict) -> float:
    mv = float(player.get("market_value", 10) or 10)
    tc = float(player.get("transfer_confidence", 40) or 40)
    fv = float(player.get("future_value", mv) or mv)

    if tc >= 65:
        baseline = mv * 0.80      # Very High — club will sell cheap
    elif tc >= 45:
        baseline = mv * 0.90      # High — open to offers
    elif tc >= 25:
        baseline = mv * 1.00      # Moderate — market rate
    else:
        baseline = mv * 1.10      # Low — premium needed

    return baseline


def compute_fees(player: dict) -> dict:
    baseline  = _fee_baseline(player)
    mv        = float(player.get("market_value", 10) or 10)
    fv        = float(player.get("future_value", mv) or mv)
    opening   = round(baseline * 0.80, 1)
    expected  = round(baseline * 0.95, 1)
    walk_away = round(max(baseline * 1.12, fv * 0.88), 1)
    return {
        "opening_bid":  opening,
        "expected_fee": expected,
        "max_fee":      walk_away,
    }


# ---------------------------------------------------------------------------
# Candidate finder
# ---------------------------------------------------------------------------

def find_candidates(
    players:        list[dict],
    pos_code:       str,
    exclude_team:   str,
    budget_cap:     float,
    n:              int = 3,
    mode:           str = "psi",
) -> list[dict]:
    """
    Return up to n candidates for a position, sorted by mode score,
    excluding the given team and players above budget_cap.
    """
    pool = [
        p for p in players
        if p.get("position") == pos_code
        and p.get("team", "") != exclude_team
        and float(p.get("market_value", 0) or 0) <= budget_cap
    ]

    def score(p: dict) -> float:
        psi = float(p.get("psi", 0) or 0)
        mv  = max(0.1, float(p.get("market_value", 0.1) or 0.1))
        fv  = float(p.get("future_value", mv) or mv)
        tc  = float(p.get("transfer_confidence", 40) or 40)
        if mode == "psi":
            return psi
        if mode == "value":
            return psi / mv
        if mode == "future":
            return max(0, (fv - mv) / mv) * (1 + psi)
        return psi

    pool.sort(key=score, reverse=True)
    return pool[:n]


# ---------------------------------------------------------------------------
# Sell recommendations
# ---------------------------------------------------------------------------

def find_sell_candidates(
    players:       list[dict],
    team:          str,
    needed_codes:  list[str],
    n:             int = 3,
) -> list[dict]:
    """
    From the team's current squad, find players who are good sell candidates:
    high transfer_confidence, position surplus to requirements,
    or declining arc.  Returns up to n recommendations.
    """
    squad = [p for p in players if p.get("team", "") == team]
    if not squad:
        return []

    # Score each player's "sellability"
    def sell_score(p: dict) -> float:
        tc      = float(p.get("transfer_confidence", 0) or 0)
        psi     = float(p.get("psi", 0) or 0)
        arc     = str(p.get("arc_phase", ""))
        mv      = float(p.get("market_value", 0) or 0)
        pos     = p.get("position", "")

        # Boost if position is being reinforced (so current player = surplus)
        pos_surplus = 1.3 if pos in needed_codes else 1.0
        # Boost declining arc players
        arc_boost   = 1.2 if arc == "post-peak" else 1.0
        # Prefer higher market value (more useful proceeds)
        value_bonus = min(2.0, mv / 20)

        return tc * pos_surplus * arc_boost * (0.7 + value_bonus * 0.3)

    squad.sort(key=sell_score, reverse=True)

    candidates = []
    for p in squad:
        tc    = float(p.get("transfer_confidence", 0) or 0)
        arc   = str(p.get("arc_phase", ""))
        pos   = p.get("position", "")
        mv    = float(p.get("market_value", 0) or 0)

        if tc < 25 and arc != "post-peak":
            continue   # settled, no reason to sell

        # Build reason string
        reasons = []
        if tc >= 65:
            reasons.append("very open to a move")
        elif tc >= 45:
            reasons.append("transfer talk likely")
        if arc == "post-peak":
            reasons.append("past their peak")
        if pos in needed_codes:
            reasons.append("surplus to requirements with incoming signing")
        if not reasons:
            continue

        sell_value = round(mv * 0.90, 1)   # expect ~90 % of market value
        candidates.append({
            **p,
            "sell_reason":     " · ".join(reasons),
            "estimated_sale":  sell_value,
        })
        if len(candidates) >= n:
            break

    return candidates


# ---------------------------------------------------------------------------
# Signing order
# ---------------------------------------------------------------------------

def determine_signing_order(
    targets: list[dict],
) -> list[dict]:
    """
    Sort targets by:
    1. Urgency = transfer_confidence of primary (high = approach NOW)
    2. Fee size (expensive deals first while budget is fresh)
    3. Position scarcity (fewer alternatives = handle first)
    Returns the same list re-ordered with a 'priority' field added.
    """
    def sort_key(t: dict) -> tuple:
        primary    = t.get("primary") or {}
        tc         = float(primary.get("transfer_confidence", 40) or 40)
        fee        = float(t.get("fees", {}).get("expected_fee", 0))
        n_alts     = len(t.get("alternatives", []))
        scarcity   = -n_alts          # fewer alts → more urgent
        return (-tc, -fee, scarcity)  # sort ascending so most urgent is first

    ordered = sorted(targets, key=sort_key)
    for i, t in enumerate(ordered):
        t["priority"] = i + 1
    return ordered


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def build_window_prompt(
    team:           str,
    budget:         float,
    formation:      str,
    targets:        list[dict],
    sells:          list[dict],
    extra_context:  str = "",
) -> str:
    fmt_map = {
        "433":"4-3-3","442":"4-4-2","4231":"4-2-3-1",
        "352":"3-5-2","532":"5-3-2","343":"3-4-3",
    }
    fmt_label = fmt_map.get(formation, formation)

    sell_total = sum(s.get("estimated_sale", 0) for s in sells)
    total_avail = budget + sell_total

    lines = [
        f"You are the sporting director at {team or 'the club'}.",
        f"Transfer window brief — {fmt_label} formation, €{budget}M direct budget" +
        (f" + €{sell_total:.0f}M expected from sales = €{total_avail:.0f}M total available." if sell_total else "."),
        "",
    ]

    if extra_context:
        lines += [f"Director's notes: {extra_context}", ""]

    lines.append("TARGETS IDENTIFIED:")
    for t in targets:
        p = t.get("primary") or {}
        fees = t.get("fees") or {}
        pos_label = t.get("pos_label", t.get("position_slug", ""))
        lines.append(
            f"\nPRIORITY {t['priority']}: {pos_label.upper()}"
        )
        lines.append(
            f"  Primary: {p.get('player','?')} · {p.get('age','?')}y · "
            f"{p.get('team','?')} · PSI {round((p.get('psi',0) or 0)*100)}% · "
            f"Value €{p.get('market_value',0):.0f}M · "
            f"Transfer: {p.get('transfer_label','?')}"
        )
        lines.append(
            f"  Offer: €{fees.get('opening_bid',0):.0f}M → "
            f"Expect: €{fees.get('expected_fee',0):.0f}M → "
            f"Max: €{fees.get('max_fee',0):.0f}M"
        )
        for j, alt in enumerate(t.get("alternatives", []), 1):
            lines.append(
                f"  Alt {j}: {alt.get('player','?')} ({alt.get('team','?')}) "
                f"PSI {round((alt.get('psi',0) or 0)*100)}% · €{alt.get('market_value',0):.0f}M"
            )

    if sells:
        lines.append("\nSELL CANDIDATES:")
        for s in sells:
            lines.append(
                f"  {s.get('player','?')} ({s.get('position','?')}) · "
                f"€{s.get('estimated_sale',0):.0f}M — {s.get('sell_reason','')}"
            )

    lines += [
        "",
        "Write a concise 4-paragraph transfer plan for the board:",
        "1. Signing sequence — who to approach first and exactly why that order.",
        "2. Negotiation strategy — opening bids, expected pushback, when to walk away.",
        "3. Contingency plan — if the primary target in each slot fails, what next.",
        "4. Window summary — total committed spend, net spend after sales, squad impact.",
        "",
        "Be specific. Name every player. Give actual €figures. Write like a football director. Under 350 words.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Template fallback (no LLM)
# ---------------------------------------------------------------------------

def generate_template_narrative(
    team:     str,
    budget:   float,
    targets:  list[dict],
    sells:    list[dict],
) -> str:
    sell_total = sum(s.get("estimated_sale", 0) for s in sells)
    committed  = sum(t.get("fees", {}).get("expected_fee", 0) for t in targets)
    net        = committed - sell_total
    remaining  = budget - net

    paras = []

    # Signing sequence
    seq_parts = []
    for t in targets:
        p = t.get("primary") or {}
        fees = t.get("fees") or {}
        seq_parts.append(
            f"Priority {t['priority']}: approach {p.get('player','?')} at "
            f"{p.get('team','?')} with an opening bid of "
            f"€{fees.get('opening_bid',0):.0f}M"
        )
    paras.append("SIGNING SEQUENCE — " + ". ".join(seq_parts) + ".")

    # Negotiation
    neg_parts = []
    for t in targets:
        p = t.get("primary") or {}
        fees = t.get("fees") or {}
        neg_parts.append(
            f"{p.get('player','?')}: open at €{fees.get('opening_bid',0):.0f}M, "
            f"expect to settle around €{fees.get('expected_fee',0):.0f}M, "
            f"walk away above €{fees.get('max_fee',0):.0f}M"
        )
    paras.append("NEGOTIATION — " + "; ".join(neg_parts) + ".")

    # Contingency
    cont_parts = []
    for t in targets:
        alts = t.get("alternatives", [])
        p    = t.get("primary") or {}
        if alts:
            alt_names = " or ".join(a.get("player", "?") for a in alts[:2])
            cont_parts.append(
                f"If {p.get('player','?')} falls through, pivot to {alt_names}"
            )
    if cont_parts:
        paras.append("CONTINGENCY — " + ". ".join(cont_parts) + ".")

    # Summary
    sell_text = (
        f"Sell proceeds of €{sell_total:.0f}M from recommended sales reduce net spend to €{net:.0f}M. "
        if sell_total > 0 else ""
    )
    paras.append(
        f"WINDOW SUMMARY — Total committed: €{committed:.0f}M. {sell_text}"
        f"Budget remaining after commitments: €{remaining:.0f}M."
    )

    return "\n\n".join(paras)


# ---------------------------------------------------------------------------
# Master planner
# ---------------------------------------------------------------------------

def plan_transfer_window(
    players:        list[dict],
    team:           str,
    budget:         float,
    formation:      str,
    needed:         list[dict],   # [{"slug": "ST", "mode": "psi"}, ...]
    extra_context:  str = "",
) -> dict:
    """
    Build the full structured plan (no LLM — that's streamed separately).

    Returns
    -------
    dict:
        targets       list of transfer target dicts
        sells         list of sell recommendation dicts
        prompt        str   — ready to feed to the LLM
        budget_breakdown dict
    """
    pos_codes_needed = list({POSITION_META[n["slug"]]["code"] for n in needed if n["slug"] in POSITION_META})

    # Budget ceiling per target = whole budget (they share; optimizer handles overlap)
    targets = []
    for item in needed:
        slug = item.get("slug", "ST")
        meta = POSITION_META.get(slug, {"code": "FW", "label": slug, "icon": "⚽"})
        mode = item.get("mode", "psi")

        candidates = find_candidates(
            players      = players,
            pos_code     = meta["code"],
            exclude_team = team,
            budget_cap   = budget,
            n            = 3,
            mode         = mode,
        )
        if not candidates:
            continue

        primary   = candidates[0]
        alts      = candidates[1:]
        fees      = compute_fees(primary)

        targets.append({
            "position_slug": slug,
            "pos_label":     meta["label"],
            "pos_icon":      meta["icon"],
            "pos_code":      meta["code"],
            "primary":       primary,
            "alternatives":  alts,
            "fees":          fees,
            "priority":      0,   # set by determine_signing_order
        })

    targets = determine_signing_order(targets)

    # Sell recommendations
    sells = find_sell_candidates(players, team, pos_codes_needed, n=3)

    # Budget breakdown
    committed   = sum(t["fees"]["expected_fee"] for t in targets)
    sell_total  = sum(s.get("estimated_sale", 0) for s in sells)
    net_spend   = committed - sell_total
    remaining   = budget - committed

    budget_breakdown = {
        "budget":       budget,
        "committed":    round(committed, 1),
        "sell_proceeds":round(sell_total, 1),
        "net_spend":    round(net_spend, 1),
        "remaining":    round(remaining, 1),
        "total_available": round(budget + sell_total, 1),
    }

    prompt = build_window_prompt(team, budget, formation, targets, sells, extra_context)

    return {
        "targets":          targets,
        "sells":            sells,
        "prompt":           prompt,
        "budget_breakdown": budget_breakdown,
        "template_narrative": generate_template_narrative(team, budget, targets, sells),
    }
