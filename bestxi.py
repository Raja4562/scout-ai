"""
ScoutAI — Best XI Builder  (Feature 25)
========================================
Constrained squad optimisation: given a budget, formation, and optional
filters, select the optimal 11-player squad from the full player database.

Optimisation approach
---------------------
Positions are fixed by formation (e.g. 4-3-3 → 1 GK, 4 DF, 3 MF, 3 FW).
Within each position group the problem reduces to selecting the best-scoring
subset that fits inside the remaining budget.  We use a multi-lambda greedy
sweep (8 penalty values) that trades off raw score against cost, then keep
the feasible solution with the highest total score.  This finds the global
optimum in the vast majority of cases for datasets of this size.

Scoring modes
-------------
psi       raw PSI (match-fitness index, 0-1)
value     PSI / market_value          — quality per €M
future    upside × quality            — (fv-mv)/mv × (1+psi)
young     PSI × youth bonus           — peaks at age 18, zero at 25+
transfer  PSI × transfer_confidence   — available players first
balanced  0.55 × PSI + 0.45 × value  — compromise
"""

from __future__ import annotations
from collections import Counter
from typing import Any
import logging
import random

logger = logging.getLogger("scoutai.bestxi")

# ---------------------------------------------------------------------------
# Formation definitions
# ---------------------------------------------------------------------------

FORMATION_POSITIONS: dict[str, dict[str, int]] = {
    "433":  {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "442":  {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4231": {"GK": 1, "DF": 4, "MF": 5, "FW": 1},
    "352":  {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
    "532":  {"GK": 1, "DF": 5, "MF": 3, "FW": 2},
    "343":  {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
}

MODE_LABELS: dict[str, str] = {
    "psi":      "Peak Performance",
    "value":    "Best Value",
    "future":   "Future Stars",
    "young":    "Young Guns",
    "transfer": "Transfer Ready",
    "balanced": "Balanced",
}

MODE_DESCRIPTIONS: dict[str, str] = {
    "psi":      "Maximise total PSI — the highest-performing XI money can buy",
    "value":    "Maximise PSI per €M spent — overachievers and hidden gems",
    "future":   "Maximise model upside — players whose value will grow most",
    "young":    "U24 focus — build for the future, peak in 2-3 seasons",
    "transfer": "Prioritise players with high transfer availability signal",
    "balanced": "Blend of peak performance and budget efficiency",
}

# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def _score(player: dict, mode: str) -> float:
    psi = float(player.get("psi", 0) or 0)
    mv  = max(0.1, float(player.get("market_value",  0.1) or 0.1))
    fv  = float(player.get("future_value", mv) or mv)
    age = int(player.get("age", 25) or 25)
    tc  = float(player.get("transfer_confidence", 50) or 50)

    if mode == "psi":
        return psi

    if mode == "value":
        return psi / mv

    if mode == "future":
        upside = max(0.0, (fv - mv) / mv)
        return upside * (1.0 + psi)

    if mode == "young":
        bonus = max(0.0, (25 - age) / 7.0) if age < 25 else 0.0
        return psi * (1.0 + bonus)

    if mode == "transfer":
        return psi * (tc / 100.0)

    if mode == "balanced":
        return 0.55 * psi + 0.45 * (psi / mv)

    return psi


def _player_key(p: dict) -> str:
    return f"{p.get('player', '')}|{p.get('team', '')}"


# ---------------------------------------------------------------------------
# Core greedy pass
# ---------------------------------------------------------------------------

def _greedy_pass(
    by_pos:       dict[str, list[dict]],
    pos_counts:   dict[str, int],
    budget:       float,
    mode:         str,
    lam:          float,
    max_per_team: int,
    pre_selected: list[dict],
    variation:    int = 0,
) -> list[dict] | None:
    """
    Single greedy pass with budget-penalty lambda.
    Returns 11-player list or None if infeasible at this lambda.
    variation: 0 = deterministic; 1-4 = shuffle within score buckets
    """
    xi        = list(pre_selected)
    remaining = budget - sum(float(p.get("market_value", 0) or 0) for p in pre_selected)
    team_cnt  = Counter(p.get("team", "") for p in pre_selected)
    used      = {_player_key(p) for p in pre_selected}

    # Positions still to fill
    filled = Counter(p.get("position", "") for p in pre_selected)
    needs  = {pos: max(0, cnt - filled.get(pos, 0))
               for pos, cnt in pos_counts.items()}

    if remaining < -0.01:
        return None  # locked players exceed budget

    # Fill order: FW (most expensive) → MF → DF → GK
    for pos in ["FW", "MF", "DF", "GK"]:
        need = needs.get(pos, 0)
        if need <= 0:
            continue

        pool = by_pos.get(pos, [])
        # Score with lambda cost-penalty
        scored: list[tuple[float, dict]] = []
        for p in pool:
            if _player_key(p) in used:
                continue
            if team_cnt.get(p.get("team", ""), 0) >= max_per_team:
                continue
            mv  = float(p.get("market_value", 0) or 0)
            adj = _score(p, mode) - lam * (mv / max(budget, 1.0))
            scored.append((adj, p))

        scored.sort(key=lambda x: -x[0])

        # Apply variation: inject mild randomness in order within ±5% score buckets
        if variation > 0 and len(scored) > need:
            rng = random.Random(variation * 31337 + pos.__hash__())
            top_score = scored[0][0] if scored else 0
            tier = [x for x in scored if top_score - x[0] < top_score * 0.05 + 0.05]
            rest = scored[len(tier):]
            rng.shuffle(tier)
            scored = tier + rest

        picked = 0
        for _, p in scored:
            if picked >= need:
                break
            mv = float(p.get("market_value", 0) or 0)
            if mv <= remaining + 0.001:
                xi.append(p)
                used.add(_player_key(p))
                team_cnt[p.get("team", "")] = team_cnt.get(p.get("team", ""), 0) + 1
                remaining -= mv
                picked += 1

        if picked < need:
            return None  # infeasible at this lambda / variation

    return xi if len(xi) == 11 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_best_xi(
    players:           list[dict],
    formation:         str        = "433",
    budget:            float      = 200.0,
    mode:              str        = "psi",
    league:            str        = "all",
    max_age:           int | None = None,
    min_age:           int | None = None,
    exclude_team:      str | None = None,
    locked_keys:       set[str]   = None,
    max_per_team:      int        = 3,
    nationality_limit: int | None = None,
    variation:         int        = 0,
) -> dict:
    """
    Build the optimal XI subject to constraints.

    Returns
    -------
    dict with keys:
        xi            list[dict]   11 players (or best partial if infeasible)
        feasible      bool
        budget_used   float
        budget_left   float
        total_psi     float        sum of PSI values (0-1 each)
        avg_psi       float
        avg_age       float
        mode_label    str
        mode_desc     str
        formation     str
        missing_pos   list[str]    positions that couldn't be filled
        min_budget    float        cheapest possible XI ignoring score
    """
    locked_keys  = locked_keys or set()
    pos_counts   = FORMATION_POSITIONS.get(formation, FORMATION_POSITIONS["433"])

    # ── 1. Filter candidate pool ─────────────────────────────────────────────
    pool: list[dict] = []
    for p in players:
        if league != "all" and p.get("league") != league:
            continue
        age = int(p.get("age", 25) or 25)
        if max_age is not None and age > max_age:
            continue
        if min_age is not None and age < min_age:
            continue
        if exclude_team and p.get("team") == exclude_team:
            continue
        pool.append(p)

    # ── 2. Separate locked vs free players ───────────────────────────────────
    locked   = [p for p in pool if _player_key(p) in locked_keys]
    unlocked = [p for p in pool if _player_key(p) not in locked_keys]

    # Drop excess locked players for any position (keep highest PSI)
    locked_cnt = Counter(p.get("position", "") for p in locked)
    for pos, need in pos_counts.items():
        if locked_cnt.get(pos, 0) > need:
            same_pos = sorted(
                [p for p in locked if p.get("position") == pos],
                key=lambda p: p.get("psi", 0),
            )
            n_drop = locked_cnt[pos] - need
            for p in same_pos[:n_drop]:
                locked.remove(p)
                unlocked.append(p)

    # ── 3. Group unlocked by position, sort by score ──────────────────────────
    by_pos: dict[str, list[dict]] = {pos: [] for pos in pos_counts}
    for p in unlocked:
        pos = p.get("position", "MF")
        if pos in by_pos:
            by_pos[pos].append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: _score(p, mode), reverse=True)

    # ── 4. Multi-lambda sweep ─────────────────────────────────────────────────
    lambdas = [0.0, 0.15, 0.35, 0.6, 0.9, 1.4, 2.2, 4.0]
    best_xi:    list[dict] | None = None
    best_score: float             = -1.0

    for lam in lambdas:
        xi = _greedy_pass(
            by_pos, pos_counts, budget, mode, lam,
            max_per_team, locked, variation,
        )
        if xi is None:
            continue
        total = sum(_score(p, mode) for p in xi)
        if total > best_score:
            best_score = total
            best_xi    = xi

    # ── 5. Minimum feasible budget (cheapest players per position) ───────────
    min_budget = _estimate_min_budget(pool, pos_counts)

    # ── 6. Handle infeasible ──────────────────────────────────────────────────
    if best_xi is None:
        partial = _greedy_pass(
            by_pos, pos_counts, 1e9, mode, 0.0, max_per_team, locked, 0,
        ) or []
        missing = _missing_positions(partial, pos_counts)
        total_needed = sum(float(p.get("market_value", 0) or 0) for p in partial)
        return {
            "xi":          partial,
            "feasible":    False,
            "budget_used": round(total_needed, 1),
            "budget_left": round(budget - total_needed, 1),
            "total_psi":   round(sum(p.get("psi", 0) for p in partial), 3),
            "avg_psi":     round(_avg(partial, "psi"), 3),
            "avg_age":     round(_avg(partial, "age"), 1),
            "mode_label":  MODE_LABELS.get(mode, mode),
            "mode_desc":   MODE_DESCRIPTIONS.get(mode, ""),
            "formation":   formation,
            "missing_pos": missing,
            "min_budget":  min_budget,
        }

    # ── 7. Nationality cap (best-effort swap) ─────────────────────────────────
    if nationality_limit:
        best_xi = _enforce_nationality(
            best_xi, by_pos, pos_counts,
            nationality_limit, budget, mode,
        )

    budget_used = sum(float(p.get("market_value", 0) or 0) for p in best_xi)

    return {
        "xi":          best_xi,
        "feasible":    True,
        "budget_used": round(budget_used, 1),
        "budget_left": round(budget - budget_used, 1),
        "total_psi":   round(sum(p.get("psi", 0) for p in best_xi), 3),
        "avg_psi":     round(_avg(best_xi, "psi"), 3),
        "avg_age":     round(_avg(best_xi, "age"), 1),
        "mode_label":  MODE_LABELS.get(mode, mode),
        "mode_desc":   MODE_DESCRIPTIONS.get(mode, ""),
        "formation":   formation,
        "missing_pos": [],
        "min_budget":  min_budget,
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _avg(players: list[dict], key: str) -> float:
    vals = [float(p.get(key, 0) or 0) for p in players]
    return sum(vals) / len(vals) if vals else 0.0


def _missing_positions(xi: list[dict], pos_counts: dict[str, int]) -> list[str]:
    filled  = Counter(p.get("position", "") for p in xi)
    missing = []
    for pos, need in pos_counts.items():
        for _ in range(max(0, need - filled.get(pos, 0))):
            missing.append(pos)
    return missing


def _estimate_min_budget(pool: list[dict], pos_counts: dict[str, int]) -> float:
    """Cheapest valid XI ignoring score / team diversity."""
    by_pos: dict[str, list[float]] = {}
    for p in pool:
        pos = p.get("position", "MF")
        mv  = float(p.get("market_value", 0) or 0)
        by_pos.setdefault(pos, []).append(mv)
    total = 0.0
    for pos, count in pos_counts.items():
        costs = sorted(by_pos.get(pos, [0]))
        total += sum(costs[:count])
    return round(total, 1)


def _enforce_nationality(
    xi:       list[dict],
    by_pos:   dict[str, list[dict]],
    pos_counts: dict[str, int],
    limit:    int,
    budget:   float,
    mode:     str,
) -> list[dict]:
    """Swap out players that violate nationality cap (best-effort, one pass)."""
    nat_cnt = Counter(p.get("nationality", "") for p in xi)
    used    = {_player_key(p) for p in xi}

    for nat, cnt in list(nat_cnt.items()):
        if not nat or cnt <= limit:
            continue
        excess = cnt - limit
        # Weakest players of this nationality first
        to_swap = sorted(
            [p for p in xi if p.get("nationality") == nat],
            key=lambda p: _score(p, mode),
        )[:excess]

        budget_left = budget - sum(float(p.get("market_value", 0) or 0) for p in xi)

        for victim in to_swap:
            pos   = victim.get("position", "MF")
            slack = float(victim.get("market_value", 0) or 0) + budget_left
            candidates = [
                p for p in by_pos.get(pos, [])
                if _player_key(p) not in used
                and p.get("nationality") != nat
                and float(p.get("market_value", 0) or 0) <= slack
            ]
            candidates.sort(key=lambda p: _score(p, mode), reverse=True)
            if candidates:
                rep = candidates[0]
                budget_left += (
                    float(victim.get("market_value", 0) or 0)
                    - float(rep.get("market_value", 0) or 0)
                )
                xi.remove(victim)
                xi.append(rep)
                used.discard(_player_key(victim))
                used.add(_player_key(rep))

    return xi
