"""
ScoutAI Agent  (Feature 16)
============================
Handles complex multi-step scouting queries via a ReAct loop:
  Thought → Action (tool call) → Observation (result) → … → Final Answer

Tools:
  search_players      – semantic + filter search
  analyze_team_gaps   – positional PSI weaknesses for a club
  find_transfers      – recommend players for a specific team + budget
  compare_players     – side-by-side stat comparison
  get_player_profile  – full stats deep dive
  simulate_window     – suggest 2-3 signings within a budget

SSE events emitted (all JSON-encoded after "data: "):
  {"type": "agent_step",        "label": "Calling search_players…"}
  {"type": "agent_tool_call",   "tool": "search_players", "args": {…}}
  {"type": "agent_tool_result", "tool": "search_players", "summary": "Found 5 FW…", "data": […]}
  {"type": "agent_token",       "content": "token"}
  {"type": "agent_done",        "turns": 3}
  {"type": "agent_error",       "message": "…"}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

import numpy as np

logger = logging.getLogger("scoutai.agent")

# ---------------------------------------------------------------------------
# Compact player formatter — keeps LLM context lean
# ---------------------------------------------------------------------------

def _compact(p: dict, fields=None) -> dict:
    """Return a minimal player summary safe to embed in the LLM context."""
    fields = fields or [
        "player", "age", "team", "league", "position",
        "market_value", "future_value", "psi",
        "goals_per90", "assists_per90",
        "style_label", "arc_phase", "contract_years",
        "injury_risk_label", "nationality",
    ]
    return {k: p[k] for k in fields if k in p}


def _fmt_player(p: dict) -> str:
    """One-line summary for embedding in Observations."""
    return (
        f"{p.get('player','?')} ({p.get('age','?')}y, {p.get('position','?')}, "
        f"{p.get('team','?')}) — PSI {p.get('psi',0):.2f}, "
        f"MV {p.get('market_value',0):.0f}M, "
        f"{p.get('style_label','?')}, {p.get('arc_phase','?')}"
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_search_players(
    query: str,
    position: str | None = None,
    max_age: int | None = None,
    max_price: float | None = None,
    league: str | None = None,
    nationality: str | None = None,
    top_k: int = 5,
) -> dict:
    from search import search_players
    filters: dict = {}
    if position:    filters["position"]    = position
    if max_age:     filters["max_age"]     = int(max_age)
    if max_price:   filters["max_price"]   = float(max_price)
    if league:      filters["league"]      = league
    if nationality: filters["nationality"] = nationality

    results = search_players(query, filters=filters, top_k=int(top_k))
    lines   = [_fmt_player(p) for p in results]
    return {
        "count":   len(results),
        "summary": f"Found {len(results)} players matching '{query[:50]}'",
        "players": lines,
        "data":    [_compact(p) for p in results],
    }


def _tool_analyze_team_gaps(team: str) -> dict:
    from data_loader import load_players
    df = load_players()

    # Fuzzy-match the team name
    team_lc = team.lower().strip()
    mask = df["team"].str.lower().str.contains(team_lc, na=False, regex=False)
    team_df = df[mask]

    if team_df.empty:
        return {"error": f"No players found for team '{team}'. Check the spelling."}

    canonical = team_df["team"].iloc[0]

    # Position-level PSI for this team vs the full-league average
    POS_ORDER = ["GK", "DF", "MF", "FW"]
    league_avg  = df.groupby("position_primary")["psi"].mean().to_dict()
    team_avg    = team_df.groupby("position_primary")["psi"].mean().to_dict()
    team_count  = team_df.groupby("position_primary")["psi"].count().to_dict()

    rows = []
    for pos in POS_ORDER:
        t_psi = team_avg.get(pos)
        l_psi = league_avg.get(pos)
        n     = team_count.get(pos, 0)
        if t_psi is None or n == 0:
            continue
        gap   = t_psi - l_psi if l_psi else 0
        rows.append({
            "position":   pos,
            "team_psi":   round(t_psi, 3),
            "league_avg": round(l_psi, 3) if l_psi else 0,
            "gap":        round(gap, 3),
            "n_players":  int(n),
            "rating":     "strong" if gap > 0.05 else "average" if gap > -0.05 else "weak",
        })

    rows.sort(key=lambda x: x["gap"])
    weak = [r for r in rows if r["rating"] == "weak"]
    lines = [
        f"{r['position']}: team PSI {r['team_psi']:.3f} vs league avg {r['league_avg']:.3f} "
        f"(gap {r['gap']:+.3f}) — {r['rating'].upper()}"
        for r in rows
    ]

    return {
        "team":          canonical,
        "summary":       f"{canonical} — weakest at: {', '.join(r['position'] for r in weak) or 'no clear weakness'}",
        "analysis":      lines,
        "weak_positions": [r["position"] for r in weak],
        "detail":        rows,
    }


def _tool_find_transfers(
    team: str,
    budget: float,
    formation: str | None = None,
    weak_positions: list[str] | None = None,
) -> dict:
    """Find best available players for a team's weakest positions within budget."""
    from search import search_players
    from data_loader import load_players
    df = load_players()

    # Determine which positions to target
    if not weak_positions:
        gap_result = _tool_analyze_team_gaps(team)
        weak_positions = gap_result.get("weak_positions", ["MF"])
    weak_positions = weak_positions[:3]   # cap at 3 positions

    team_lc = team.lower().strip()
    team_players_names = set(
        df[df["team"].str.lower().str.contains(team_lc, na=False)]["player"].tolist()
    )

    budget_per_pos = float(budget) / max(len(weak_positions), 1)

    recommendations: dict[str, list] = {}
    for pos in weak_positions:
        pos_queries = {
            "FW": "clinical forward goals scorer",
            "MF": "creative midfielder assists and goals",
            "DF": "strong reliable centre-back or fullback",
            "GK": "reliable goalkeeper high save percentage",
        }
        query = pos_queries.get(pos, "quality player")
        results = search_players(
            query,
            filters={"position": pos, "max_price": budget_per_pos},
            top_k=8,
        )
        # Exclude current team players
        results = [r for r in results if r.get("player") not in team_players_names][:3]
        recommendations[pos] = [_fmt_player(r) for r in results]

    lines = []
    for pos, players in recommendations.items():
        lines.append(f"[{pos}] (budget ~{budget_per_pos:.0f}M each):")
        lines.extend(f"  • {p}" for p in players)

    return {
        "team":            team,
        "budget":          budget,
        "target_positions": weak_positions,
        "summary":         f"Transfer targets for {team} ({budget:.0f}M budget) across {weak_positions}",
        "recommendations": lines,
        "data":            {pos: recommendations[pos] for pos in weak_positions},
    }


def _tool_compare_players(player_a: str, player_b: str) -> dict:
    from data_loader import load_players, player_to_dict
    df = load_players()

    results = {}
    for name in (player_a, player_b):
        mask = df["player"].str.lower().str.contains(name.lower(), na=False, regex=False)
        rows = df[mask]
        if rows.empty:
            return {"error": f"Player '{name}' not found. Check spelling."}
        row = rows.sort_values("minutes", ascending=False).iloc[0]
        results[name] = player_to_dict(row)

    def _stat(p: dict, key: str, fmt: str = ".2f") -> str:
        v = p.get(key, 0) or 0
        return format(float(v), fmt)

    pa, pb = list(results.values())
    STATS = [
        ("Goals/90",    "goals_per90",     ".3f"),
        ("Assists/90",  "assists_per90",   ".3f"),
        ("Shots/90",    "sh_per90",        ".2f"),
        ("Shot acc",    "g_per_shot",      ".3f"),
        ("Tackles",     "tackles_tkl",     ".0f"),
        ("PSI",         "psi",             ".3f"),
        ("Market Val",  "market_value",    ".0f"),
        ("Future Val",  "future_value",    ".0f"),
        ("Age",         "age",             ".0f"),
    ]
    rows = []
    for label, key, fmt in STATS:
        va = _stat(pa, key, fmt)
        vb = _stat(pb, key, fmt)
        rows.append(f"  {label:<12}  {pa['player'][:18]:<18} {va:>8}   {pb['player'][:18]:<18} {vb:>8}")

    header = f"  {'Stat':<12}  {pa['player'][:18]:<18} {'':>8}   {pb['player'][:18]:<18}"
    lines  = [header, "  " + "-" * 72] + rows

    summary_bits = []
    if float(pa.get("psi",0)) > float(pb.get("psi",0)):
        summary_bits.append(f"{pa['player']} has higher PSI ({pa.get('psi',0):.3f} vs {pb.get('psi',0):.3f})")
    else:
        summary_bits.append(f"{pb['player']} has higher PSI ({pb.get('psi',0):.3f} vs {pa.get('psi',0):.3f})")
    if float(pa.get("market_value",0)) < float(pb.get("market_value",0)):
        summary_bits.append(f"{pa['player']} is cheaper ({pa.get('market_value',0):.0f}M vs {pb.get('market_value',0):.0f}M)")
    else:
        summary_bits.append(f"{pb['player']} is cheaper ({pb.get('market_value',0):.0f}M vs {pa.get('market_value',0):.0f}M)")

    return {
        "summary":   " | ".join(summary_bits),
        "table":     lines,
        "player_a":  _compact(pa),
        "player_b":  _compact(pb),
    }


def _tool_get_player_profile(player_name: str) -> dict:
    from data_loader import load_players, player_to_dict
    df = load_players()

    mask = df["player"].str.lower().str.contains(player_name.lower(), na=False, regex=False)
    rows = df[mask]
    if rows.empty:
        return {"error": f"Player '{player_name}' not found."}
    row  = rows.sort_values("minutes", ascending=False).iloc[0]
    p    = player_to_dict(row)

    pos = p.get("position", "MF")
    if pos == "GK":
        key_stats = (
            f"Save% {p.get('gk_savepct',0):.1f}%, "
            f"CS% {p.get('gk_cspct',0):.1f}%, "
            f"GA/90 {p.get('gk_ga90',0):.2f}"
        )
    elif pos == "DF":
        key_stats = (
            f"Tackles {p.get('tackles_tkl',0):.0f}, "
            f"Int {p.get('int',0):.0f}, "
            f"Assists/90 {p.get('assists_per90',0):.3f}"
        )
    elif pos == "MF":
        key_stats = (
            f"Goals/90 {p.get('goals_per90',0):.3f}, "
            f"Assists/90 {p.get('assists_per90',0):.3f}, "
            f"Tackles {p.get('tackles_tkl',0):.0f}"
        )
    else:
        key_stats = (
            f"Goals/90 {p.get('goals_per90',0):.3f}, "
            f"Assists/90 {p.get('assists_per90',0):.3f}, "
            f"Shot acc {p.get('g_per_shot',0)*100:.1f}%"
        )

    gap = float(p.get("future_value", 0)) - float(p.get("market_value", 0))
    value_signal = (
        f"Undervalued by {gap:.0f}M" if gap > 5 else
        f"Overvalued by {abs(gap):.0f}M" if gap < -5 else "Fairly priced"
    )

    summary = (
        f"{p['player']} | {p.get('age','?')}y {p.get('position','?')} at {p.get('team','?')} ({p.get('league','')})\n"
        f"Style: {p.get('style_label','?')} | Arc: {p.get('arc_phase','?')} | "
        f"PSI: {p.get('psi',0):.3f} | Minutes: {p.get('minutes',0)}\n"
        f"Key stats: {key_stats}\n"
        f"Market: {p.get('market_value',0):.0f}M | Future: {p.get('future_value',0):.0f}M | {value_signal}\n"
        f"Contract: {p.get('contract_years',0)}yr until {p.get('contract_expires','?')} | "
        f"Injury risk: {p.get('injury_risk_label','?')} ({p.get('injury_risk',0)}/100)"
    )

    return {"summary": summary, "data": _compact(p)}


def _tool_simulate_window(
    team: str,
    budget: float,
    n_signings: int = 3,
) -> dict:
    """Suggest a complete set of signings: identifies gaps, allocates budget, finds best fits."""
    from search import search_players
    from data_loader import load_players

    df = load_players()
    budget = float(budget)
    n_signings = max(1, min(int(n_signings), 4))

    gap_result = _tool_analyze_team_gaps(team)
    if "error" in gap_result:
        return gap_result

    weak   = gap_result.get("weak_positions", [])
    detail = gap_result.get("detail", [])

    # Rank positions by weakness severity (most negative gap first)
    ranked = sorted(detail, key=lambda r: r["gap"])
    targets = [r["position"] for r in ranked[:n_signings]]

    team_lc = team.lower().strip()
    team_names = set(
        df[df["team"].str.lower().str.contains(team_lc, na=False)]["player"].tolist()
    )

    # Allocate budget: weakest position gets most, in ratio 50:30:20
    alloc_weights = [0.50, 0.30, 0.20, 0.15][:n_signings]
    total_weight  = sum(alloc_weights)
    budgets       = [budget * w / total_weight for w in alloc_weights]

    POS_QUERIES = {
        "FW": "prolific clinical striker goals",
        "MF": "creative midfielder assists press-resistant",
        "DF": "reliable ball-playing centre-back interceptions",
        "GK": "commanding goalkeeper high save rate",
    }

    signings: list[dict] = []
    for pos, alloc in zip(targets, budgets):
        results = search_players(
            POS_QUERIES.get(pos, "quality player"),
            filters={"position": pos, "max_price": alloc},
            top_k=6,
        )
        results = [r for r in results if r.get("player") not in team_names]
        if results:
            best = results[0]
            gap  = float(best.get("future_value", 0)) - float(best.get("market_value", 0))
            signings.append({
                "position":     pos,
                "allocated":    round(alloc, 1),
                "player":       best.get("player", "?"),
                "age":          best.get("age", "?"),
                "club":         best.get("team", "?"),
                "market_value": best.get("market_value", 0),
                "psi":          best.get("psi", 0),
                "style":        best.get("style_label", "?"),
                "value_gap":    round(gap, 1),
                "contract_yrs": best.get("contract_years", 0),
                "summary":      _fmt_player(best),
            })

    total_cost = sum(s["market_value"] for s in signings)
    lines = [f"Simulated {n_signings}-signing window for {gap_result['team']} (budget {budget:.0f}M):"]
    for s in signings:
        lines.append(
            f"  [{s['position']}] {s['player']} ({s['age']}y, {s['club']}) — "
            f"{s['market_value']:.0f}M, PSI {s['psi']:.2f}, {s['style']}"
        )
    lines.append(f"Total estimated cost: {total_cost:.0f}M / {budget:.0f}M budget")
    if budget - total_cost > 5:
        lines.append(f"Remaining: {budget - total_cost:.0f}M available for wages/fees")

    return {
        "team":        gap_result["team"],
        "budget":      budget,
        "n_signings":  n_signings,
        "weak_positions": targets,
        "signings":    signings,
        "total_cost":  round(total_cost, 1),
        "summary":     "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_DEFS = [
    {
        "name":        "search_players",
        "description": (
            "Semantic search for players. Use for: finding specific player types, "
            "scouting targets matching a description, or exploring by style/position."
        ),
        "params": "query(str), position(FW|MF|DF|GK), max_age(int), max_price(float), "
                  "league(EPL|La Liga|Bundesliga|Serie A|Ligue 1), nationality(ISO code), top_k(int=5)",
    },
    {
        "name":        "analyze_team_gaps",
        "description": (
            "Find positional PSI weaknesses for a club. "
            "Returns which positions are below the league average."
        ),
        "params": "team(str — exact or partial club name)",
    },
    {
        "name":        "find_transfers",
        "description": (
            "Recommend transfer targets for a specific team within a budget. "
            "Automatically identifies weak positions and suggests affordable upgrades."
        ),
        "params": "team(str), budget(float in M), formation(str optional), "
                  "weak_positions(list optional)",
    },
    {
        "name":        "compare_players",
        "description": "Side-by-side statistical comparison of two players.",
        "params":      "player_a(str), player_b(str)",
    },
    {
        "name":        "get_player_profile",
        "description": "Full stats deep dive for a single player.",
        "params":      "player_name(str)",
    },
    {
        "name":        "simulate_window",
        "description": (
            "Simulate a transfer window: identifies team gaps, allocates the budget "
            "across the weakest positions, and suggests the best available signings."
        ),
        "params": "team(str), budget(float in M), n_signings(int=3)",
    },
]

TOOL_FN_MAP = {
    "search_players":    _tool_search_players,
    "analyze_team_gaps": _tool_analyze_team_gaps,
    "find_transfers":    _tool_find_transfers,
    "compare_players":   _tool_compare_players,
    "get_player_profile": _tool_get_player_profile,
    "simulate_window":   _tool_simulate_window,
}

TOOL_ICONS = {
    "search_players":    "🔍",
    "analyze_team_gaps": "📊",
    "find_transfers":    "🎯",
    "compare_players":   "⚖️",
    "get_player_profile": "👤",
    "simulate_window":   "🪟",
}


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    tool_lines = []
    for t in TOOL_DEFS:
        tool_lines.append(f"- {t['name']}({t['params']}): {t['description']}")
    tools_str = "\n".join(tool_lines)

    return f"""You are ScoutAI, an expert football scouting agent. You have access to a live database of 2135 professional players across 5 top European leagues (EPL, La Liga, Bundesliga, Serie A, Ligue 1).

Available tools:
{tools_str}

Respond using this EXACT format — do not deviate:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"arg": "value"}}

After receiving an Observation, continue with another Thought/Action pair or conclude:

Thought: I have enough information.
Final Answer: [your complete scouting analysis in 3-5 sentences]

Rules:
- Always start with a Thought
- Action Input must be valid JSON
- Maximum 5 tool calls per query
- Final Answer must be concise and actionable
- Never make up player names or statistics — only use what tools return"""


# ---------------------------------------------------------------------------
# Tool call parser
# ---------------------------------------------------------------------------

_ACTION_RE      = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.IGNORECASE | re.DOTALL)
_FINAL_RE       = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract (tool_name, args_dict) from buffered LLM output, or None."""
    am = _ACTION_RE.search(text)
    im = _ACTION_INPUT_RE.search(text)
    if not (am and im):
        return None
    name = am.group(1).strip()
    if name not in TOOL_FN_MAP:
        return None
    try:
        args = json.loads(im.group(1))
    except json.JSONDecodeError:
        # Attempt to extract partial JSON
        try:
            raw = im.group(1)
            args = json.loads(raw + "}")   # common: missing closing brace
        except Exception:
            return None
    return name, args


def _parse_final_answer(text: str) -> str | None:
    """Extract the Final Answer text, or None if not present."""
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


def _has_complete_tool_call(text: str) -> bool:
    """True when we've accumulated a full Action + Action Input block."""
    return bool(_ACTION_RE.search(text) and _ACTION_INPUT_RE.search(text))


# ---------------------------------------------------------------------------
# Execute a single tool call
# ---------------------------------------------------------------------------

def execute_tool(name: str, args: dict) -> dict:
    """Run a tool synchronously and return its result dict."""
    fn = TOOL_FN_MAP.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        result = fn(**args)
        return result if isinstance(result, dict) else {"result": str(result)}
    except TypeError as exc:
        return {"error": f"Tool argument error: {exc}"}
    except Exception as exc:
        logger.exception("Tool %s raised: %s", name, exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Format observation for re-injection into conversation
# ---------------------------------------------------------------------------

def _format_observation(tool: str, result: dict) -> str:
    """Compact text version of tool result for the LLM conversation."""
    if "error" in result:
        return f"Error: {result['error']}"

    parts: list[str] = []

    # Each tool has a different natural result format
    if tool == "search_players":
        parts.append(result.get("summary", ""))
        parts.extend(result.get("players", []))
    elif tool == "analyze_team_gaps":
        parts.append(result.get("summary", ""))
        parts.extend(result.get("analysis", []))
    elif tool == "find_transfers":
        parts.append(result.get("summary", ""))
        parts.extend(result.get("recommendations", []))
    elif tool == "compare_players":
        parts.append(result.get("summary", ""))
        parts.extend(result.get("table", []))
    elif tool == "get_player_profile":
        parts.append(result.get("summary", ""))
    elif tool == "simulate_window":
        parts.append(result.get("summary", ""))

    return "\n".join(str(p) for p in parts if p)


# ---------------------------------------------------------------------------
# Main agent loop  — async generator of SSE-ready dicts
# ---------------------------------------------------------------------------

async def run_agent(query: str, max_turns: int = 5) -> AsyncIterator[dict]:
    """
    Async generator.  Yields event dicts — the caller wraps them in SSE.

    Event shapes:
      {"type": "agent_step",        "label": str}
      {"type": "agent_tool_call",   "tool": str, "args": dict, "icon": str}
      {"type": "agent_tool_result", "tool": str, "summary": str, "data": any}
      {"type": "agent_token",       "content": str}
      {"type": "agent_done",        "turns": int}
      {"type": "agent_error",       "message": str}
    """
    import llm as _llm

    backend = await _llm.detect_backend()
    if backend is None:
        yield {"type": "agent_error",
               "message": "No LLM backend available. Start ollama: ollama serve"}
        return

    system_prompt = _build_system_prompt()
    messages: list[dict] = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": query},
    ]

    turns_used = 0
    yield {"type": "agent_step", "label": "Thinking…"}

    import asyncio

    for turn in range(max_turns):
        # ── Stream one LLM turn ───────────────────────────────────────────
        buffer      = ""
        final_found = False

        async for token in _llm.stream_llm(
            _messages_to_prompt(messages),
            max_tokens=450,
        ):
            buffer += token
            yield {"type": "agent_token", "content": token}

            # As soon as we see a complete tool call, stop accumulating
            if _has_complete_tool_call(buffer):
                break

            # Final Answer — break as soon as we see the marker + some content
            if _FINAL_RE.search(buffer):
                # Give it a few more tokens to finish the sentence, then stop
                final_found = True
                # Don't break yet — keep collecting until end of stream or punct
                if buffer.rstrip().endswith((".", "!", "?", "\n")):
                    break

        turns_used = turn + 1

        # ── Check for Final Answer ────────────────────────────────────────
        final_text = _parse_final_answer(buffer)
        if final_text or final_found:
            yield {"type": "agent_done", "turns": turns_used}
            return

        # ── Check for tool call ───────────────────────────────────────────
        parsed = _parse_tool_call(buffer)
        if parsed is None:
            # LLM didn't produce a valid tool call or Final Answer.
            # If we have tool results in context (i.e. turn > 0), treat the
            # buffer as the final answer rather than erroring out.
            has_observations = any(
                "Observation:" in m.get("content", "") for m in messages
            )
            if has_observations and buffer.strip():
                # Best-effort: present the raw LLM output as the answer
                yield {"type": "agent_done", "turns": turns_used}
                return
            # Otherwise — last turn with nothing useful
            if turn == max_turns - 1:
                yield {"type": "agent_error",
                       "message": "Agent did not produce a final answer. Try rephrasing your question."}
                return
            # Mid-loop: continue to next turn
            messages.append({"role": "assistant", "content": buffer})
            messages.append({
                "role": "user",
                "content": "Please provide your Final Answer now using the format:\nFinal Answer: [your answer]"
            })
            yield {"type": "agent_step", "label": "Thinking…"}
            continue

        tool_name, tool_args = parsed
        icon = TOOL_ICONS.get(tool_name, "🔧")

        yield {
            "type": "agent_tool_call",
            "tool": tool_name,
            "args": tool_args,
            "icon": icon,
            "label": f"{icon} Calling {tool_name}…",
        }

        # ── Execute tool (run in executor to avoid blocking event loop) ───
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: execute_tool(tool_name, tool_args)
        )

        observation = _format_observation(tool_name, result)
        yield {
            "type":    "agent_tool_result",
            "tool":    tool_name,
            "icon":    icon,
            "summary": result.get("summary", observation[:160]),
            "data":    result.get("data") or result.get("detail") or result.get("signings"),
        }

        # ── Append to conversation and continue ──────────────────────────
        messages.append({"role": "assistant", "content": buffer})
        # Count how many tool observations we've accumulated
        tool_obs_count = sum(
            1 for m in messages if m["role"] == "user" and "Observation:" in m.get("content", "")
        )
        # After 2+ tool results, aggressively demand a Final Answer
        if tool_obs_count >= 2:
            next_user_msg = (
                f"Observation:\n{observation}\n\n"
                "IMPORTANT: You have already called several tools. "
                "DO NOT call any more tools. "
                "Write ONLY your Final Answer now:\n"
                "Final Answer: [your concise scouting analysis based on the data above]"
            )
        else:
            next_user_msg = (
                f"Observation:\n{observation}\n\n"
                "Based on this data, provide your Final Answer:\n"
                "Final Answer: [concise scouting analysis]"
            )
        messages.append({"role": "user", "content": next_user_msg})

        yield {"type": "agent_step", "label": "Thinking…"}

    yield {"type": "agent_error", "message": "Reached maximum tool calls without a conclusion."}


# ---------------------------------------------------------------------------
# Convert messages → flat prompt for ollama /api/generate
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: list[dict]) -> str:
    """
    Flatten a messages list into a single prompt string.
    Uses llama3-style format (simple role labels).
    """
    parts = []
    for m in messages:
        role    = m["role"]
        content = m["content"].strip()
        if role == "system":
            parts.append(f"<|system|>\n{content}\n<|end|>")
        elif role == "user":
            parts.append(f"\n<|user|>\n{content}\n<|end|>")
        elif role == "assistant":
            parts.append(f"\n<|assistant|>\n{content}")
    parts.append("\n<|assistant|>\n")
    return "".join(parts)
