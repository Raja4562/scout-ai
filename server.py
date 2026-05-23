"""
ScoutAI - FastAPI Backend
All API endpoints. Serves the frontend, handles streaming search via SSE,
and manages per-session memory (team, budget, formation, shortlist).
Phase 4: Multi-league (5 leagues, 2135 players).
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from config import TEMPLATE_DIR, TOP_K_DEFAULT, SUPPORTED_LEAGUES, LEAGUE_COLORS, LEAGUE_LABELS
from data_loader import load_players, player_to_dict, reload_players, get_player_seasons
from search import search_players, find_similar, generate_summary
from embeddings import explain_query
import llm as _llm
from memory import (
    empty_memory, extract_memory,
    add_to_shortlist, remove_from_shortlist,
    memory_to_context, memory_applies_filter,
    detect_reset, detect_refinement, stack_search_context,
    active_filters_to_labels, clear_active_filters, remove_active_filter,
)
from report import (
    generate_player_report, build_scout_prompt, extract_structured_sections,
    build_explain_prompt, generate_explain_bullets,
)
from agent import run_agent
from bestxi import build_best_xi, FORMATION_POSITIONS, MODE_LABELS, MODE_DESCRIPTIONS
from transfer_window import plan_transfer_window, POSITION_META
from dna import build_team_dna, find_dna_matches

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("scoutai.server")

_DESCRIPTION = """
## ScoutAI — Football Intelligence Platform

A production-grade football scouting API powered by **semantic search**, **ML valuation models**,
and **streaming LLM analysis** across 2,100+ players from five top leagues.

### Streaming endpoints
All analysis endpoints use **Server-Sent Events (SSE)**.
Set `Accept: text/event-stream` and read the `data:` lines.

### Feature set
| Endpoint group | What it does |
|---|---|
| **Search** | Natural-language player discovery — embeds query, cosine-ranks players |
| **Best XI** | Constrained squad optimisation (6 scoring modes, 6 formations, budget cap) |
| **Transfer Window** | Fee negotiation bands, signing order, sell recommendations |
| **DNA Matching** | PSI-weighted team identity vector → find stylistic fits in the whole database |
| **Hidden Gems** | ML value-gap detection — future_value minus market_value, ranked |
| **Cheaper Alts** | Cosine-similar players within a price ratio target |
| **Scout Report** | Full structured report: radar data, stats, fee model, LLM narrative |
| **Session Memory** | Per-session preference stacking, shortlist management |
"""

_TAGS = [
    {"name": "core",            "description": "Health check, status, league & style metadata"},
    {"name": "search",          "description": "Semantic player search — returns SSE stream of player cards + summary"},
    {"name": "players",         "description": "Individual player lookup, similarity, scouting reports, season progression"},
    {"name": "agent",           "description": "ReAct agent for complex multi-step scouting queries (SSE)"},
    {"name": "memory",          "description": "Per-session preference memory — stacked filters, remembered context"},
    {"name": "shortlist",       "description": "Session shortlist: add, remove, retrieve"},
    {"name": "best-xi",         "description": "Constrained squad optimisation — pick the best XI within budget & constraints"},
    {"name": "transfer-window", "description": "Transfer window planner — targets, fee bands, signing order, sell candidates"},
    {"name": "gems",            "description": "Hidden gems / undervalued player detection"},
    {"name": "dna",             "description": "Team DNA / tactical identity vector matching"},
    {"name": "cheaper",         "description": "Cost-effective similar player discovery"},
    {"name": "scout-report",    "description": "Structured player scouting report — stats, radar, valuation, LLM narrative"},
    {"name": "admin",           "description": "Data management — hot-reload CSV without server restart"},
]

app = FastAPI(
    title       = "ScoutAI",
    version     = "2.0",
    description = _DESCRIPTION,
    contact     = {"name": "ScoutAI API", "url": "https://github.com/scoutai"},
    openapi_tags = _TAGS,
)
templates = Jinja2Templates(directory=TEMPLATE_DIR)
_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_ready: bool = False

# Per-session memory keyed by browser crypto.randomUUID()
_memory: dict[str, dict] = {}


def _get_memory(sid: str) -> dict:
    if sid not in _memory:
        _memory[sid] = empty_memory()
    return _memory[sid]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _ready
    loop = asyncio.get_running_loop()
    logger.info("Building player embeddings on startup ...")
    await loop.run_in_executor(_executor, _preload)
    _ready = True
    logger.info("ScoutAI ready.")


def _preload():
    from embeddings import build_or_load_embeddings
    df = load_players()
    build_or_load_embeddings(df)


@app.on_event("startup")
async def _detect_llm():
    """Probe LLM backend once at startup so /api/status always has it."""
    backend = await _llm.detect_backend()
    logger.info("LLM backend detected: %s", _llm.backend_label())


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query:       str
    sid:         str = ""          # session ID from browser localStorage
    position:    Optional[str]   = None
    league:      Optional[str]   = None
    max_age:     Optional[int]   = None
    max_price:   Optional[float] = None
    expiring:    bool = False       # filter to contract_years <= 1 only
    style:       Optional[str]   = None   # style archetype filter (Feature 11)
    nationality: Optional[str]   = None   # ISO code, e.g. "BRA" (Feature 14)
    arc_phase:   Optional[str]   = None   # "pre-peak" | "peak" | "post-peak" (Feature 14)
    min_minutes: Optional[int]   = None   # minimum season minutes (Feature 14)
    top_k:       int = TOP_K_DEFAULT


class AgentRequest(BaseModel):
    query:     str
    max_turns: int = 5


class ShortlistRequest(BaseModel):
    player:       str
    team:         str
    position:     str   = ""
    age:          int   = 0
    market_value: float = 0.0
    future_value: float = 0.0
    psi:          float = 0.0
    goals_per90:  float = 0.0
    assists_per90:float = 0.0
    xg_per90:     float = 0.0


# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status", tags=["core"], summary="System readiness and backend info")
async def status():
    df = load_players() if _ready else None
    league_counts = {}
    if df is not None and "league" in df.columns:
        league_counts = df.groupby("league").size().to_dict()
    return {
        "ready":         _ready,
        "version":       "1.0",
        "player_count":  len(df) if df is not None else 0,
        "leagues":       league_counts,
        "league_colors": LEAGUE_COLORS,
        "league_labels": LEAGUE_LABELS,
        "llm_backend":   _llm.backend_label(),   # Feature 15
    }


@app.get("/api/leagues", tags=["core"], summary="Available leagues with player counts and branding")
async def get_leagues():
    """Return available leagues and their player counts."""
    if not _ready:
        return {"leagues": [], "colors": LEAGUE_COLORS, "labels": LEAGUE_LABELS}
    df = load_players()
    if "league" not in df.columns:
        return {
            "leagues": [{"id": "EPL", "label": "Premier League", "count": len(df), "color": LEAGUE_COLORS["EPL"]}],
            "colors":  LEAGUE_COLORS,
            "labels":  LEAGUE_LABELS,
        }
    leagues = []
    for league_id in SUPPORTED_LEAGUES:
        count = int((df["league"] == league_id).sum())
        if count > 0:
            leagues.append({
                "id":    league_id,
                "label": LEAGUE_LABELS.get(league_id, league_id),
                "count": count,
                "color": LEAGUE_COLORS.get(league_id, "#38bdf8"),
            })
    return {"leagues": leagues, "colors": LEAGUE_COLORS, "labels": LEAGUE_LABELS}


# ---------------------------------------------------------------------------
# Search (SSE)
# ---------------------------------------------------------------------------

@app.post(
    "/api/search",
    tags=["search"],
    summary="Semantic player search (SSE)",
    description="""
Stream player results for a natural-language query.

**SSE event types**

| type | payload | meaning |
|---|---|---|
| `step` | `{label, status}` | pipeline progress |
| `memory` | `{signal, state}` | session memory update |
| `context` | `{labels, is_refinement}` | active filter chips |
| `interpreted` | `{filters}` | newly parsed filters |
| `total` | `{total}` | number of results found |
| `player` | `{player, index}` | a single player card |
| `similar` | `{player, similar}` | players similar to top result |
| `token` | `{content}` | streaming summary word |
| `done` | — | stream complete |
""",
)
async def api_search(req: SearchRequest):
    if not _ready:
        raise HTTPException(503, "ScoutAI is still loading - please wait a moment.")

    sid = req.sid or ""

    async def generate():
        loop = asyncio.get_running_loop()

        def _step(label: str, status: str = "active"):
            return f"data: {json.dumps({'type': 'step', 'label': label, 'status': status})}\n\n"

        # ── 0. Load + update high-level session memory ─────────────────────
        mem = _get_memory(sid) if sid else empty_memory()
        mem, hl_signals = extract_memory(req.query, mem)
        if sid:
            _memory[sid] = mem

        # High-level memory toast notifications
        for sig in hl_signals:
            yield f"data: {json.dumps({'type': 'memory', 'signal': sig, 'state': mem})}\n\n"

        # ── 1. Parsing + preference stacking ──────────────────────────────
        yield _step("Analyzing query")
        await asyncio.sleep(0.04)

        # Reset detection ("start over", "forget that", ...)
        is_reset = detect_reset(req.query)
        if is_reset:
            mem = clear_active_filters(mem)
            if sid:
                _memory[sid] = mem
            yield f"data: {json.dumps({'type': 'context_reset'})}\n\n"

        # Refinement vs new-query detection
        is_refine = (not is_reset) and detect_refinement(req.query, mem)

        # Parse NL filters from this query text
        from search import parse_filters as _parse_nl
        parsed_nl = await loop.run_in_executor(_executor, lambda: _parse_nl(req.query))

        # Stack onto accumulated preference memory (pure Python — no I/O, no thread needed)
        try:
            aug_query, stacked_mem, ctx_signals = stack_search_context(
                req.query, mem, parsed_nl, is_refine
            )
        except Exception as _e:
            logger.exception("stack_search_context failed: %s", _e)
            aug_query, stacked_mem, ctx_signals = req.query, mem, []

        # Persist updated memory (safe: dict returned by stack_search_context is a new obj)
        if sid:
            _memory[sid] = stacked_mem

        # ── 2. Build search filters: stack → explicit UI overrides ─────────
        filters: dict = dict(stacked_mem.get("active_filters", {}))

        # Explicit UI params always override remembered preferences
        if req.position:    filters["position"]    = req.position
        if req.league:      filters["league"]      = req.league
        if req.max_age:     filters["max_age"]     = req.max_age
        if req.max_price:   filters["max_price"]   = req.max_price
        if req.expiring:    filters["expiring"]    = True
        if req.style:       filters["style"]       = req.style
        if req.nationality: filters["nationality"] = req.nationality
        if req.arc_phase:   filters["arc_phase"]   = req.arc_phase
        if req.min_minutes: filters["min_minutes"] = req.min_minutes

        # High-level budget cap (if no explicit UI price and not already in stack)
        if not filters.get("max_price") and stacked_mem.get("max_price"):
            filters["max_price"] = stacked_mem["max_price"]

        # ── 3. Emit context bar (full preference stack) ────────────────────
        all_labels = active_filters_to_labels(stacked_mem.get("active_filters", {}))
        for i, q in enumerate(stacked_mem.get("text_qualifiers", [])):
            all_labels[f"__qual_{i}"] = f"✦ {q}"   # ✦ qualifier

        if all_labels:
            yield f"data: {json.dumps({'type': 'context', 'labels': all_labels, 'is_refinement': is_refine})}\n\n"

        # ── 4. Emit per-query interpreted chips (new changes only) ──────────
        if ctx_signals:
            new_labels = {f"new_{i}": s["label"] for i, s in enumerate(ctx_signals)}
            yield f"data: {json.dumps({'type': 'interpreted', 'filters': new_labels, 'is_refinement': is_refine})}\n\n"

        # ── 5. Search ─────────────────────────────────────────────────────
        df        = load_players()
        n_players = len(df)
        yield _step(f"Searching {n_players:,} players")

        search_query = aug_query or req.query
        results = await loop.run_in_executor(
            _executor,
            lambda: search_players(search_query, filters, req.top_k),
        )

        # ---- STEP 3: Scoring ------------------------------------------
        yield _step(f"Scoring {len(results)} candidates")
        await asyncio.sleep(0.04)

        # Mark shortlisted players
        shortlist_keys = {(p["player"], p["team"]) for p in stacked_mem.get("shortlist", [])}
        for r in results:
            r["shortlisted"] = (r["player"], r["team"]) in shortlist_keys

        # Send total count so the grid knows how many cards are coming
        yield f"data: {json.dumps({'type': 'total', 'total': len(results)})}\n\n"

        # Stream cards one by one so they appear progressively
        for i, player in enumerate(results):
            yield f"data: {json.dumps({'type': 'player', 'player': player, 'index': i})}\n\n"
            # Small cascade delay for the first 6 cards - visible stagger effect
            if i < 6:
                await asyncio.sleep(0.06)

        # ---- STEP 4: Generating insights ------------------------------
        yield _step("Generating scouting insights")

        # Fetch similar players for the top result in the background
        if results:
            top_similar = await loop.run_in_executor(
                _executor,
                lambda: find_similar(results[0]["player"]),
            )
            yield f"data: {json.dumps({'type': 'similar', 'player': results[0]['player'], 'similar': top_similar})}\n\n"

        # Stream the search summary token-by-token
        ctx     = memory_to_context(stacked_mem)
        summary = generate_summary(req.query, results, filters, context=ctx)
        for word in summary.split():
            chunk = json.dumps({"type": "token", "content": word + " "})
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.025)

        yield _step("Done", "done")
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Player endpoints
# ---------------------------------------------------------------------------

@app.get("/api/player/{player_name}", tags=["players"], summary="Full player record by name")
async def get_player(player_name: str):
    df = load_players()
    matches = df[df["player"].str.lower() == player_name.lower()]
    if matches.empty:
        raise HTTPException(404, f"Player '{player_name}' not found")
    row     = matches.sort_values("minutes", ascending=False).iloc[0]
    similar = find_similar(str(row["player"]))
    data    = player_to_dict(row)
    data["similar"] = similar
    return data


@app.post(
    "/api/report",
    tags=["players"],
    summary="Full scouting report for a player (SSE)",
    description="Streams LLM narrative + structured value/verdict sections. Falls back to template if no LLM.",
)
async def api_report(player: dict = Body(...)):
    """
    SSE endpoint: streams a full scouting report for a player.

    LLM mode  (ollama / transformers available):
      1. [SCOUTING REPORT] — LLM narrative, tokens arrive as they're generated
      2. [VALUE ASSESSMENT] — template numbers (reliable)
      3. [VERDICT: X]       — template verdict tag + rationale

    Template mode (no local LLM):
      Full 5-section template streamed word-by-word as before.

    Both modes emit identical SSE events so the frontend is unchanged.
    """
    if not _ready:
        raise HTTPException(503, "ScoutAI is still loading.")

    async def generate():
        loop    = asyncio.get_running_loop()
        backend = await _llm.detect_backend()

        # ── LLM mode ──────────────────────────────────────────────────────
        if backend is not None:
            # 1. Build prompt (sync, fast)
            prompt = await loop.run_in_executor(
                _executor, lambda: build_scout_prompt(player)
            )

            # 2. Stream the LLM narrative
            yield f"data: {json.dumps({'type': 'section_start', 'title': 'SCOUTING REPORT', 'verdict': '', 'llm': True, 'model': _llm.backend_label()})}\n\n"

            token_count = 0
            try:
                async for token in _llm.stream_llm(prompt, max_tokens=250):
                    yield f"data: {json.dumps({'type': 'report_token', 'content': token})}\n\n"
                    token_count += 1
            except Exception as exc:
                logger.warning("LLM stream error, falling back to template: %s", exc)
                backend = None   # trip fallback below

            if backend is not None:
                yield f"data: {json.dumps({'type': 'section_end'})}\n\n"
                await asyncio.sleep(0.15)

                # 3. Template VALUE + VERDICT sections (reliable numbers)
                structured = await loop.run_in_executor(
                    _executor, lambda: extract_structured_sections(player)
                )
                for sec in structured:
                    yield f"data: {json.dumps({'type': 'section_start', 'title': sec['section'], 'verdict': sec.get('verdict', '')})}\n\n"
                    await asyncio.sleep(0.08)
                    words = sec["text"].split()
                    for word in words:
                        yield f"data: {json.dumps({'type': 'report_token', 'content': word + ' '})}\n\n"
                        await asyncio.sleep(0.018 if sec.get("verdict") else 0.022)
                    yield f"data: {json.dumps({'type': 'section_end'})}\n\n"
                    await asyncio.sleep(0.12)

                yield f"data: {json.dumps({'type': 'report_done'})}\n\n"
                return

        # ── Template fallback (no LLM or mid-stream failure) ──────────────
        sections = await loop.run_in_executor(
            _executor, lambda: generate_player_report(player)
        )
        for sec in sections:
            yield f"data: {json.dumps({'type': 'section_start', 'title': sec['section'], 'verdict': sec.get('verdict', '')})}\n\n"
            await asyncio.sleep(0.1)
            words = sec["text"].split()
            for word in words:
                yield f"data: {json.dumps({'type': 'report_token', 'content': word + ' '})}\n\n"
                await asyncio.sleep(0.018 if sec.get("verdict") else 0.022)
            yield f"data: {json.dumps({'type': 'section_end'})}\n\n"
            await asyncio.sleep(0.15)

        yield f"data: {json.dumps({'type': 'report_done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/api/explain",
    tags=["players"],
    summary="'Why this player?' explainability (SSE)",
    description="Given a player + the original query + active filters, streams a concise LLM explanation of why the player was recommended.",
)
async def api_explain(body: dict = Body(...)):
    """
    SSE endpoint — Feature 19: "Why this player?" explainability.

    Accepts: {player: {...}, query: str, filters: {...}}
    Streams:
      {"type": "explain_token",   "content": "..."}   — LLM mode
      {"type": "explain_bullet",  "text":    "..."}   — template fallback
      {"type": "explain_done"}
    """
    if not _ready:
        raise HTTPException(503, "ScoutAI is still loading.")

    player  = body.get("player",  {})
    query   = body.get("query",   "")
    filters = body.get("filters", {})

    async def generate():
        loop    = asyncio.get_running_loop()
        backend = await _llm.detect_backend()

        if backend is not None:
            # LLM mode — stream prose explanation token by token
            prompt = await loop.run_in_executor(
                _executor, lambda: build_explain_prompt(player, query, filters)
            )
            try:
                async for token in _llm.stream_llm(prompt, max_tokens=120):
                    yield f"data: {json.dumps({'type': 'explain_token', 'content': token})}\n\n"
            except Exception as exc:
                logger.warning("explain LLM stream error: %s", exc)
                # Fall through to bullet fallback
                bullets = await loop.run_in_executor(
                    _executor, lambda: generate_explain_bullets(player, query, filters)
                )
                for b in bullets:
                    yield f"data: {json.dumps({'type': 'explain_bullet', 'text': b})}\n\n"
        else:
            # Template fallback — emit structured bullets
            bullets = await loop.run_in_executor(
                _executor, lambda: generate_explain_bullets(player, query, filters)
            )
            for b in bullets:
                yield f"data: {json.dumps({'type': 'explain_bullet', 'text': b})}\n\n"

        yield f"data: {json.dumps({'type': 'explain_done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/api/agent",
    tags=["agent"],
    summary="ReAct agent for complex scouting queries (SSE)",
    description="Multi-turn reasoning agent. Streams thought, tool-call, tool-result, and answer events. Use for queries like 'find me a young left-back under €30M who can cover the right side'.",
)
async def api_agent(req: AgentRequest):
    """
    SSE endpoint for the ReAct agent (Feature 16).

    Streams a sequence of events representing the agent's reasoning loop:
      {"type": "agent_step",        "label": "Thinking…"}
      {"type": "agent_token",       "content": "token text"}
      {"type": "agent_tool_call",   "tool": "search_players", "args": {…}, "icon": "🔍", "label": "…"}
      {"type": "agent_tool_result", "tool": "search_players", "icon": "🔍", "summary": "…", "data": […]}
      {"type": "agent_done",        "turns": 2}
      {"type": "agent_error",       "message": "…"}
    """
    if not _ready:
        raise HTTPException(503, "ScoutAI is still loading.")

    async def generate():
        try:
            async for event in run_agent(req.query, max_turns=req.max_turns):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.exception("Agent error: %s", exc)
            yield f"data: {json.dumps({'type': 'agent_error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/similar/{player_name}", tags=["players"], summary="Semantically similar players")
async def api_similar(player_name: str):
    if not _ready:
        raise HTTPException(503, "ScoutAI is still loading.")
    loop    = asyncio.get_running_loop()
    similar = await loop.run_in_executor(
        _executor, lambda: find_similar(player_name),
    )
    return {"similar": similar}


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------

@app.get("/api/memory/{sid}", tags=["memory"], summary="Get full session memory state")
async def get_memory(sid: str):
    """Return the full memory state for a session."""
    return _get_memory(sid)


@app.delete("/api/memory/{sid}", tags=["memory"], summary="Wipe all memory for a session")
async def clear_memory(sid: str):
    """Wipe everything remembered for a session."""
    _memory.pop(sid, None)
    return {"ok": True, "memory": empty_memory()}


@app.delete("/api/memory/{sid}/field/{field}", tags=["memory"], summary="Clear one memory field (team/budget/formation)")
async def clear_memory_field(sid: str, field: str):
    """Clear a single memory field (team, budget, max_price, formation)."""
    allowed = {"team", "budget", "max_price", "formation"}
    if field not in allowed:
        raise HTTPException(400, f"Unknown field '{field}'")
    mem = _get_memory(sid)
    mem[field] = None
    _memory[sid] = mem
    return {"ok": True, "memory": mem}


@app.delete("/api/memory/{sid}/context", tags=["memory"], summary="Clear active filter stack (keep team/budget)")
async def clear_context(sid: str):
    """Wipe the active preference stack (position/age/price/qualifiers) but keep team/budget."""
    mem = _get_memory(sid)
    mem = clear_active_filters(mem)
    _memory[sid] = mem
    return {"ok": True, "memory": mem}


@app.delete("/api/memory/{sid}/filter/{key}", tags=["memory"], summary="Remove a single filter from the preference stack")
async def clear_filter_key(sid: str, key: str):
    """Remove a single key from the active preference stack."""
    mem = _get_memory(sid)
    mem = remove_active_filter(mem, key)
    _memory[sid] = mem
    return {"ok": True, "memory": mem, "labels": active_filters_to_labels(mem.get("active_filters", {}))}


# ---------------------------------------------------------------------------
# Shortlist endpoints
# ---------------------------------------------------------------------------

@app.post("/api/shortlist/{sid}", tags=["shortlist"], summary="Add a player to the shortlist")
async def shortlist_add(sid: str, req: ShortlistRequest):
    """Add a player to the session shortlist."""
    mem = _get_memory(sid)
    player_dict = req.model_dump()
    mem, added = add_to_shortlist(mem, player_dict)
    _memory[sid] = mem
    return {
        "added": added,
        "count": len(mem["shortlist"]),
        "shortlist": mem["shortlist"],
    }


@app.delete("/api/shortlist/{sid}/{player_name}", tags=["shortlist"], summary="Remove a player from the shortlist")
async def shortlist_remove(sid: str, player_name: str, team: str = ""):
    """Remove a player from the session shortlist."""
    mem = _get_memory(sid)
    mem = remove_from_shortlist(mem, player_name, team)
    _memory[sid] = mem
    return {
        "count": len(mem["shortlist"]),
        "shortlist": mem["shortlist"],
    }


@app.get("/api/shortlist/{sid}", tags=["shortlist"], summary="Get the current shortlist")
async def shortlist_get(sid: str):
    """Get the current shortlist."""
    mem = _get_memory(sid)
    return {"shortlist": mem.get("shortlist", []), "count": len(mem.get("shortlist", []))}


# ---------------------------------------------------------------------------
# Admin: hot-reload data
# ---------------------------------------------------------------------------
# Feature 25 — Best XI Builder
# ---------------------------------------------------------------------------

@app.post(
    "/api/bestxi",
    tags=["best-xi"],
    summary="Build optimal XI within budget and constraints",
    description="""
Constrained squad optimisation. Returns the highest-scoring 11-player squad
that fits within the given budget and formation.

**Scoring modes**: `psi` (raw), `value` (PSI/€M), `future` (upside), `young` (U24 bonus),
`transfer` (availability-weighted), `balanced` (55% PSI + 45% value).

Also returns an unconstrained (unlimited budget) XI for benchmark comparison.
""",
)
async def api_bestxi(body: dict = Body(...)):
    """
    Build the optimal XI from the full player database subject to:
      budget, formation, mode, league, age range, team exclusion,
      locked players, max-per-club, nationality cap, variation seed.

    Also returns an unconstrained (unlimited budget) XI for comparison.
    """
    budget            = float(body.get("budget", 200))
    formation         = str(body.get("formation", "433"))
    mode              = str(body.get("mode", "psi"))
    league            = str(body.get("league", "all"))
    max_age           = body.get("max_age")
    min_age           = body.get("min_age")
    exclude_team      = body.get("exclude_team") or None
    locked_list       = body.get("locked", [])        # [{player, team}, ...]
    max_per_team      = int(body.get("max_per_team", 3))
    nationality_limit = body.get("nationality_limit")  # None = no cap
    variation         = int(body.get("variation", 0))

    if max_age is not None:
        max_age = int(max_age)
    if min_age is not None:
        min_age = int(min_age)
    if nationality_limit is not None:
        nationality_limit = int(nationality_limit)

    locked_keys = {f"{p.get('player','')}|{p.get('team','')}" for p in locked_list}

    df   = load_players()
    loop = asyncio.get_event_loop()
    all_players = await loop.run_in_executor(
        _executor, lambda: [player_to_dict(r) for _, r in df.iterrows()]
    )

    # Constrained XI
    result = await loop.run_in_executor(
        _executor,
        lambda: build_best_xi(
            all_players,
            formation         = formation,
            budget            = budget,
            mode              = mode,
            league            = league,
            max_age           = max_age,
            min_age           = min_age,
            exclude_team      = exclude_team,
            locked_keys       = locked_keys,
            max_per_team      = max_per_team,
            nationality_limit = nationality_limit,
            variation         = variation,
        ),
    )

    # Unconstrained (unlimited budget) XI for comparison overlay
    unconstrained = await loop.run_in_executor(
        _executor,
        lambda: build_best_xi(
            all_players,
            formation    = formation,
            budget       = 999_999,
            mode         = mode,
            league       = league,
            max_age      = max_age,
            min_age      = min_age,
            exclude_team = exclude_team,
            max_per_team = max_per_team,
        ),
    )

    result["unconstrained_xi"]    = unconstrained.get("xi", [])
    result["unconstrained_score"] = unconstrained.get("total_psi", 0)

    logger.info(
        "Best XI: budget=€%sM formation=%s mode=%s feasible=%s avg_psi=%.2f",
        budget, formation, mode, result["feasible"], result.get("avg_psi", 0),
    )
    return result


@app.get("/api/bestxi/meta", tags=["best-xi"], summary="Formation options and scoring mode definitions")
async def api_bestxi_meta():
    """Return formation options and mode definitions for the UI."""
    return {
        "formations": list(FORMATION_POSITIONS.keys()),
        "modes": [
            {"key": k, "label": v, "desc": MODE_DESCRIPTIONS.get(k, "")}
            for k, v in MODE_LABELS.items()
        ],
        "formation_positions": FORMATION_POSITIONS,
    }

# ---------------------------------------------------------------------------

@app.post("/api/admin/reload", tags=["admin"], summary="Hot-reload player data without server restart")
async def admin_reload():
    """
    Hot-reload player data from disk without restarting the server.
    Called automatically by pipeline/refresh.py after a data update.

    Steps:
      1. Reload players CSV into memory
      2. Rebuild embedding matrix from fresh player texts
      3. Return new player/league counts
    """
    global _ready
    _ready = False
    try:
        loop = asyncio.get_running_loop()

        def _do_reload():
            from embeddings import build_or_load_embeddings, reset_embeddings
            reset_embeddings()
            df = reload_players()
            build_or_load_embeddings(df, force_rebuild=True)
            return df

        df = await loop.run_in_executor(_executor, _do_reload)
        _ready = True

        league_counts = {}
        if "league" in df.columns:
            league_counts = df.groupby("league").size().to_dict()

        logger.info(
            "Hot-reload complete: %d players across %d league(s)",
            len(df), len(league_counts),
        )
        return {
            "ok":           True,
            "player_count": len(df),
            "league_count": len(league_counts),
            "leagues":      league_counts,
        }
    except Exception as exc:
        _ready = True   # restore so the app stays usable on partial failure
        logger.error("Hot-reload failed: %s", exc)
        raise HTTPException(500, f"Reload failed: {exc}")


# ── Style clustering endpoints ──────────────────────────────────────────────

@app.get("/api/embeddings/explain", tags=["admin"], summary="Debug: inspect embedding space for a query")
async def embeddings_explain(q: str = "press-resistant midfielder"):
    """
    Debug endpoint: show which players a query retrieves semantically
    and the first 200 chars of their embedding text.
    Useful for validating that the embedding space is well-structured.
    """
    df = load_players()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, explain_query, q, df)
    return result


@app.get("/api/styles", tags=["core"], summary="All style archetypes with player counts")
async def get_styles():
    """Return all distinct style labels with player counts (for the filter UI)."""
    df = load_players()
    if "style_label" not in df.columns:
        return {"styles": []}
    counts = df["style_label"].value_counts().to_dict()
    # Group by position archetype for the UI
    pos_map = {
        "FW": ["Poacher", "Creative Forward", "Pressing Forward", "Target Man", "Inverted Winger"],
        "MF": ["Ball Winner", "Box-to-Box", "Attacking Midfielder", "Playmaker",
               "Press-Resistant 8", "Holding Midfielder"],
        "DF": ["Attacking Fullback", "Ball-Playing CB", "Defensive Anchor",
               "Sweeper CB", "Stopper"],
        "GK": ["Shot-Stopper", "Sweeper Keeper", "Distribution GK"],
    }
    styles = []
    for pos, labels in pos_map.items():
        for lbl in labels:
            if lbl in counts:
                styles.append({"label": lbl, "position": pos, "count": counts[lbl]})
    # Anything not in the map
    known = {l for ls in pos_map.values() for l in ls}
    for lbl, cnt in counts.items():
        if lbl not in known and lbl not in ("Unknown", ""):
            styles.append({"label": lbl, "position": "?", "count": cnt})
    return {"styles": styles}


@app.get("/api/styles/scatter", tags=["core"], summary="UMAP scatter-plot data for style clustering visualisation")
async def styles_scatter(position: str = "all", league: str = "all"):
    """
    Return UMAP scatter-plot data for all players (optionally filtered by position/league).
    Each item: {player, team, style_label, position, umap_x, umap_y, psi, market_value}.
    """
    df = load_players()
    if "umap_x" not in df.columns:
        return {"points": []}

    mask = pd.Series([True] * len(df), index=df.index)
    if position != "all":
        mask &= df["position_primary"] == position.upper()
    if league != "all":
        mask &= df["league"] == league

    df2 = df[mask]
    points = []
    for _, row in df2.iterrows():
        points.append({
            "player":      str(row.get("player", "")),
            "team":        str(row.get("team", "")),
            "style_label": str(row.get("style_label", "")),
            "position":    str(row.get("position_primary", "")),
            "umap_x":      float(row.get("umap_x", 0)),
            "umap_y":      float(row.get("umap_y", 0)),
            "psi":         float(row.get("psi", 0)),
            "market_value": float(row.get("market_value", 0)),
        })
    return {"points": points}


# ---------------------------------------------------------------------------
# Feature 23 — Squad Strength Heatmap
# ---------------------------------------------------------------------------

@app.get("/api/teams", tags=["core"], summary="All team names (optionally filtered by league)")
async def api_teams(league: str = "all"):
    """Return sorted list of all teams (optionally filtered by league)."""
    df = load_players()
    if league != "all":
        df = df[df["league"] == league]
    teams = sorted(df["team"].dropna().unique().tolist())
    return {"teams": teams}


@app.get("/api/seasons", tags=["players"], summary="Per-season stats for a player (all seasons, no dedup)")
async def api_seasons(player: str, team: str = ""):
    """
    Return per-season stats for a player (all seasons, no dedup).
    Feature 24 — Season Progression Chart.
    """
    loop   = asyncio.get_event_loop()
    data   = await loop.run_in_executor(_executor, lambda: get_player_seasons(player, team))
    if not data:
        raise HTTPException(status_code=404, detail=f"No season data found for '{player}'")
    return {"player": player, "team": team, "seasons": data, "count": len(data)}


@app.get("/api/squad", tags=["players"], summary="Full squad for a team, sorted by position then PSI")
async def api_squad(team: str):
    """
    Return all players for a given team, serialised via player_to_dict.
    Players are sorted by position_primary then PSI descending.
    """
    df = load_players()
    team_df = df[df["team"] == team].copy()
    if team_df.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team}' not found")

    # Sort: position order GK→DF→MF→FW, then PSI desc within each group
    pos_order = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}
    team_df["_pos_ord"] = team_df["position_primary"].map(pos_order).fillna(9)
    team_df = team_df.sort_values(["_pos_ord", "psi"], ascending=[True, False])

    players = [player_to_dict(row) for _, row in team_df.iterrows()]
    league  = str(team_df["league"].iloc[0]) if not team_df.empty else "EPL"
    return {"players": players, "team": team, "league": league, "count": len(players)}


# ---------------------------------------------------------------------------
# Feature 26 — Transfer Window Simulator
# ---------------------------------------------------------------------------

@app.post(
    "/api/transfer-window",
    tags=["transfer-window"],
    summary="Build a structured transfer window plan",
    description="""
Returns targets (primary + 2 alternatives per position), fee negotiation bands
(opening bid → expected → walk-away max), urgency-ordered signing sequence,
sell recommendations to fund the window, and a ready-to-stream LLM prompt.

Also includes a `template_narrative` field for no-LLM fallback.
""",
)
async def api_transfer_window(body: dict = Body(...)):
    """
    Build a structured transfer window plan.

    Body params
    -----------
    team        str   — club name (must match dataset)
    budget      float — direct budget in €M
    formation   str   — e.g. "433"
    needed      list  — [{"slug": "ST", "mode": "psi"}, ...]
    extra_context str  — optional director's notes for the LLM prompt
    """
    team          = str(body.get("team", ""))
    budget        = float(body.get("budget", 50))
    formation     = str(body.get("formation", "433"))
    needed        = body.get("needed", [])
    extra_context = str(body.get("extra_context", ""))

    if not team:
        raise HTTPException(status_code=400, detail="team is required")
    if not needed:
        raise HTTPException(status_code=400, detail="needed positions list is required")

    df   = load_players()
    loop = asyncio.get_event_loop()
    all_players = await loop.run_in_executor(
        _executor, lambda: [player_to_dict(r) for _, r in df.iterrows()]
    )

    result = await loop.run_in_executor(
        _executor,
        lambda: plan_transfer_window(
            players       = all_players,
            team          = team,
            budget        = budget,
            formation     = formation,
            needed        = needed,
            extra_context = extra_context,
        ),
    )

    logger.info(
        "Transfer window: team=%s budget=€%sM formation=%s positions=%s targets=%d sells=%d",
        team, budget, formation,
        [n.get("slug") for n in needed],
        len(result.get("targets", [])),
        len(result.get("sells", [])),
    )
    return result


@app.get("/api/transfer-window/meta", tags=["transfer-window"], summary="Position slugs, formations and scoring modes for the planner UI")
async def api_transfer_window_meta():
    """Return position slugs and formation options for the Transfer Window UI."""
    return {
        "positions": [
            {"slug": slug, "label": meta["label"], "icon": meta["icon"], "code": meta["code"]}
            for slug, meta in POSITION_META.items()
        ],
        "formations": ["433", "442", "4231", "352", "532", "343"],
        "modes": [
            {"key": "psi",    "label": "Best Performance",  "desc": "Highest PSI — proven quality"},
            {"key": "value",  "label": "Best Value",        "desc": "PSI per €M — budget efficiency"},
            {"key": "future", "label": "Future Upside",     "desc": "Players with biggest value growth"},
        ],
    }


# ---------------------------------------------------------------------------
# Feature 27 — Hidden Gems / Bargain Board
# ---------------------------------------------------------------------------

@app.get(
    "/api/gems",
    tags=["gems"],
    summary="Hidden gems — undervalued players ranked by value gap",
    description="`sort=gap` (absolute €M upside), `sort=ratio` (fv/mv multiplier), `sort=efficiency` (PSI/€M). Filters: position, league, max_age, min_minutes, max_mv, min_gap.",
)
async def api_gems(
    sort:        str   = "gap",    # gap | ratio | efficiency
    position:    str   = "all",    # all | GK | DF | MF | FW
    league:      str   = "all",
    max_age:     int   = 28,
    min_minutes: int   = 500,
    max_mv:      float = 9999,     # market value cap (€M); 9999 = no cap
    min_gap:     float = 0,        # minimum value_gap (€M)
    limit:       int   = 50,
):
    """
    Return the top undervalued players sorted by chosen metric.

    sort=gap        → future_value − market_value  (biggest absolute upside)
    sort=ratio      → future_value / market_value  (best proportional deal)
    sort=efficiency → PSI / market_value × 100     (performance per €M)
    """
    df   = load_players()
    loop = asyncio.get_event_loop()

    def _compute():
        players = [player_to_dict(r) for _, r in df.iterrows()]

        # Apply filters
        filtered = []
        for p in players:
            if position != "all" and p.get("position") != position.upper():
                continue
            if league != "all" and p.get("league") != league:
                continue
            if p.get("age", 99) > max_age:
                continue
            if p.get("minutes", 0) < min_minutes:
                continue
            mv = p.get("market_value", 0) or 0
            fv = p.get("future_value",  0) or 0
            if mv <= 0 or fv <= 0:
                continue
            if mv > max_mv:
                continue
            gap = fv - mv
            if gap < min_gap:
                continue

            # Derived metrics
            ratio      = round(fv / mv, 2) if mv > 0 else 0
            efficiency = round((p.get("psi", 0) or 0) * 100 / mv, 3) if mv > 0 else 0

            p["_gap"]        = round(gap, 1)
            p["_ratio"]      = ratio
            p["_efficiency"] = efficiency
            filtered.append(p)

        # Sort
        if sort == "ratio":
            filtered.sort(key=lambda p: p["_ratio"],      reverse=True)
        elif sort == "efficiency":
            filtered.sort(key=lambda p: p["_efficiency"], reverse=True)
        else:
            filtered.sort(key=lambda p: p["_gap"],        reverse=True)

        # Attach rank
        results = []
        for rank, p in enumerate(filtered[:limit], 1):
            results.append({
                **p,
                "rank":       rank,
                "gem_gap":        p["_gap"],
                "gem_ratio":      p["_ratio"],
                "gem_efficiency": p["_efficiency"],
            })
        return results

    results = await loop.run_in_executor(_executor, _compute)

    logger.info(
        "Gems: sort=%s pos=%s league=%s max_age=%d max_mv=%.0f min_gap=%.0f → %d results",
        sort, position, league, max_age, max_mv, min_gap, len(results),
    )
    return {
        "gems":  results,
        "count": len(results),
        "sort":  sort,
        "filters": {
            "position":    position,
            "league":      league,
            "max_age":     max_age,
            "min_minutes": min_minutes,
            "max_mv":      max_mv,
            "min_gap":     min_gap,
        },
    }


# ---------------------------------------------------------------------------
# Feature 28 — Player DNA Matching
# ---------------------------------------------------------------------------

@app.post(
    "/api/dna",
    tags=["dna"],
    summary="Build team DNA vector and find stylistic matches",
    description="""
Computes a PSI-weighted centroid of the team's squad embeddings (L2-normalised),
then cosine-ranks all external players.

**DNA tiers** (cosine sim × 100): Excellent ≥ 82 · Strong ≥ 72 · Decent ≥ 62 · Modest < 62
""",
)
async def api_dna(body: dict = Body(...)):
    """
    Build a team's DNA vector from their squad embeddings, then rank
    external players by cosine similarity.

    Body params
    -----------
    team          str   — club name (must match dataset)
    squad_filter  str   — "all" | "starters" (top 11 by PSI)
    position      str   — "all" | "GK" | "DF" | "MF" | "FW"
    league        str   — "all" | "EPL" | …
    max_age       int   — default 35
    max_price     float — default 9999
    min_minutes   int   — default 500
    arc_phase     str   — "all" | "pre-peak" | "peak" | "post-peak"
    limit         int   — default 40
    """
    team         = str(body.get("team", ""))
    squad_filter = str(body.get("squad_filter", "all"))
    position     = str(body.get("position",    "all"))
    league       = str(body.get("league",      "all"))
    max_age      = int(body.get("max_age",     35))
    max_price    = float(body.get("max_price", 9999))
    min_minutes  = int(body.get("min_minutes", 500))
    arc_phase    = str(body.get("arc_phase",   "all"))
    limit        = min(int(body.get("limit",   40)), 80)

    if not team:
        raise HTTPException(status_code=400, detail="team is required")

    df   = load_players()
    loop = asyncio.get_event_loop()

    from embeddings import build_or_load_embeddings

    def _run():
        matrix = build_or_load_embeddings(df)

        dna_result = build_team_dna(
            df, matrix, team,
            squad_filter = squad_filter,
            psi_weighted = True,
        )
        if dna_result is None:
            return None

        matches = find_dna_matches(
            df, matrix,
            dna_vec      = dna_result["dna_vec"],
            exclude_team = team,
            position     = position,
            league       = league,
            max_age      = max_age,
            max_price    = max_price,
            min_minutes  = min_minutes,
            arc_phase    = arc_phase,
            limit        = limit,
        )
        return {
            "profile": dna_result["profile"],
            "squad":   dna_result["squad"],
            "matches": matches,
            "count":   len(matches),
        }

    result = await loop.run_in_executor(_executor, _run)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Team '{team}' not found in dataset")

    logger.info(
        "DNA: team=%s filter=%s pos=%s league=%s → %d matches",
        team, squad_filter, position, league, result["count"],
    )
    return result


# ---------------------------------------------------------------------------
# Feature 29 — Cheaper Alternatives
# ---------------------------------------------------------------------------

@app.get("/api/player-names", tags=["cheaper"], summary="Player name autocomplete (substring match)")
async def api_player_names(q: str = "", limit: int = 12):
    """
    Autocomplete endpoint — returns matching players by substring on name.
    Returns compact list: [{player, team, position, market_value, age}]
    Sorted by PSI descending so the most prominent players appear first.
    """
    df   = load_players()
    q    = q.strip()

    if not q:
        return {"players": []}

    mask = df["player"].str.contains(q, case=False, na=False)
    hits = df[mask].sort_values("psi", ascending=False).head(limit)

    return {
        "players": [
            {
                "player":       str(r.get("player", "")),
                "team":         str(r.get("team", "")),
                "position":     str(r.get("position_primary", "")),
                "market_value": round(float(r.get("market_value", 0) or 0), 1),
                "age":          int(r.get("age", 0)),
                "league":       str(r.get("league", "")),
            }
            for _, r in hits.iterrows()
        ]
    }


@app.get(
    "/api/cheaper",
    tags=["cheaper"],
    summary="Find cosine-similar players within a price ratio",
    description="Returns same-position players with embedding similarity ≥ `min_sim` and market_value ≤ `ratio × target_value`. Each result includes `sim_pct`, `price_pct` of target, and `saving_pct`.",
)
async def api_cheaper(
    player:   str,
    team:     str   = "",
    ratio:    float = 0.60,   # max price as fraction of target (0.60 = 60%)
    min_sim:  float = 0.65,   # minimum cosine similarity
    n:        int   = 5,      # results to return
):
    """
    Find the n most similar players to the target who cost ≤ ratio × target_mv.

    Each result includes:
        sim_score  float  cosine similarity (0-1)
        sim_pct    float  × 100
        price_pct  float  market_value / target_mv × 100
        saving_pct float  100 − price_pct  (how much cheaper)
    """
    from embeddings import build_or_load_embeddings
    import numpy as np

    df   = load_players()
    loop = asyncio.get_event_loop()

    def _run():
        matrix = build_or_load_embeddings(df)

        # Resolve target player
        mask = df["player"] == player
        if team:
            mask &= df["team"] == team
        if not mask.any():
            # Fallback: case-insensitive contains
            mask = df["player"].str.contains(player, case=False, na=False)
            if team:
                mask &= df["team"].str.contains(team, case=False, na=False)
        if not mask.any():
            return None

        target_row = df[mask].sort_values("minutes", ascending=False).iloc[0]
        target_idx = int(target_row["_idx"])
        target_mv  = float(target_row.get("market_value", 0) or 0)
        target_pos = str(target_row.get("position_primary", "MF"))
        target_team= str(target_row.get("team", ""))

        anchor = matrix[target_idx]
        sims   = (matrix @ anchor).astype(np.float64)

        max_price = target_mv * ratio if target_mv > 0 else 9999

        results = []
        for i in np.argsort(sims)[::-1]:
            r = df.iloc[i]
            if int(i) == target_idx:
                continue
            if str(r.get("position_primary", "")) != target_pos:
                continue
            if str(r.get("team", "")) == target_team:
                continue
            if float(r.get("minutes", 0)) < 500:
                continue

            rmv = float(r.get("market_value", 0) or 0)
            if rmv <= 0:
                continue
            if target_mv > 0 and rmv > max_price:
                continue

            sim = float(sims[i])
            if sim < min_sim:
                break   # sorted descending — nothing below will qualify

            p = player_to_dict(r)
            p["sim_score"]  = round(sim, 4)
            p["sim_pct"]    = round(sim * 100, 1)
            p["price_pct"]  = round(rmv / target_mv * 100, 1) if target_mv > 0 else None
            p["saving_pct"] = round(100 - rmv / target_mv * 100, 1) if target_mv > 0 else None
            results.append(p)

            if len(results) >= n:
                break

        return {
            "target": player_to_dict(target_row),
            "alts":   results,
            "count":  len(results),
            "ratio":  ratio,
            "max_price": round(max_price, 1),
        }

    result = await loop.run_in_executor(_executor, _run)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Player '{player}' not found")

    logger.info(
        "Cheaper alts: %s → %d results (ratio=%.0f%% max=€%.0fM)",
        player, result["count"], ratio * 100, result["max_price"],
    )
    return result


# ---------------------------------------------------------------------------
# Feature 32 — Scouting Report PDF
# ---------------------------------------------------------------------------

@app.get(
    "/api/scout-report",
    tags=["scout-report"],
    summary="Full structured data for a one-page scouting report PDF",
    description="""
Returns all data needed to render a professional scouting report:
player stats, radar chart values, valuation model outputs (fee bands,
confidence interval), style/arc context, and a pre-built LLM prompt string.

The LLM narrative itself must be streamed separately via `POST /api/report`
(SSE) — this endpoint is intentionally synchronous for easy client caching.
""",
)
async def api_scout_report(player: str, team: str = ""):
    """
    Return enriched player record for PDF generation.

    Includes everything from player_to_dict plus:
      fee_model   dict  — opening_bid / expected_fee / walk_away (from transfer_window logic)
      report_prompt str  — ready-to-send scout prompt for the LLM
    """
    from transfer_window import compute_fees
    from report import build_scout_prompt

    df   = load_players()
    mask = df["player"].str.lower() == player.lower()
    if team:
        mask &= df["team"].str.lower() == team.lower()
    if not mask.any():
        # relax to substring
        mask = df["player"].str.lower().str.contains(player.lower(), na=False)
        if team:
            mask &= df["team"].str.lower() == team.lower()
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Player '{player}' not found")

    row  = df[mask].sort_values("psi", ascending=False).iloc[0]
    loop = asyncio.get_event_loop()

    data = await loop.run_in_executor(
        _executor, lambda: player_to_dict(row)
    )

    # Fee negotiation model (reuse transfer_window logic)
    fee_model = compute_fees(data)
    data["fee_model"] = fee_model

    # Pre-built scout prompt (so clients can pass straight to their own LLM)
    try:
        data["report_prompt"] = build_scout_prompt(data)
    except Exception:
        data["report_prompt"] = ""

    logger.info("Scout report data: %s (%s)", data["player"], data["team"])
    return data
