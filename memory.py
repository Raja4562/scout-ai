"""
ScoutAI - Session Memory  (Feature 17: Conversational Preference Stacking)
===========================================================================
Extracts and stores user preferences across conversation turns without a
database. Session state lives in a Python dict keyed by a browser UUID.

Two-tier preference model
--------------------------
1. Structured filters  (active_filters)
   Hard constraints that filter the result set:
     position / max_age / min_age / max_price / nationality / league /
     arc_phase / min_minutes / max_contract_years / free_agent / style

   New values override old values for the same key.
   Old values for *other* keys are preserved automatically.

2. Text qualifiers  (text_qualifiers)
   Free-text phrases like "left footed", "physically dominant",
   "comfortable under pressure" that cannot be expressed as structured
   filters.  They are injected into the semantic search query so the
   embedding model handles them naturally.

   These accumulate across turns (up to MAX_QUALIFIERS).

High-level memory (unchanged from before)
-----------------------------------------
  team / budget / max_price / formation / shortlist / search_hist / notes

Usage in api_search
-------------------
  1. Call detect_reset(query)  → if True, clear active_filters + qualifiers
  2. Call detect_refinement(query, mem) → bool
  3. Call extract_memory(query, mem) → (mem, signals)   [team/budget/formation]
  4. Call stack_search_context(query, mem, parsed_filters)
        → (augmented_query, updated_mem, change_signals)
  5. Pass augmented_query to search_players
  6. Pass mem["active_filters"] as base filters (explicit UI params override)
"""

import re
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("scoutai.memory")

# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------

EPL_TEAMS = {
    "arsenal", "chelsea", "liverpool", "manchester city", "man city",
    "manchester united", "man united", "man utd", "tottenham", "spurs",
    "newcastle", "aston villa", "west ham", "brighton", "brentford",
    "fulham", "crystal palace", "everton", "nottingham forest",
    "wolves", "wolverhampton", "bournemouth", "luton",
    "sheffield united", "burnley", "leicester", "ipswich", "southampton",
}

_TEAM_NORMALISE = {
    "man city":          "Manchester City",
    "manchester city":   "Manchester City",
    "man united":        "Manchester United",
    "man utd":           "Manchester United",
    "manchester united": "Manchester United",
    "spurs":             "Tottenham",
    "tottenham":         "Tottenham",
    "wolves":            "Wolverhampton",
    "wolverhampton":     "Wolverhampton",
    "newcastle":         "Newcastle",
    "brighton":          "Brighton",
    "brentford":         "Brentford",
    "fulham":            "Fulham",
    "chelsea":           "Chelsea",
    "arsenal":           "Arsenal",
    "liverpool":         "Liverpool",
    "everton":           "Everton",
    "bournemouth":       "Bournemouth",
    "west ham":          "West Ham",
    "aston villa":       "Aston Villa",
    "crystal palace":    "Crystal Palace",
    "nottingham forest": "Nottingham Forest",
    "luton":             "Luton",
    "sheffield united":  "Sheffield United",
    "burnley":           "Burnley",
    "leicester":         "Leicester",
    "ipswich":           "Ipswich",
    "southampton":       "Southampton",
}

FORMATIONS = [
    "4-3-3", "4-2-3-1", "4-4-2", "3-5-2", "3-4-3",
    "4-5-1", "5-3-2", "4-1-4-1", "4-3-2-1", "4-2-2-2", "3-4-2-1",
]

_TEAM_CONTEXT = re.compile(
    r"(?:for|at|sign for|playing for|our|we(?:'re| are| play| need| want)|"
    r"i(?:'m| am) (?:at|with)|scout(?:ing)? for|manager of|coach(?:ing)? at|"
    r"transfers? for|buy for|need for)\s+"
)

# Max free-text qualifiers to keep in the stack before evicting oldest
MAX_QUALIFIERS = 6

# ---------------------------------------------------------------------------
# Reset / refinement classification
# ---------------------------------------------------------------------------

_RESET_PHRASES = [
    "start over", "start fresh", "new search", "reset", "clear filters",
    "forget that", "forget it", "never mind", "different type",
    "completely different", "ignore the previous", "scratch that",
    "start again", "begin again", "wipe that",
]

_REFINEMENT_STARTERS = [
    "make him", "make her", "make them", "make it",
    "but also", "but ", "also ", "and also",
    "add ", "plus ", "now also", "additionally",
    "actually", "wait ", "no wait", "instead ",
    "refine", "narrow", "filter more",
    "same but", "similar but", "like that but",
    "with ", "who is ", "who are ",
]

# Physical / attribute phrases that are clearly refinements, not new queries
_ATTRIBUTE_PHRASES = re.compile(
    r"\b(?:left.?foot(?:ed)?|right.?foot(?:ed)?|two.?foot(?:ed)?|"
    r"taller?|shorter?|faster?|quicker?|stronger?|powerful|pacey|"
    r"technically gifted|good dribbler?|aerial threat|"
    r"dominant in the air|good passer|long.?range|"
    r"experienced|clinical|creative|combative|press.?resistant)\b",
    re.IGNORECASE,
)


def detect_reset(query: str) -> bool:
    """Return True if the query is a clear-context signal."""
    q = query.lower().strip()
    return any(phrase in q for phrase in _RESET_PHRASES)


def detect_refinement(query: str, memory: dict) -> bool:
    """
    Return True if this query is refining the previous search rather than
    starting a new one.

    A query is a refinement when:
      - Active filters already exist in memory (there IS a previous context), AND
      - The query starts with a refinement phrase, OR
      - The query is short AND contains only attribute/constraint words
        (no position keyword that would indicate a new search intent).
    """
    if not memory.get("active_filters") and not memory.get("text_qualifiers"):
        return False   # nothing to refine

    q = query.lower().strip()

    # Explicit refinement starters
    for phrase in _REFINEMENT_STARTERS:
        if q.startswith(phrase) or f" {phrase}" in q:
            return True

    # Pure attribute phrases with no position keyword → refinement
    from config import POSITION_KEYWORDS
    has_position = any(kw in f" {q} " for kw in POSITION_KEYWORDS)
    has_attribute = bool(_ATTRIBUTE_PHRASES.search(q))
    is_short = len(q.split()) <= 6

    if has_attribute and not has_position:
        return True
    if is_short and not has_position and memory.get("active_filters", {}).get("position"):
        return True

    return False


# ---------------------------------------------------------------------------
# Empty memory structure
# ---------------------------------------------------------------------------

def empty_memory() -> dict:
    return {
        # High-level session context (existing)
        "team":            None,
        "budget":          None,
        "max_price":       None,
        "formation":       None,
        "shortlist":       [],
        "search_hist":     [],
        "notes":           [],
        "updated_at":      None,

        # Feature 17: preference stack
        "active_filters":  {},     # accumulated structured filter constraints
        "text_qualifiers": [],     # accumulated free-text search qualifiers
        "base_query":      None,   # last "full intent" query (for refinement context)
        "is_refinement":   False,  # whether the last query was a refinement
    }


# ---------------------------------------------------------------------------
# Free-text qualifier extraction
# ---------------------------------------------------------------------------

# Map of phrase patterns → canonical qualifier text to store
_QUALIFIER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bleft.?foot(?:ed)?\b",     re.I), "left footed"),
    (re.compile(r"\bright.?foot(?:ed)?\b",    re.I), "right footed"),
    (re.compile(r"\btwo.?foot(?:ed)?\b",      re.I), "two footed"),
    (re.compile(r"\bpac(?:e|ey|y)\b|\bfast\b|\bquick\b|\belectric\b", re.I), "pace and acceleration"),
    (re.compile(r"\baerial threat\b|\baerial duel\b|\bwins headers\b|\bgood in the air\b", re.I), "aerial threat wins headers"),
    (re.compile(r"\bphysically (?:strong|dominant|imposing)\b|\bbig and strong\b", re.I), "physically strong and imposing"),
    (re.compile(r"\btechnic(?:al|ally) gifted\b|\bgreat technique\b|\bskillful\b", re.I), "technically gifted skillful"),
    (re.compile(r"\bgood (?:dribbl(?:er|ing)|on the ball)\b|\bbeats (?:players|defenders)\b", re.I), "dribbler beats defenders"),
    (re.compile(r"\blong.?range (?:shot|shooting|effort)\b", re.I), "long range shooting"),
    (re.compile(r"\bpress.?resistant\b|\bgood under pressure\b|\bcomposure\b", re.I), "press resistant composure"),
    (re.compile(r"\bclinical\b|\befficient in front of goal\b|\bgoal.?scorer\b", re.I), "clinical goalscorer"),
    (re.compile(r"\bcreative\b|\bchance creator\b|\bkey pass\b", re.I), "creative chance creator"),
    (re.compile(r"\bcombative\b|\baggressive\b|\bbattle\b", re.I), "combative aggressive midfield"),
    (re.compile(r"\bleadership\b|\bcaptain\b|\bcommanding\b", re.I), "leader commanding presence"),
    (re.compile(r"\bset.?piece\b|\bdelivery\b|\bdead.?ball\b", re.I), "set piece specialist"),
    (re.compile(r"\bexperienced\b|\bveteran pres\b|\bproven\b", re.I), "experienced proven"),
]


def _extract_qualifiers(query: str) -> list[str]:
    """Extract free-text qualifiers from query text."""
    found = []
    for pattern, canonical in _QUALIFIER_PATTERNS:
        if pattern.search(query):
            found.append(canonical)
    return found


# ---------------------------------------------------------------------------
# Filter stacking
# ---------------------------------------------------------------------------

# Human-readable labels for filter chips in the UI
_FILTER_LABELS = {
    "position":          lambda v: {"FW": "⚽ Forward", "MF": "🔄 Midfielder",
                                    "DF": "🛡 Defender",  "GK": "🧤 Goalkeeper"}.get(v, v),
    "max_age":           lambda v: f"📅 ≤{v} yrs",
    "min_age":           lambda v: f"📅 ≥{v} yrs",
    "max_price":         lambda v: f"💶 ≤{v:.0f}M",
    "nationality":       lambda v: f"🌍 {v}",
    "league":            lambda v: f"🏟 {v}",
    "arc_phase":         lambda v: {"pre-peak": "📈 Pre-peak", "peak": "⭐ Prime", "post-peak": "📉 Post-peak"}.get(v, v),
    "min_minutes":       lambda v: f"⏱ ≥{v} mins",
    "max_contract_years":lambda v: f"📋 ≤{v}yr contract",
    "free_agent":        lambda v: "📋 Free agent",
    "expiring":          lambda v: "📋 Expiring contract",
    "style":             lambda v: f"◈ {v}",
}

# Keys to carry across in the preference stack (exclude transient things)
_STACKABLE_KEYS = {
    "position", "max_age", "min_age", "max_price",
    "nationality", "league", "arc_phase", "min_minutes",
    "max_contract_years", "free_agent", "expiring", "style", "stat_boosts",
}


def stack_filters(
    active: dict,
    new_filters: dict,
) -> tuple[dict, list[str]]:
    """
    Merge new_filters onto the active filter stack.
    New values override old for the same key.
    Returns (merged_filters, list_of_human_readable_change_labels).
    """
    merged  = {**active}
    changes: list[str] = []

    for k, v in new_filters.items():
        if k not in _STACKABLE_KEYS:
            continue
        old = merged.get(k)
        if k == "stat_boosts":
            # Accumulate boost weights rather than replacing
            old_boosts = merged.get("stat_boosts", {})
            new_boosts = {**old_boosts}
            for col, w in v.items():
                new_boosts[col] = new_boosts.get(col, 0.0) + w
            merged["stat_boosts"] = new_boosts
            continue
        if old != v:
            merged[k] = v
            label_fn = _FILTER_LABELS.get(k)
            if label_fn:
                try:
                    changes.append(label_fn(v))
                except Exception:
                    changes.append(f"{k}: {v}")

    return merged, changes


def active_filters_to_labels(active_filters: dict) -> dict:
    """
    Convert active_filters dict → {key: human_readable_label} for the
    frontend context bar.  Each entry can have its key deleted individually.
    """
    labels = {}
    for k, v in active_filters.items():
        if k in _FILTER_LABELS:
            try:
                labels[k] = _FILTER_LABELS[k](v)
            except Exception:
                labels[k] = f"{k}: {v}"
    return labels


# ---------------------------------------------------------------------------
# Augmented query builder
# ---------------------------------------------------------------------------

def build_augmented_query(
    current_query: str,
    memory: dict,
    is_refine: bool,
) -> str:
    """
    Build the query string that will be sent to the embedding model.

    For a refinement:
      base_query + text_qualifiers + current constraint words
      e.g. "striker" + "left footed" + "under 25M" → "left footed striker under 25M"

    For a new query:
      current_query + accumulated text_qualifiers
      (qualifiers from previous turns enrich new searches in the same session)
    """
    qualifiers = memory.get("text_qualifiers", [])
    base       = memory.get("base_query") or ""

    if is_refine and base:
        # Build: qualifiers + base position + current refinement text
        parts = []
        if qualifiers:
            parts.extend(qualifiers)
        parts.append(base)
        # Append only new words not already in the combined text (avoid repetition)
        combined = " ".join(parts).lower()
        for word in current_query.split():
            if word.lower() not in combined:
                parts.append(word)
        return " ".join(parts)
    else:
        # New intent query — keep as-is but append qualifiers if they add signal
        if not qualifiers:
            return current_query
        qual_str = " ".join(qualifiers)
        # Avoid duplication if query already contains the qualifier content
        q_lower = current_query.lower()
        new_quals = [q for q in qualifiers if q.split()[0] not in q_lower]
        if new_quals:
            return current_query + " " + " ".join(new_quals)
        return current_query


# ---------------------------------------------------------------------------
# Main context stacking entry point
# ---------------------------------------------------------------------------

def stack_search_context(
    query: str,
    memory: dict,
    parsed_filters: dict,
    is_refine: bool,
) -> tuple[str, dict, list[str]]:
    """
    Called after parse_filters() on each query.
    Returns (augmented_query, updated_memory, change_signals_for_UI).

    change_signals list items that are NEW additions get a "+" prefix.
    Items that were already present (unchanged) get no prefix.
    """
    mem = {**memory,
           "active_filters": dict(memory.get("active_filters", {})),
           "text_qualifiers": list(memory.get("text_qualifiers", []))}

    # Stack structured filters
    old_active = dict(mem["active_filters"])
    merged, changes = stack_filters(old_active, parsed_filters)
    mem["active_filters"] = merged

    # Extract and stack text qualifiers
    new_quals = _extract_qualifiers(query)
    existing  = set(mem["text_qualifiers"])
    added_quals = []
    for q in new_quals:
        if q not in existing:
            mem["text_qualifiers"].append(q)
            added_quals.append(q)
    # Evict oldest if over limit
    if len(mem["text_qualifiers"]) > MAX_QUALIFIERS:
        mem["text_qualifiers"] = mem["text_qualifiers"][-MAX_QUALIFIERS:]

    # Update base query (only for non-refinement, non-reset queries)
    if not is_refine and not detect_reset(query):
        mem["base_query"] = query

    mem["is_refinement"] = is_refine
    mem["updated_at"]    = datetime.now().isoformat()

    # Build augmented query for the embedding search
    aug_query = build_augmented_query(query, mem, is_refine)

    # Build human-readable signals for the UI context bar
    signals = []
    for c in changes:
        signals.append({"label": c, "new": True})
    for q in added_quals:
        signals.append({"label": f"✦ {q}", "new": True})

    return aug_query, mem, signals


# ---------------------------------------------------------------------------
# High-level memory extraction (team / budget / formation)
# ---------------------------------------------------------------------------

def extract_memory(query: str, memory: dict) -> tuple[dict, list[str]]:
    """
    Parse a query for high-level memory signals (team, budget, formation).
    Returns (updated_memory_dict, list_of_human_readable_signals).
    Signals are displayed as toast notifications.
    Does not mutate the input dict.
    """
    q   = query.lower().strip()
    mem = {**memory}
    signals: list[str] = []

    # -- Team ----------------------------------------------------------------
    for team_kw in sorted(EPL_TEAMS, key=len, reverse=True):
        if team_kw in q:
            start  = q.find(team_kw)
            window = q[max(0, start - 40): start + len(team_kw)]
            if _TEAM_CONTEXT.search(window) or any(
                phrase in q for phrase in [
                    f"for {team_kw}", f"at {team_kw}", f"our {team_kw}",
                    f"{team_kw} need", f"{team_kw} want",
                    f"{team_kw} require", f"{team_kw} should sign",
                ]
            ):
                normalised = _TEAM_NORMALISE.get(team_kw, team_kw.title())
                if mem.get("team") != normalised:
                    mem["team"] = normalised
                    signals.append(f"Scouting for {normalised}")
            break

    # -- Total budget --------------------------------------------------------
    m = re.search(
        r"(?:budget|can spend|spending|window|transfer fund|overall)[^\d]{0,20}"
        r"(\d+(?:\.\d+)?)\s*[mM](?:illion)?", q,
    )
    if m:
        val = float(m.group(1))
        if mem.get("budget") != val:
            mem["budget"] = val
            signals.append(f"Budget {val:.0f}M")

    # -- Per-player price cap ------------------------------------------------
    m2 = re.search(
        r"(?:under|below|within|max(?:imum)?|less than|no more than|up to)"
        r"\s*[\$€£]?\s*(\d+(?:\.\d+)?)\s*[mM](?:illion)?", q,
    )
    if m2:
        val = float(m2.group(1))
        if mem.get("max_price") != val:
            mem["max_price"] = val
            signals.append(f"Per-player cap {val:.0f}M")

    # -- Formation -----------------------------------------------------------
    for fmt in FORMATIONS:
        if fmt in q or fmt.replace("-", " ") in q:
            if mem.get("formation") != fmt:
                mem["formation"] = fmt
                signals.append(f"Formation {fmt}")
            break

    # -- Search history (last 10) -------------------------------------------
    hist = list(mem.get("search_hist", []))
    if query not in hist:
        mem["search_hist"] = ([query] + hist)[:10]

    if signals:
        mem["updated_at"] = datetime.now().isoformat()

    return mem, signals


# ---------------------------------------------------------------------------
# Shortlist management
# ---------------------------------------------------------------------------

def add_to_shortlist(memory: dict, player: dict) -> tuple[dict, bool]:
    mem = {**memory}
    sl  = list(mem.get("shortlist", []))
    key = (player.get("player", ""), player.get("team", ""))
    if any((p.get("player", ""), p.get("team", "")) == key for p in sl):
        return mem, False
    sl.append({
        "player":       player.get("player", ""),
        "team":         player.get("team", ""),
        "position":     player.get("position", ""),
        "age":          player.get("age", 0),
        "market_value": player.get("market_value", 0),
        "future_value": player.get("future_value", 0),
        "psi":          player.get("psi", 0),
        "goals_per90":  player.get("goals_per90", 0),
        "assists_per90":player.get("assists_per90", 0),
        "xg_per90":     player.get("xg_per90", 0),
        "added_at":     datetime.now().isoformat(),
    })
    mem["shortlist"]  = sl
    mem["updated_at"] = datetime.now().isoformat()
    return mem, True


def remove_from_shortlist(memory: dict, player_name: str, team: str) -> dict:
    mem = {**memory}
    mem["shortlist"] = [
        p for p in mem.get("shortlist", [])
        if not (p.get("player") == player_name and p.get("team") == team)
    ]
    mem["updated_at"] = datetime.now().isoformat()
    return mem


# ---------------------------------------------------------------------------
# Context builders for search enrichment
# ---------------------------------------------------------------------------

def memory_to_context(memory: dict) -> str:
    """Short string injected into search summary."""
    parts = []
    if memory.get("team"):
        parts.append(f"scouting for {memory['team']}")
    if memory.get("formation"):
        parts.append(f"playing {memory['formation']}")
    if memory.get("budget"):
        parts.append(f"total budget {memory['budget']:.0f}M")
    if memory.get("max_price"):
        parts.append(f"per-player cap {memory['max_price']:.0f}M")
    return ", ".join(parts)


def memory_applies_filter(memory: dict) -> dict:
    """
    Return a filters dict derived from accumulated memory.
    Serves as the base that explicit UI params can override.
    """
    f: dict = {}
    # High-level price cap
    if memory.get("max_price"):
        f["max_price"] = memory["max_price"]
    # Full preference stack
    for k, v in memory.get("active_filters", {}).items():
        if k not in f:   # don't overwrite explicit signals above
            f[k] = v
    return f


def clear_active_filters(memory: dict) -> dict:
    """Wipe the preference stack (but keep team/budget/shortlist)."""
    mem = {**memory}
    mem["active_filters"]  = {}
    mem["text_qualifiers"] = []
    mem["base_query"]      = None
    mem["is_refinement"]   = False
    mem["updated_at"]      = datetime.now().isoformat()
    return mem


def remove_active_filter(memory: dict, key: str) -> dict:
    """Remove a single key from the active preference stack."""
    mem = {**memory,
           "active_filters": dict(memory.get("active_filters", {}))}
    mem["active_filters"].pop(key, None)
    mem["updated_at"] = datetime.now().isoformat()
    return mem
