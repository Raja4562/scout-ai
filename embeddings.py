"""
ScoutAI - Player Embeddings  (Feature 13 — enriched)
=====================================================
Builds and caches a 384-dim embedding matrix for every player using
sentence-transformers/all-MiniLM-L6-v2.

Each player is converted to a rich natural-language scouting profile that:
  • Names the style archetype directly  (e.g. "Press-Resistant 8",
    "Inverted Winger") so queries like those retrieve the right players
  • Uses percentile-based language ("elite finisher", "top-10%") derived
    from the normalizer's _pct columns rather than raw numbers that vary
    by position and league
  • Works with multi-league data (gracefully skips columns that are zero)
  • Adds arc phase, injury risk, and contract context

Rebuilding is triggered automatically whenever the player CSV changes or
when force_rebuild=True is passed (e.g. after Feature 11 or 12 updates).
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from config import EMBED_FILE, EMBED_FILE_MULTI, EMBED_MODEL

logger = logging.getLogger("scoutai.embeddings")

_model   = None
_matrix  = None   # shape (N, 384), L2-normalised


# ---------------------------------------------------------------------------
# Style archetype elaborations
# When style_label is present we include both the label itself (for exact
# match) and a one-line elaboration (for fuzzy synonym queries).
# ---------------------------------------------------------------------------

STYLE_DESCRIPTIONS: dict[str, str] = {
    # Forwards
    "Poacher":            "clinical penalty-box striker, highly efficient finisher, scores with few shots",
    "Creative Forward":   "forward who creates chances for team-mates and scores goals, high assist output",
    "Pressing Forward":   "high-pressing forward with defensive work rate and counter-press ability",
    "Target Man":         "aerial threat in the forward line, hold-up play, wins headers, brings team-mates into play",
    "Inverted Winger":    "wide forward who cuts inside onto stronger foot, supports build-up, direct dribbler",
    # Midfielders
    "Ball Winner":        "combative defensive midfielder and ball winner, aggressive tackles in the middle third, wins possession back for the team, breaks up attacks in midfield",
    "Box-to-Box":         "energetic all-action midfielder covering ground, contributing goals and defensive pressing work",
    "Attacking Midfielder":"creative attacking midfielder, high goal and assist threat, plays between the lines in the pocket",
    "Playmaker":          "deep creative midfielder with range of passing, unlocks defences with key passes and assists",
    "Press-Resistant 8":  "ball-carrying midfielder who retains possession under pressure in midfield, presses high, strong passing range",
    "Holding Midfielder": "deep-lying defensive midfielder sitting in front of the back four, screens the defence, breaks up play, positionally disciplined",
    # Defenders
    "Attacking Fullback": "attacking full-back or wing-back, provides width and delivers crosses, overlapping runs, assist contributions",
    "Ball-Playing CB":    "centre-back comfortable in possession, range of passing from the back, progresses ball from deep",
    "Defensive Anchor":   "no-nonsense centre-back, dominant aerial duellist, physical and commanding in the penalty area",
    "Sweeper CB":         "sweeper centre-back, reads the game well, covers space behind the defensive line",
    "Stopper":            "commanding centre-back, wins aerial duels and headers, physically dominant in the penalty box",
    # Goalkeepers
    "Shot-Stopper":       "goalkeeper with exceptional reflexes and high save percentage",
    "Sweeper Keeper":     "sweeper keeper, sweeps behind defence, comfortable on the ball, good with feet",
    "Distribution GK":    "goalkeeper with strong long-ball distribution and aerial claiming ability",
}


# ---------------------------------------------------------------------------
# Percentile-to-qualifier helpers
# ---------------------------------------------------------------------------

def _q(pct: float,
       labels=("poor", "modest", "decent", "good", "very good", "excellent", "elite")) -> str:
    """Convert a 0-1 percentile into a human-readable qualifier."""
    thresholds = (0.10, 0.25, 0.40, 0.60, 0.75, 0.90)
    for i, t in enumerate(thresholds):
        if pct < t:
            return labels[i]
    return labels[-1]


def _top_pct_label(pct: float) -> str:
    """Return 'top X%' string, e.g. 0.92 → 'top 8%'."""
    val = max(1, round((1 - pct) * 100))
    return f"top {val}%"


def _stat_phrase(pct: float, stat_name: str, high_is_good: bool = True) -> str:
    """Build a phrase like 'elite shot accuracy (top 4%)'."""
    if pct <= 0.01:                # column absent / all zeros
        return ""
    if high_is_good:
        if pct >= 0.85:
            return f"elite {stat_name} ({_top_pct_label(pct)})"
        if pct >= 0.70:
            return f"excellent {stat_name}"
        if pct >= 0.50:
            return f"good {stat_name}"
        return ""                  # below average — don't advertise
    else:                          # low is good (e.g. goals-against)
        if pct >= 0.85:
            return f"elite {stat_name} ({_top_pct_label(pct)})"
        if pct >= 0.70:
            return f"excellent {stat_name}"
        return ""


def _nonzero(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        return float(row.get(col, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Per-position text builders
# ---------------------------------------------------------------------------

def _text_fw(row: pd.Series) -> str:
    g90      = _nonzero(row, "goals_per90")
    a90      = _nonzero(row, "assists_per90")
    sh90     = _nonzero(row, "sh_per90")
    g_pct    = _nonzero(row, "goals_per90_pct")
    a_pct    = _nonzero(row, "assists_per90_pct")
    sh_pct   = _nonzero(row, "sh_per90_pct")
    sot_pct  = _nonzero(row, "sot_per90_pct")
    gps_pct  = _nonzero(row, "g_per_shot_pct")   # finishing efficiency

    phrases = []

    # Scoring
    if g_pct >= 0.85:
        phrases.append(f"elite scorer ({_top_pct_label(g_pct)} for goals per 90, {g90:.2f} g/90)")
    elif g_pct >= 0.60:
        phrases.append(f"reliable goal scorer ({g90:.2f} goals per 90)")
    elif g90 > 0.05:
        phrases.append(f"contributes {g90:.2f} goals per 90")

    # Efficiency (clinical finishing)
    ef = _stat_phrase(gps_pct, "shot efficiency / clinical finishing")
    if ef:
        phrases.append(ef)

    # Shot volume
    sv = _stat_phrase(sh_pct, "shot volume")
    if sv:
        phrases.append(sv)

    # Shot accuracy
    sa = _stat_phrase(sot_pct, "shot accuracy")
    if sa and "shot volume" not in (phrases[-1] if phrases else ""):
        phrases.append(sa)

    # Creativity / assists
    if a_pct >= 0.80:
        phrases.append(f"creative forward with {_top_pct_label(a_pct)} assist rate ({a90:.2f} per 90)")
    elif a_pct >= 0.60:
        phrases.append(f"good chance creator ({a90:.2f} assists per 90)")
    elif a90 > 0.05:
        phrases.append(f"contributes {a90:.2f} assists per 90")

    # Pressing contribution (tackles as proxy)
    tkl_pct = _nonzero(row, "tackles_tkl_pct")
    if tkl_pct >= 0.75:
        phrases.append("high defensive work rate and pressing contribution")

    return "; ".join(phrases) + "." if phrases else f"Scores {g90:.2f} goals per 90."


def _text_mf(row: pd.Series) -> str:
    g90    = _nonzero(row, "goals_per90")
    a90    = _nonzero(row, "assists_per90")
    g_pct  = _nonzero(row, "goals_per90_pct")
    a_pct  = _nonzero(row, "assists_per90_pct")
    sh_pct = _nonzero(row, "sh_per90_pct")
    tkl_pct= _nonzero(row, "tackles_tkl_pct")
    int_pct= _nonzero(row, "int_pct")
    fd_pct = _nonzero(row, "fouls_drawn_pct")

    phrases = []

    # Attacking output
    if g_pct >= 0.80:
        phrases.append(f"goal-scoring midfielder, {_top_pct_label(g_pct)} for goals per 90 ({g90:.2f})")
    elif g_pct >= 0.60:
        phrases.append(f"goal threat from midfield ({g90:.2f} g/90)")

    if a_pct >= 0.80:
        phrases.append(f"creative playmaker, {_top_pct_label(a_pct)} for assists ({a90:.2f} per 90)")
    elif a_pct >= 0.60:
        phrases.append(f"good chance creator ({a90:.2f} assists per 90)")

    # Defensive contribution — use ball-winner / defensive midfielder terminology clearly
    if tkl_pct >= 0.80 and int_pct >= 0.70:
        phrases.append("elite midfielder ball winner, combative in tackles and interceptions in midfield, presses relentlessly to win possession")
    elif tkl_pct >= 0.75:
        phrases.append("midfield ball winner, aggressive tackles, wins possession in the press")
    elif int_pct >= 0.75:
        phrases.append("intelligent defensive midfielder, high interception numbers, reads the game in midfield")

    # Press resistance / fouls drawn
    if fd_pct >= 0.75:
        phrases.append("press-resistant, draws fouls and wins free kicks in dangerous areas")

    # Shot threat
    sv = _stat_phrase(sh_pct, "shot volume from midfield")
    if sv:
        phrases.append(sv)

    return "; ".join(phrases) + "." if phrases else f"Midfielder contributing {g90:.2f} g/90 and {a90:.2f} a/90."


def _text_df(row: pd.Series) -> str:
    g_pct   = _nonzero(row, "goals_per90_pct")
    a_pct   = _nonzero(row, "assists_per90_pct")
    tkl_pct = _nonzero(row, "tackles_tkl_pct")
    int_pct = _nonzero(row, "int_pct")
    fd_pct  = _nonzero(row, "fouls_drawn_pct")

    g90 = _nonzero(row, "goals_per90")
    a90 = _nonzero(row, "assists_per90")

    phrases = []

    # Defensive solidity — use centre-back / defender context, avoid "ball winner" phrasing
    if tkl_pct >= 0.80 and int_pct >= 0.75:
        phrases.append(f"dominant defensive centre-back — {_top_pct_label(tkl_pct)} tackles and {_top_pct_label(int_pct)} interceptions, commanding defensive presence")
    elif tkl_pct >= 0.75:
        phrases.append(f"aggressive-tackling centre-back ({_top_pct_label(tkl_pct)} for tackles, wins duels)")
    elif int_pct >= 0.75:
        phrases.append(f"intelligent-reading centre-back ({_top_pct_label(int_pct)} interceptions, positional awareness)")
    else:
        phrases.append("positionally disciplined defender, reliable in defensive shape")

    # Attacking contribution (fullback indicator)
    if a_pct >= 0.75:
        phrases.append(f"attacking fullback or wing-back with excellent assist rate ({a90:.2f} per 90)")
    elif a_pct >= 0.60:
        phrases.append(f"contributes offensively ({a90:.2f} assists per 90)")

    if g_pct >= 0.80:
        phrases.append(f"goal-scoring defender, {_top_pct_label(g_pct)} for goals")

    # Fouls / physicality
    if fd_pct >= 0.70:
        phrases.append("physically imposing, wins fouls and holds off opponents")

    return "; ".join(phrases) + "." if phrases else "Reliable defensive player."


def _text_gk(row: pd.Series) -> str:
    sp_pct  = _nonzero(row, "gk_savepct_pct")
    cs_pct  = _nonzero(row, "gk_cspct_pct")
    ga_pct  = _nonzero(row, "gk_ga90_pct")    # higher = fewer goals against = better
    sv_pct  = _nonzero(row, "gk_saves_pct")

    sp_raw  = _nonzero(row, "gk_savepct")
    cs_raw  = _nonzero(row, "gk_cspct")

    style_label = str(row.get("style_label", ""))
    phrases = []

    # Lead with sweeper keeper identity if applicable (helps semantic retrieval)
    if style_label == "Sweeper Keeper":
        phrases.append("sweeper goalkeeper who sweeps aggressively behind the defensive line, comfortable on the ball with feet, acts as an extra outfield player")
    elif style_label == "Distribution GK":
        phrases.append("distribution goalkeeper, launches attacks with long accurate passes from goalkeeper position")

    # GK stats stored as raw % values (74.1 = 74.1%, not 0.741)
    if sp_pct >= 0.75:
        phrases.append(f"excellent shot-stopper ({_top_pct_label(sp_pct)} save percentage, {sp_raw:.1f}% saves)")
    elif sp_raw > 0:
        phrases.append(f"save percentage {sp_raw:.1f}%")

    if cs_pct >= 0.75:
        phrases.append(f"strong clean-sheet record ({_top_pct_label(cs_pct)}, {cs_raw:.1f}% CS rate)")
    elif cs_raw > 0:
        phrases.append(f"clean-sheet rate {cs_raw:.1f}%")

    if ga_pct >= 0.80:
        phrases.append("very low goals conceded per 90, excellent shot-stopping")
    elif ga_pct >= 0.60:
        phrases.append("good goals-against record")

    if sv_pct >= 0.75:
        phrases.append("high save volume — plays behind a busy defence")

    return "; ".join(phrases) + "." if phrases else "Goalkeeper."


# ---------------------------------------------------------------------------
# Main text builder
# ---------------------------------------------------------------------------

def _player_to_text(row: pd.Series) -> str:
    """
    Convert a player row into a rich natural-language scouting profile.

    Structure:
      1. Identity (name, age, position, team, league)
      2. Style archetype + elaboration (Feature 11)
      3. Position-specific percentile-based performance description
      4. Career arc (Feature 10)
      5. Contract situation
      6. Injury risk flag (Feature 12, only when Elevated+)
      7. Value context
    """
    pos    = str(row.get("position_primary", "MF"))
    name   = str(row.get("player", "Unknown"))
    team   = str(row.get("team", "Unknown"))
    age    = int(row.get("age", 25))
    mv     = _nonzero(row, "market_value")
    psi    = _nonzero(row, "psi")
    mins   = int(row.get("minutes", 0))
    league = str(row.get("league", "EPL"))
    season = str(row.get("season", ""))

    league_label = {
        "EPL":        "the English Premier League",
        "La Liga":    "the Spanish La Liga",
        "Bundesliga": "the German Bundesliga",
        "Serie A":    "the Italian Serie A",
        "Ligue 1":    "the French Ligue 1",
    }.get(league, league)

    pos_label = {
        "GK": "goalkeeper",
        "DF": "defender",
        "MF": "midfielder",
        "FW": "forward",
    }.get(pos, "player")

    parts: list[str] = []

    # ── 1. Identity ────────────────────────────────────────────────────────
    parts.append(
        f"{name} is a {age}-year-old {pos_label} for {team} "
        f"in {league_label}, {season} season, {mins} minutes played."
    )

    # ── 2. Style archetype ─────────────────────────────────────────────────
    style_label = str(row.get("style_label", "")).strip()
    if style_label and style_label not in ("Unknown", "nan", ""):
        desc = STYLE_DESCRIPTIONS.get(style_label, "")
        if desc:
            parts.append(
                f"Playing archetype: {style_label}. {desc}."
            )
        else:
            parts.append(f"Playing archetype: {style_label}.")

    # ── 3. Performance description ─────────────────────────────────────────
    if pos == "GK":
        perf = _text_gk(row)
    elif pos == "DF":
        perf = _text_df(row)
    elif pos == "MF":
        perf = _text_mf(row)
    else:  # FW
        perf = _text_fw(row)
    if perf:
        parts.append(f"Performance: {perf}")

    # ── 4. PSI tier ────────────────────────────────────────────────────────
    if psi >= 0.80:
        parts.append("PSI: elite performer, top tier by composite metric.")
    elif psi >= 0.60:
        parts.append("PSI: strong performer above average for their position.")
    elif psi <= 0.25:
        parts.append("PSI: below-average performer at this level.")

    # ── 5. Career arc ──────────────────────────────────────────────────────
    arc_phase = str(row.get("arc_phase", "peak"))
    peak_age  = int(row.get("peak_age", 27))

    if arc_phase == "pre-peak":
        yrs = max(0, peak_age - age)
        if yrs >= 4:
            parts.append(
                f"Career arc: pre-peak, {yrs} years from prime — strong upside potential, "
                f"market likely underprices remaining development."
            )
        elif yrs >= 1:
            parts.append(
                f"Career arc: approaching peak years, {yrs} year(s) from prime window."
            )
    elif arc_phase == "post-peak":
        yrs_past = max(0, age - peak_age)
        parts.append(
            f"Career arc: post-peak, {yrs_past} year(s) past prime — "
            f"value declining, buy for immediate impact only."
        )
    else:
        parts.append("Career arc: in peak years, prime performance window.")

    # ── 6. Contract ────────────────────────────────────────────────────────
    cyr  = int(row.get("contract_years", 2))
    cexp = int(row.get("contract_expires", 2026))
    if cyr == 0:
        parts.append("Contract: free agent, available immediately at no transfer fee.")
    elif cyr == 1:
        parts.append(
            f"Contract: expires {cexp}, pre-contract available, "
            f"potential free agent — high urgency, low cost."
        )
    elif cyr <= 2:
        parts.append(
            f"Contract: {cyr} year(s) remaining, expiring soon — "
            f"seller may be motivated."
        )
    else:
        parts.append(f"Contract: {cyr} years remaining until {cexp}, secured.")

    # ── 7. Injury risk (only when elevated) ───────────────────────────────
    risk_label = str(row.get("injury_risk_label", "Moderate"))
    risk_score = int(row.get("injury_risk", 30))
    if risk_label in ("High", "Very High"):
        parts.append(
            f"Injury risk flag: {risk_label} ({risk_score}/100) based on age and "
            f"season minutes load — factor into squad planning."
        )
    elif risk_label == "Elevated":
        parts.append(f"Injury risk: Elevated ({risk_score}/100) — monitor workload.")

    # ── 8. Value context ───────────────────────────────────────────────────
    fv = _nonzero(row, "future_value")
    mv_low  = _nonzero(row, "mv_low")
    mv_high = _nonzero(row, "mv_high")
    gap = fv - mv

    if mv > 0:
        if mv_high > mv_low + 2:
            parts.append(
                f"Market value: {mv:.0f}M euros (model range: {mv_low:.0f}–{mv_high:.0f}M)."
            )
        else:
            parts.append(f"Market value: {mv:.0f}M euros.")

    if gap > 10:
        parts.append(
            f"Undervalued: model projects {fv:.0f}M future value, "
            f"{gap:.0f}M above current market — strong buy signal."
        )
    elif gap < -10:
        parts.append(
            f"Overvalued: model projects {fv:.0f}M future value, "
            f"{abs(gap):.0f}M below current market — overpriced."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Build / load matrix
# ---------------------------------------------------------------------------

def _pos_label(pos: str) -> str:
    return {"GK": "goalkeeper", "DF": "defender",
            "MF": "midfielder",  "FW": "forward"}.get(pos, "player")


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
        logger.info("Embedding model ready")
    return _model


def _cache_path(df: "pd.DataFrame") -> Path:
    import os
    from config import DATA_FILE_MULTI
    if os.path.exists(DATA_FILE_MULTI):
        return Path(EMBED_FILE_MULTI)
    return Path(EMBED_FILE)


def reset_embeddings() -> None:
    """Clear in-memory embedding matrix so the next call forces a rebuild."""
    global _matrix
    _matrix = None
    logger.info("Embedding matrix cleared (will rebuild on next request)")


def build_or_load_embeddings(df: "pd.DataFrame", force_rebuild: bool = False) -> np.ndarray:
    """
    Load cached embedding matrix or build from scratch.
    Matrix is L2-normalised so cosine similarity = dot product.

    Args:
        df:            Player DataFrame (used for size validation and building).
        force_rebuild: Skip cache entirely and rebuild from scratch.
    """
    global _matrix
    if _matrix is not None and not force_rebuild:
        return _matrix

    cache = _cache_path(df)
    if cache.exists() and not force_rebuild:
        logger.info("Loading cached embeddings from %s", cache)
        _matrix = np.load(str(cache))
        if _matrix.shape[0] == len(df):
            logger.info("Embeddings loaded: shape %s", _matrix.shape)
            return _matrix
        logger.warning(
            "Cache size mismatch (%d rows vs %d players) — rebuilding",
            _matrix.shape[0], len(df),
        )

    logger.info(
        "Building enriched embeddings for %d players …  (style, percentiles, arc, risk)",
        len(df),
    )
    model  = _load_model()
    texts  = [_player_to_text(row) for _, row in df.iterrows()]

    # Log a sample text so we can see what the model sees
    logger.info("Sample text (player 0):\n  %s", texts[0][:300])

    vecs   = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    _matrix = vecs.astype(np.float32)
    np.save(str(cache), _matrix)
    logger.info("Embeddings built and cached: shape %s → %s", _matrix.shape, cache)
    return _matrix


def encode_query(query: str) -> np.ndarray:
    """Encode a natural language query into a normalised 384-dim vector."""
    model = _load_model()
    vec   = model.encode([query], normalize_embeddings=True)
    return vec[0].astype(np.float32)


def explain_query(query: str, df: pd.DataFrame) -> dict:
    """
    Return top semantic matches for a query with similarity scores.
    Useful for debugging what the embedding retrieves.
    """
    matrix = build_or_load_embeddings(df)
    q_vec  = encode_query(query)
    sims   = (matrix @ q_vec)
    top5   = np.argsort(sims)[::-1][:5]
    return {
        "query": query,
        "top_matches": [
            {
                "player": str(df.iloc[i].get("player", "")),
                "position": str(df.iloc[i].get("position_primary", "")),
                "style": str(df.iloc[i].get("style_label", "")),
                "similarity": round(float(sims[i]), 4),
                "text_preview": _player_to_text(df.iloc[i])[:200],
            }
            for i in top5
        ],
    }
