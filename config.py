"""
ScoutAI - Configuration
Central place for all paths, model names, and search constants.
"""

import os

# Paths
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_FILE         = os.path.join(BASE_DIR, "data", "players.csv")
DATA_FILE_MULTI   = os.path.join(BASE_DIR, "data", "players_all_leagues.csv")
EMBED_FILE        = os.path.join(BASE_DIR, "player_embeddings.npy")
EMBED_FILE_MULTI  = os.path.join(BASE_DIR, "player_embeddings_multi.npy")
TEMPLATE_DIR      = os.path.join(BASE_DIR, "templates")

# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Search constants
MIN_MINUTES      = 300     # minimum minutes to be included in search results
TOP_K_DEFAULT    = 24      # default number of results to return
TOP_K_SIMILAR    = 6       # similar players shown in profile panel
SEMANTIC_WEIGHT  = 0.70    # weight for semantic similarity in hybrid score
PSI_WEIGHT       = 0.20    # weight for PSI score in hybrid score
VALUE_WEIGHT     = 0.10    # weight for value gap in hybrid score

# ---------------------------------------------------------------------------
# Multi-league configuration
# ---------------------------------------------------------------------------

# Leagues supported. These match the "league" column in the CSV.
SUPPORTED_LEAGUES = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

# Brand colors per league (for UI pills, badges, and chart accents)
LEAGUE_COLORS = {
    "EPL":        "#38bdf8",   # sky blue
    "La Liga":    "#f97316",   # orange
    "Bundesliga": "#eab308",   # gold
    "Serie A":    "#8b5cf6",   # purple
    "Ligue 1":    "#ef4444",   # red
}

# Difficulty coefficient per league (EPL = reference 1.000)
# Used in cross-league normalization to adjust raw stats.
LEAGUE_DIFFICULTY = {
    "EPL":        1.000,
    "La Liga":    0.955,
    "Serie A":    0.940,
    "Bundesliga": 0.925,
    "Ligue 1":    0.875,
}

# Human-readable league display names
LEAGUE_LABELS = {
    "EPL":        "Premier League",
    "La Liga":    "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie A":    "Serie A",
    "Ligue 1":    "Ligue 1",
}

# ---------------------------------------------------------------------------
# Radar chart feature sets per position
# Two sets: _pct (within-league) and _xpct (cross-league)
# ---------------------------------------------------------------------------

RADAR_FEATURES = {
    # Columns use only what soccerdata v1.9 actually provides:
    # standard (goals/assists), shooting (sh/sot/g_per_shot), misc (tklw/int),
    # keeper (save%/cs%/ga90/sota/wins/saves). No xG, no aerial, no progression.

    "GK": {
        "labels": ["Save %", "Clean Sheet", "Low GA/90", "Shots Faced", "Saves", "Wins"],
        "cols":   ["gk_savepct_pct", "gk_cspct_pct", "gk_ga90_pct",
                   "gk_sota_pct",    "gk_saves_pct", "gk_wins_pct"],
        "xcols":  ["gk_savepct_xpct", "gk_cspct_xpct", "gk_ga90_xpct",
                   "gk_sota_xpct",    "gk_saves_xpct", "gk_wins_xpct"],
    },
    "DF": {
        "labels": ["Tackles", "Interceptions", "Fouls Won", "Assists/90", "Goals/90", "Shots/90"],
        "cols":   ["tackles_tkl_pct", "int_pct",        "fouls_drawn_pct",
                   "assists_per90_pct", "goals_per90_pct", "sh_per90_pct"],
        "xcols":  ["tackles_tkl_xpct", "int_xpct",       "fouls_drawn_xpct",
                   "assists_per90_xpct", "goals_per90_xpct", "sh_per90_xpct"],
    },
    "MF": {
        "labels": ["Goals/90", "Assists/90", "Shots/90", "Shot Acc", "Tackles", "Interceptions"],
        "cols":   ["goals_per90_pct", "assists_per90_pct", "sh_per90_pct",
                   "g_per_shot_pct",  "tackles_tkl_pct",   "int_pct"],
        "xcols":  ["goals_per90_xpct", "assists_per90_xpct", "sh_per90_xpct",
                   "g_per_shot_xpct",  "tackles_tkl_xpct",   "int_xpct"],
    },
    "FW": {
        "labels": ["Goals/90", "Assists/90", "Shots/90", "SoT/90", "Shot Acc", "Finishing"],
        "cols":   ["goals_per90_pct", "assists_per90_pct", "sh_per90_pct",
                   "sot_per90_pct",   "g_per_shot_pct",    "g_per_sot_pct"],
        "xcols":  ["goals_per90_xpct", "assists_per90_xpct", "sh_per90_xpct",
                   "sot_per90_xpct",   "g_per_shot_xpct",    "g_per_sot_xpct"],
    },
}

# ---------------------------------------------------------------------------
# Natural language parsing dictionaries (Feature 14)
# ---------------------------------------------------------------------------

# Position keyword mapping — ordered longest-first so "left back" matches before "back"
POSITION_KEYWORDS = {
    # Full-back variants
    "left back":      "DF", "right back":     "DF",
    "left-back":      "DF", "right-back":     "DF",
    " lb ":           "DF", " rb ":           "DF",
    # Wing-back / fullback
    "wing-back":      "DF", "wingback":       "DF",
    "fullback":       "DF", "full-back":      "DF",
    # Centre-back
    "centre-back":    "DF", "center-back":    "DF",
    "centerback":     "DF", "centrebacks":    "DF",
    " cb ":           "DF",
    # Generic defender
    "defender":       "DF",
    # Wingers / wide forwards
    "left winger":    "FW", "right winger":   "FW",
    "left wing":      "FW", "right wing":     "FW",
    " lw ":           "FW", " rw ":           "FW",
    "winger":         "FW",
    # Strikers
    "centre-forward": "FW", "center-forward": "FW",
    "false nine":     "FW", "false 9":        "FW",
    "shadow striker": "FW", "second striker": "FW",
    "striker":        "FW", "forward":        "FW",
    "attacker":       "FW",
    " cf ":           "FW",
    # Midfielders — attacking / no.10
    "number 10":      "MF", "no. 10":         "MF",
    "no.10":          "MF", "#10":            "MF",
    "number10":       "MF",
    "attacking midfielder": "MF",
    "attacking mid":  "MF",
    # Midfielders — no.8 / box-to-box
    "number 8":       "MF", "no. 8":          "MF",
    "no.8":           "MF", "#8":             "MF",
    "box-to-box":     "MF", "box to box":     "MF",
    "central midfielder": "MF",
    # Midfielders — defensive / holding
    "defensive midfielder": "MF",
    "defensive mid":  "MF",
    "holding midfielder": "MF",
    "holding mid":    "MF",
    "holding":        "MF",
    " dm ":           "MF", " cm ":           "MF",
    # Generic
    "midfielder":     "MF", "mid":            "MF",
    "playmaker":      "MF",
    # Goalkeepers
    "goalkeeper":     "GK", "goalie":         "GK",
    "keeper":         "GK", " gk ":           "GK",
}

# Nationality keyword → 3-letter ISO code (matches `nationality` column)
NATIONALITY_KEYWORDS: dict[str, str] = {
    "english":      "ENG",   "spanish":      "ESP",
    "french":       "FRA",   "german":       "GER",
    "italian":      "ITA",   "portuguese":   "POR",
    "brazilian":    "BRA",   "argentinian":  "ARG",
    "argentine":    "ARG",   "dutch":        "NED",
    "belgian":      "BEL",   "croatian":     "CRO",
    "croat":        "CRO",   "polish":       "POL",
    "senegalese":   "SEN",   "nigerian":     "NGA",
    "moroccan":     "MAR",   "ghanaian":     "GHA",
    "ivorian":      "CIV",   "colombian":    "COL",
    "uruguayan":    "URU",   "mexican":      "MEX",
    "american":     "USA",   "austrian":     "AUT",
    "swiss":        "SUI",   "swedish":      "SWE",
    "norwegian":    "NOR",   "danish":       "DEN",
    "scottish":     "SCO",   "welsh":        "WAL",
    "irish":        "IRL",   "turkish":      "TUR",
    "japanese":     "JPN",   "south korean": "KOR",
    "korean":       "KOR",   "serbian":      "SRB",
    "greek":        "GRE",   "czech":        "CZE",
    "ukrainian":    "UKR",   "algerian":     "ALG",
    "cameroonian":  "CMR",   "ecuadorian":   "ECU",
    "paraguayan":   "PAR",   "peruvian":     "PER",
    "venezuelan":   "VEN",   "icelandic":    "ISL",
    "albanian":     "ALB",   "slovenian":    "SVN",
    "slovak":       "SVK",   "bosnian":      "BIH",
    "austrian":     "AUT",   "egyptian":     "EGY",
    "congolese":    "COD",   "malian":       "MLI",
    "guinean":      "GUI",   "tunisian":     "TUN",
    "zimbabwean":   "ZIM",   "jamaican":     "JAM",
    "trinidadian":  "TRI",
}

# Stat boost keywords — phrases that add extra weight to specific percentile columns
# in the hybrid score.  Format: phrase → {pct_col_or_special: boost_weight}
# Special keys: "__value_boost" increases VALUE_WEIGHT; "__psi_boost" increases PSI_WEIGHT.
# Scan order matters — longer phrases first.
STAT_BOOST_KEYWORDS: list[tuple[str, dict]] = [
    # Finishing / clinical
    ("clinical finisher",    {"g_per_shot_pct": 0.35, "g_per_sot_pct": 0.15}),
    ("efficient finisher",   {"g_per_shot_pct": 0.35}),
    ("penalty box finisher", {"g_per_shot_pct": 0.30, "sot_per90_pct": 0.20}),
    ("clinical",             {"g_per_shot_pct": 0.30, "g_per_sot_pct": 0.15}),
    # Goals volume
    ("prolific scorer",      {"goals_per90_pct": 0.40, "sh_per90_pct": 0.15}),
    ("goals",                {"goals_per90_pct": 0.30}),
    # Creativity / assists
    ("high assist output",   {"assists_per90_pct": 0.40}),
    ("chance creator",       {"assists_per90_pct": 0.35}),
    ("creates chances",      {"assists_per90_pct": 0.35}),
    ("progresses the ball",  {"assists_per90_pct": 0.25}),
    ("ball progression",     {"assists_per90_pct": 0.25}),
    ("progressive",          {"assists_per90_pct": 0.20}),
    ("creative",             {"assists_per90_pct": 0.30}),
    ("assists",              {"assists_per90_pct": 0.30}),
    # Defensive work
    ("defensive work rate",  {"tackles_tkl_pct": 0.25, "int_pct": 0.25}),
    ("high work rate",       {"tackles_tkl_pct": 0.20}),
    ("combative",            {"tackles_tkl_pct": 0.30, "int_pct": 0.20}),
    ("aggressive tackles",   {"tackles_tkl_pct": 0.30}),
    ("tackles well",         {"tackles_tkl_pct": 0.30}),
    ("reads the game",       {"int_pct": 0.30}),
    ("interceptions",        {"int_pct": 0.25}),
    # Shot volume
    ("shoots a lot",         {"sh_per90_pct": 0.35}),
    ("shot volume",          {"sh_per90_pct": 0.30}),
    ("direct",               {"sh_per90_pct": 0.20}),
    # GK
    ("clean sheets",         {"gk_cspct_pct": 0.40}),
    ("shot stopper",         {"gk_savepct_pct": 0.40}),
    ("saves well",           {"gk_savepct_pct": 0.35}),
    # Value signals
    ("hidden gem",           {"__value_boost": 0.25}),
    ("undervalued",          {"__value_boost": 0.20}),
    ("good value",           {"__value_boost": 0.15}),
    ("bargain",              {"__value_boost": 0.20}),
    ("cheap",                {"__value_boost": 0.15}),
    ("low cost",             {"__value_boost": 0.15}),
    # Quality signals
    ("elite",                {"__psi_boost": 0.15}),
    ("world class",          {"__psi_boost": 0.20}),
    ("top level",            {"__psi_boost": 0.15}),
]

# League keyword mapping for natural language parsing
LEAGUE_KEYWORDS = {
    "premier league":    "EPL",
    "premier":           "EPL",
    "epl":               "EPL",
    "bundesliga":        "Bundesliga",
    "serie a":           "Serie A",
    "seriea":            "Serie A",
    "ligue 1":           "Ligue 1",
    "ligue1":            "Ligue 1",
    "la liga":           "La Liga",
    "laliga":            "La Liga",
}
