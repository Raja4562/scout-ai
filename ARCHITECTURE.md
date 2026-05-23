# ScoutAI - Architecture

## System Overview

```
                        SCOUTAI PLATFORM
  +------------------------------------------------------------------+
  |                                                                  |
  |   Browser (localhost:9003)                                       |
  |   +------------------------------------------------------------+ |
  |   |  Search  |  Best XI  |  Transfer  |  DNA  |  Gems  |  Agent | |
  |   |                                                            | |
  |   |  Session Memory  |  Shortlist  |  Scout Report  |  Compare | |
  |   +---------------------------+--------------------------------+ |
  |                               | HTTP / SSE                       |
  |   +---------------------------v--------------------------------+ |
  |   |              FastAPI Server (server.py)                    | |
  |   |                                                            | |
  |   |  POST /api/search          POST /api/bestxi               | |
  |   |  POST /api/agent           POST /api/transfer-window      | |
  |   |  POST /api/dna             GET  /api/gems                 | |
  |   |  POST /api/report          GET  /api/cheaper              | |
  |   |  GET  /api/memory          GET  /api/squad                | |
  |   +----+------------------+------------------+----------------+ |
  |        |                  |                  |                  |
  +--------+------------------+------------------+------------------+
           |                  |                  |
           v                  v                  v
  +----------------+  +---------------+  +---------------------+
  |  LLM Backend   |  |  ML Engine    |  |  Data Layer         |
  |                |  |               |  |                     |
  |  Ollama        |  |  all-MiniLM   |  |  2,135 players      |
  |  (local)       |  |  -L6-v2       |  |  5 leagues          |
  |     or         |  |               |  |                     |
  |  Anthropic     |  |  2135x384     |  |  LightGBM models    |
  |  Claude API    |  |  NumPy matrix |  |  K-Means clusters   |
  |                |  |               |  |                     |
  |  Template      |  |  Cosine       |  |  Data pipeline      |
  |  fallback      |  |  similarity   |  |  (pipeline/)        |
  +----------------+  +---------------+  +---------------------+
```

## Component Breakdown

### Core Modules

| File | Role |
|---|---|
| `server.py` | FastAPI application, all HTTP endpoints, SSE streaming, session store |
| `config.py` | Centralised configuration including paths, league constants, search weights, NL parsing dictionaries |
| `llm.py` | LLM abstraction layer, auto-detects Ollama or Anthropic, streams tokens, template fallback |
| `data_loader.py` | CSV loading with in-memory cache, player serialisation, season data lookup |
| `normalizer.py` | Cross-league stat normalisation using within-league percentiles and cross-league adjusted percentiles |

### Search Engine

| File | Role |
|---|---|
| `embeddings.py` | Builds and caches the 2135x384 embedding matrix using sentence-transformers |
| `search.py` | Hybrid search with NL filter parsing, preference stacking, cosine ranking, and stat-boost weighting |
| `memory.py` | Per-session preference memory with filter stacking, text qualifiers, and shortlist management |

### AI Agents and Analysis

| File | Role |
|---|---|
| `agent.py` | ReAct agent with 6 tools, Thought/Action/Observation loop, SSE streaming, fallback handling |
| `report.py` | Scout report generator with LLM narrative and structured value and verdict sections |
| `bestxi.py` | Constrained XI optimiser with greedy position-fill across 6 scoring modes and 6 formations |
| `transfer_window.py` | Transfer window planner with fee model (opening/expected/walk-away) and sell recommendations |
| `dna.py` | Team DNA with PSI-weighted embedding centroid and cosine-ranked stylistic matches |

### Data Pipeline

| File | Role |
|---|---|
| `pipeline/fetch_leagues.py` | Multi-league data ingestion across all 5 leagues |
| `pipeline/fetch_contracts.py` | Contract years and transfer market valuation data |
| `pipeline/cluster_styles.py` | K-Means style archetype clustering and UMAP dimensionality reduction |
| `pipeline/train_valuation.py` | LightGBM valuation model training with low/base/high ensemble |
| `pipeline/merge_csv.py` | League CSV merge, deduplication, and cross-league normalisation |
| `pipeline/refresh.py` | Full pipeline orchestrator with hot-reload trigger |

---

## Data Flow

```
1. Player data ingested from 5 leagues via pipeline/
   |
   v
2. Cross-league normalisation (percentile + difficulty coefficient)
   EPL = 1.000 | La Liga = 0.955 | Serie A = 0.940
   Bundesliga = 0.925 | Ligue 1 = 0.875
   |
   v
3. Style clustering (K-Means -> archetypes: Poacher, Playmaker, etc.)
   UMAP for scatter plot visualisation
   |
   v
4. LightGBM valuation model trained on performance + contract features
   Output: low / expected / high market value (confidence interval)
   |
   v
5. Sentence-transformer embeddings built for all 2135 players
   Each player -> 384-dim vector -> cached as player_embeddings_multi.npy
   |
   v
6. Server startup: loads CSV + embedding matrix into memory
   |
   v
7. Browser search query received
   |
   +-> NL parsing: extracts position, age, price, league, style, nationality
   |
   +-> Preference stacking: merges with session memory (accumulated filters)
   |
   +-> Query embedded (384-dim) -> cosine similarity against full matrix
   |
   +-> Hybrid score: 70% semantic + 20% PSI + 10% value gap
       Stat-boost keywords shift weights dynamically
   |
   +-> Results streamed card-by-card via SSE
   |
   +-> LLM summary streamed token-by-token via SSE
```

## Search Hybrid Scoring

```
For each candidate player:

  hybrid_score =
    semantic_weight  * cosine_similarity(query_embed, player_embed)
    + psi_weight     * normalised_psi
    + value_weight   * normalised_value_gap

Default weights: semantic=0.70, psi=0.20, value=0.10

Stat-boost keywords in the query shift weights. Examples:
  "clinical finisher"  -> +0.30 to g_per_shot percentile
  "hidden gem"         -> +0.25 to value_weight
  "world class"        -> +0.20 to psi_weight
```

## ReAct Agent Loop

```
User query
   |
   v
System prompt: agent persona + 6 tool definitions
   |
   v
Turn 1: LLM generates Thought + Action + Action Input
   |
   v
Tool executed:
  search_players     -> cosine search with filters
  analyze_team_gaps  -> position PSI vs league average
  find_transfers     -> budget-split across weak positions
  compare_players    -> side-by-side stat table
  get_player_profile -> full player summary
  simulate_window    -> complete 2-3 signing plan
   |
   v
Observation injected back into conversation
   |
   v
Turn 2+: LLM generates next Thought or Final Answer
   |
   v
After 2+ tool results: force Final Answer
   |
   v
SSE events streamed to browser throughout
```

## Best XI Optimiser

```
Input: budget, formation, mode, league filter, age range,
       locked players, max per club, nationality cap

Formation positions resolved (e.g. 4-3-3 -> GK LB CB CB RB CM CM CM LW ST RW)

For each position:
  1. Filter eligible players (position match, budget remaining, age, league, club cap)
  2. Score by selected mode:
       psi      -> raw PSI
       value    -> PSI / market_value
       future   -> future_value / market_value
       young    -> PSI * (1 + age_bonus for U24)
       transfer -> PSI * availability_weight
       balanced -> 0.55 * PSI + 0.45 * value_score
  3. Lock any pre-selected players at this position
  4. Pick highest scorer and deduct market value from budget
  5. Move to next position

Also computes unconstrained XI (unlimited budget) for overlay comparison
```

## Transfer Window Fee Model

```
market_value from LightGBM model

opening_bid  = market_value * 0.75
expected_fee = market_value * 1.00
walk_away    = market_value * 1.30

Adjustments:
  contract_years <= 1   -> -15% (leverage for lower fee)
  arc_phase == pre-peak -> +10% (premium for upside)
  arc_phase == post-peak -> -10% (discount for decline risk)
  value_gap > 10M       -> +5% (market undervaluation signal)
```

## Team DNA Matching

```
1. Load squad for the target club
2. Optional: filter to starters (top 11 by PSI)
3. Compute PSI-weighted centroid of squad embeddings
4. L2-normalise the DNA vector
5. Cosine-rank all external players against DNA vector
6. Tiers: Excellent >=82 | Strong >=72 | Decent >=62 | Modest <62
```

## LLM Backend Detection

```
On startup:

  ANTHROPIC_API_KEY set?  ->  use Anthropic Claude (cloud)
        |
        no
        |
        v
  Ollama at localhost:11434?  ->  use Ollama (local, private)
        |
        no
        |
        v
  No LLM  ->  template mode (all non-LLM features still work)
```

## Session Memory Architecture

```
Per browser session (crypto.randomUUID key):

  memory = {
    team:             str | None,     # remembered club name
    budget:           float | None,   # remembered budget
    formation:        str | None,     # remembered formation
    max_price:        float | None,   # remembered price ceiling
    active_filters:   {               # stacked search filters
      position, league, max_age,
      max_price, style, nationality,
      arc_phase, min_minutes
    },
    text_qualifiers:  [str],          # e.g. ["press-resistant", "clinical"]
    shortlist:        [player_dict],  # saved players
    search_history:   [str],          # recent queries
  }

detect_reset("start over")  ->  clear_active_filters
detect_refinement("also")   ->  merge with existing stack
```

## Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Embeddings:** sentence-transformers all-MiniLM-L6-v2 (384 dimensions)
- **Search:** NumPy cosine similarity with full matrix multiply per query
- **ML:** LightGBM (valuation), scikit-learn K-Means (style clusters), UMAP (visualisation)
- **LLM:** Ollama (llama3) or Anthropic Claude API, auto-detected at startup
- **Frontend:** Vanilla JS, CSS custom properties, SSE streaming with no framework
- **Data:** soccerdata for stats, custom pipeline for contracts and valuations
- **Deployment:** Single process, runs locally on port 9003
