"""
Semantic search benchmark — Feature 13 validation.
Tests that style-archetype queries retrieve the right archetype players.

Two modes:
  1. Pure semantic (no position filter, no hybrid) — tests embedding quality in isolation
  2. Full search (search_players with position filter + hybrid) — tests real system
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from data_loader import load_players
from embeddings import build_or_load_embeddings, encode_query
from search import search_players

df     = load_players()
matrix = build_or_load_embeddings(df)

QUERIES = [
    # (query, expected_style_labels, expected_positions)
    ("press-resistant midfielder with good range of passing",
     ["Press-Resistant 8", "Playmaker", "Box-to-Box"], ["MF"]),

    ("inverted winger who cuts inside onto stronger foot",
     ["Inverted Winger", "Creative Forward"], ["FW"]),

    ("defensive anchor centre-back no-nonsense tackles and interceptions",
     ["Defensive Anchor", "Stopper", "Sweeper CB"], ["DF"]),

    ("sweeper keeper comfortable on the ball",
     ["Sweeper Keeper"], ["GK"]),

    ("poacher clinical penalty-box finisher efficient goals",
     ["Poacher"], ["FW"]),

    ("attacking fullback with high assist output overlapping runs",
     ["Attacking Fullback", "Ball-Playing CB"], ["DF"]),

    ("young striker pre-peak strong upside trajectory undervalued",
     ["Poacher", "Creative Forward", "Target Man", "Inverted Winger", "Pressing Forward"], ["FW"]),

    ("ball winner combative defensive midfielder aggressive tackles",
     ["Ball Winner", "Holding Midfielder", "Press-Resistant 8"], ["MF"]),
]


def precision_at_k(sims, df, expected_styles, expected_positions, k=10):
    """What fraction of the top-k results match expected style or position."""
    top_idx = np.argsort(sims)[::-1][:k]
    hits = 0
    for i in top_idx:
        row = df.iloc[i]
        style = str(row.get("style_label", ""))
        pos   = str(row.get("position_primary", ""))
        if style in expected_styles or pos in expected_positions:
            hits += 1
    return hits / k


def search_precision_at_k(results, expected_styles, k=10):
    """Precision of search_players results against expected styles only."""
    top = results[:k]
    hits = sum(1 for r in top if r.get("style_label", "") in expected_styles)
    return hits / max(len(top), 1)


# ── 1. Pure semantic mode (embedding-only, no filters) ────────────────────────
print(f"\n{'=== MODE 1: Pure Semantic (embedding only, no position filter) ':=<110}")
print(f"\n{'Query':<60} {'P@5':>6} {'P@10':>6}  Top-3 retrieved")
print("-" * 110)

for query, exp_styles, exp_pos in QUERIES:
    q_vec = encode_query(query)
    sims  = (matrix @ q_vec)
    top10 = np.argsort(sims)[::-1][:10]

    p5  = precision_at_k(sims, df, exp_styles, exp_pos, k=5)
    p10 = precision_at_k(sims, df, exp_styles, exp_pos, k=10)

    top3 = []
    for i in top10[:3]:
        row = df.iloc[i]
        top3.append(f"{row['player'][:15]} [{row.get('style_label','?')[:12]}]")

    q_short = query[:58]
    print(f"{q_short:<60} {p5:>5.0%} {p10:>5.0%}  {' | '.join(top3)}")


# ── 2. Full search mode (search_players with position filter + hybrid score) ──
print(f"\n\n{'=== MODE 2: Full Search (search_players with position filter + hybrid) ':=<110}")
print(f"\n{'Query':<60} {'Style-P@5':>10} {'Style-P@10':>11}  Top-3 retrieved")
print("-" * 115)

for query, exp_styles, exp_pos in QUERIES:
    results = search_players(query, top_k=10)

    sp5  = search_precision_at_k(results, exp_styles, k=5)
    sp10 = search_precision_at_k(results, exp_styles, k=10)

    top3 = []
    for r in results[:3]:
        top3.append(f"{r['player'][:15]} [{r.get('style_label','?')[:12]}]")

    q_short = query[:58]
    print(f"{q_short:<60} {sp5:>9.0%} {sp10:>10.0%}  {' | '.join(top3)}")

print()
