import pandas as pd
from embeddings import _player_to_text

df = pd.read_csv('data/players_all_leagues.csv', low_memory=False)

for name in ['Haaland', 'Bellingham', 'van Dijk', 'Alisson']:
    row = df[df['player'].str.contains(name, case=False, na=False)]
    if not row.empty:
        r = row.iloc[0]
        style = str(r.get('style_label', '?'))
        text = _player_to_text(r)
        print(f'=== {r["player"]} [{style}] ===')
        print(text[:500])
        print()
