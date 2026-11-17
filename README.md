# Premier League forward transfer-value model (2026-27)

Predicts Transfermarkt market values for Premier League forwards and wingers from FBref stats,
wages, contracts and Transfermarkt value history. Live site: https://claude.ai/artifact/7R5o7DMMQV4BqNVXFcrnog

## Rebuild

```bash
pip install lightgbm scikit-learn pandas rapidfuzz unidecode
python3 src/build_dataset.py        # data/raw/*.json -> data/interim/players_all.csv
python3 src/train.py > out/train_log.txt 2>&1   # nested-CV comparison (~6 min); don't pipe to head
python3 src/export.py > out/export_log.txt 2>&1 # final models + out/site_data.json
python3 -c "t=open('dashboard/template.html').read(); d=open('out/site_data.json').read(); open('dashboard/scouting_report.html','w').write(t.replace('__DATA__', d))"
```

## Raw data (collected 2026-11-16 via the browser)

- `data/raw/transfermarkt_2026.json`: 547 players from all 20 PL squads, including profiles, contracts, caps, full value history
  (`/ceapi/marketValueDevelopment/graph/<id>`).
- `data/raw/fbref_2026.json`: PL 2026-27 standard/shooting/misc/playing time/wages, PL 2025-26 equivalents, and Big-5
  2025-26 standard/shooting/misc. FBref sits behind Cloudflare: it worked in the user's real Chrome (Claude in Chrome),
  not in the built-in browser. FBref has no xG/xA/passing stats any more.

## Result (v13)

50/50 LightGBM + ridge blend on log value (today's money, 10%/yr inflation), value-trajectory and
valuation-date features, last-season stats from 16 leagues, trained on all positions, scored on the 153 forwards:
median APE 12.3%, mean 18.1%, 82% within ±25%, R² (log) 0.91. Naive "previous valuation" baseline: 18.2%.
Raw files: fbref_2026.json (PL + Big-5), fbref_other_2026.json (11 more leagues, 2025-26),
transfermarkt_2026.json, transfermarkt_headers_2026.json (profile labels).
