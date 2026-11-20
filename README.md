# Premier League forward transfer-value model (2026-27)

Predicts Transfermarkt market values for Premier League forwards and wingers from FBref stats,
wages, contracts and Transfermarkt value history.

**Live site: https://hughod.com/transfer-value-model/** (GitHub Pages, served from `docs/index.html`).

## Rebuild

```bash
pip install lightgbm scikit-learn pandas rapidfuzz unidecode
python3 src/build_dataset.py        # data/raw/*.json -> data/interim/players_all.csv
python3 src/train.py > out/train_log.txt 2>&1   # nested-CV comparison (~6 min); don't pipe to head
python3 src/export.py > out/export_log.txt 2>&1 # final models + out/site_data.json
python3 -c "t=open('dashboard/template.html').read(); d=open('out/site_data.json').read(); open('dashboard/scouting_report.html','w').write(t.replace('__DATA__', d))"
cp dashboard/scouting_report.html docs/index.html   # publishes to GitHub Pages on push
```

## Raw data (collected 2026-11-16 via the browser)

- `data/raw/transfermarkt_2026.json`: 547 players from all 20 PL squads, including profiles, contracts, caps, full value history
  (`/ceapi/marketValueDevelopment/graph/<id>`).
- `data/raw/fbref_2026.json`: PL 2026-27 standard/shooting/misc/playing time/wages, PL 2025-26 equivalents, and Big-5
  2025-26 standard/shooting/misc. FBref sits behind Cloudflare: it worked in the user's real Chrome (Claude in Chrome),
  not in the built-in browser. FBref has no xG/xA/passing stats any more.

## Result (v14, 2026-11-20)

All 545 Premier League players. 50/50 LightGBM + ridge blend on log value (today's money, 10%/yr inflation),
value-trajectory and valuation-date features, trained and scored with nested cross-validation over all players:
median APE 10.4% overall (forwards 13.2%, midfielders 9.3%, defenders 9.1%, goalkeepers 11.9%), 84% within
±25%, R² (log) 0.97. Naive "previous valuation" baseline: 18.3%. Stats-only model (with FotMob xG and
defensive stats): 26% median.

Scripts: `src/allpos.py` (per-position OOF), `src/cmp.py` (feature-set comparison on forwards),
`src/fotmob.py` (FotMob features). Prediction log: `tracking/`.
