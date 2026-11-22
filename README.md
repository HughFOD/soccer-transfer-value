# Premier League forward transfer-value model (2026-27)

Predicts Transfermarkt market values for Premier League forwards and wingers from FBref stats,
wages, contracts and Transfermarkt value history.

**Live site: https://hughod.com/soccer-transfer-value/** (GitHub Pages, served from `docs/index.html`).

## Rebuild

```bash
pip install lightgbm scikit-learn pandas rapidfuzz unidecode
python3 src/build_dataset.py        # data/raw/*.json -> data/interim/players_all.csv (Premier League, v17)
python3 src/convert_big5.py         # v19: raw La Liga/Bundesliga/Serie A/Ligue 1 + FBref Big-5 scrape -> data/raw/*big5*
BIG5=1 OUTFILE=players_b5.csv python3 src/build_dataset.py   # all five leagues -> data/interim/players_b5.csv
python3 src/train.py > out/train_log.txt 2>&1   # nested-CV comparison (~6 min); don't pipe to head
python3 src/fee.py                  # likely-fee ratios from 610 real transfers -> data/interim/fee_model.pkl
# v19 site: five-league OOF runs, then export (also needs the v17 PL OOF files for the 50/50 PL average)
(cd src && INPUT=players_b5.csv FS=full_iy_lg TARGET=change ET=1 REPEATS=2 TAG=b5big BIGGRID=1 python3 allpos.py)
(cd src && INPUT=players_b5.csv FS=stats_xg_lg TARGET=level REPEATS=2 TAG=b5 BIGGRID=1 python3 allpos.py)
V19=1 INPUT=players_b5.csv FULL_FS=full_iy_lg FULL_TAG=_b5big STATS_FS=stats_xg_lg TAG=b5 BIGGRID=1 python3 src/export.py > out/export_log.txt 2>&1
python3 src/build_page.py           # -> dashboard/scouting_report.html + docs/index.html (publishes on push)
```

## Raw data (collected 2026-11-16 via the browser)

- `data/raw/transfermarkt_2026.json`: 547 players from all 20 PL squads, including profiles, contracts, caps, full value history
  (`/ceapi/marketValueDevelopment/graph/<id>`).
- `data/raw/fbref_2026.json`: PL 2026-27 standard/shooting/misc/playing time/wages, PL 2025-26 equivalents, and Big-5
  2025-26 standard/shooting/misc. FBref sits behind Cloudflare: it worked in the user's real Chrome (Claude in Chrome),
  not in the built-in browser. FBref has no xG/xA/passing stats any more.

## Result (v17, 2026-11-21)

All 545 Premier League players. The full model (equal blend of LightGBM, ridge and Extra Trees) predicts the change
from each player's previous Transfermarkt valuation, with injury and youth-international features; players aged 21
or under use a youth model also trained on the 2024-2026 end-of-season revaluations (2,089 past player-seasons,
weighted 0.3). Nested cross-validation over today's players: median APE 9.1%, mean 13.7%,
86% within ±25%. Walk-forward test (trained on summers 2024-25 only, predicting summer 2026):
median APE 9.4% vs 16.3% for "no change". Each estimate on the site has a per-factor breakdown.

Scripts: `src/allpos.py` (per-position OOF), `src/cmp.py` (feature-set comparison on forwards),
`src/fotmob.py` (FotMob features), `src/extras.py` (injuries, youth caps), `src/history.py` (past-season snapshots), `src/aug_eval.py` (past-season and walk-forward tests). Prediction log: `tracking/`.
