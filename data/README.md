# Data (not committed)

The scraped inputs live outside the repo. To rebuild, place these in `data/raw/`:

- `transfermarkt_2026.json` - 547 PL players: squads, profiles, contracts, value history
  (`/ceapi/marketValueDevelopment/graph/<player_id>`), pulled from a logged-in browser tab.
- `transfermarkt_headers_2026.json` - profile header labels (international status).
- `fbref_2026.json` - FBref PL 2026-27 standard/shooting/misc/playing time/wages, PL 2025-26, Big-5 2025-26.
- `fbref_other_2026.json` - FBref 2025-26 stats for 11 more leagues (Championship, League One,
  Portugal, Netherlands, Belgium, Scotland, Denmark, Sweden, Turkey, Brazil, MLS).

- `fotmob_2026.json` - FotMob player stats (xG, xA, chance creation, defensive actions, match rating),
  2026-27 PL and 2025-26 for 16 leagues, from data.fotmob.com stat lists.
- `transfermarkt_transfers_2026.json` - every transfer and fee for the 547 players (`/ceapi/transferHistory/list/<id>`).

- `transfermarkt_extra_2026.json` - TM squads for 2023-24 to 2026-27, plus injury histories, national-team careers
  (`tmapi.transfermarkt.technology/player/<id>/national-career-history`) and value histories for 1,523 players.
- `fotmob_hist.json` - FotMob stats for 2022-23 to 2024-25 (Turkey, Brazil and MLS missing).

- v19 (scraped 2026-11-22): `tm_big5_raw.json` (Transfermarkt squads, profiles, value histories, transfers, injuries and
  national careers for 2,094 La Liga, Bundesliga, Serie A and Ligue 1 players), `fbref_big5_raw.json` (FBref Big-5
  2026-27 standard/shooting/misc/playing time, 2025-26 playing time, wages for the four leagues) and
  `tm_big5_past_raw.json` (their squads 2023-24 to 2025-26 plus 3,264 more players' histories).
  `src/convert_big5.py` turns the first two into the `*big5*` files the pipeline reads.

- v20 (2026-11-22): `fotmob_deep.json` - FotMob player and team stat lists for the five leagues, 2022-23 to
  2026-27 (xG on target, penalties, goalkeeping, team xG for/against, possession). `fbref_deep.json` - FBref
  Big-5 passing/GCA/possession/defense/keepersadv for 2023-24 to 2026-27 (advanced columns blank; kept for the
  record) and standard/shooting/misc/playing time for 2023-24 and 2024-25.

Then run the pipeline in the top-level README. `data/interim/` and `out/` are generated.
