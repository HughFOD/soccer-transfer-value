# Data (not committed)

The scraped inputs live outside the repo. To rebuild, place these in `data/raw/`:

- `transfermarkt_2026.json` - 547 PL players: squads, profiles, contracts, value history
  (`/ceapi/marketValueDevelopment/graph/<player_id>`), pulled from a logged-in browser tab.
- `transfermarkt_headers_2026.json` - profile header labels (international status).
- `fbref_2026.json` - FBref PL 2026-27 standard/shooting/misc/playing time/wages, PL 2025-26, Big-5 2025-26.
- `fbref_other_2026.json` - FBref 2025-26 stats for 11 more leagues (Championship, League One,
  Portugal, Netherlands, Belgium, Scotland, Denmark, Sweden, Turkey, Brazil, MLS).

Then run the pipeline in the top-level README. `data/interim/` and `out/` are generated.
