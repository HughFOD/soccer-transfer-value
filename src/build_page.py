"""Fill dashboard/template.html with out/site_data.json -> dashboard/scouting_report.html and docs/index.html.
The page has no calculator any more (removed 2026-11-22), so the exported trees, ridge and Extra Trees weights and the
per-player feature vectors are dropped; only what the page shows is kept."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "out/site_data.json"))
d["models"] = {k: {"importance": m["importance"]} for k, m in d["models"].items()}
for p in d["players"]:
    p.pop("fvec", None)
for k in ("median_fvec", "features_union", "priors", "medians"):
    d["meta"].pop(k, None)
html = open(ROOT / "dashboard/template.html").read().replace("__DATA__", json.dumps(d, separators=(",", ":")))
for out in (ROOT / "dashboard/scouting_report.html", ROOT / "docs/index.html"):
    out.write_text(html)
print("page bytes", len(html.encode()))
