"""Convert the raw Transfermarkt scrape of La Liga, Bundesliga, Serie A and Ligue 1 (data/raw/tm_big5_raw.json,
pulled in the browser on 2026-11-22) into the same shapes the Premier League files use:

  transfermarkt_big5_2026.json            like transfermarkt_2026.json (+ "league")
  transfermarkt_big5_headers_2026.json    like transfermarkt_headers_2026.json
  transfermarkt_big5_transfers_2026.json  like transfermarkt_transfers_2026.json
  transfermarkt_big5_extra_2026.json      like transfermarkt_extra_2026.json["players"]

FBref: data/raw/fbref_big5_raw.json (Big-5 2026-27 standard/shooting/misc/playing time, 2025-26 playing
time, wages) -> fbref_big5_2026.json.
"""
import json, re
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data/raw"
DATE = re.compile(r"^\d\d/\d\d/\d{4}$")


def player_row(o):
    sq, info, hdr = o["sq"], o.get("info", {}), o.get("hdr", [])
    cells = sq.get("cells", [])
    dob = next((c for c in cells if re.match(r"\d\d/\d\d/\d{4} \(\d+\)", c)), info.get("Date of birth/Age:", ""))
    dates = [c for c in cells if DATE.match(c)]
    caps, goals, status, team = 0, 0, "", ""
    for h in hdr:
        m = re.match(r"Caps/Goals:\s*(\d+)\s*/\s*(\d+)", h)
        if m:
            caps, goals = int(m.group(1)), int(m.group(2))
        m = re.match(r"(Current international|National player|Former International):\s*(.+)", h)
        if m:
            status, team = m.group(1), m.group(2).strip()
    lvl = next((h.split(":", 1)[1].strip() for h in hdr if h.startswith("League level:")), "")
    mvu = re.search(r"Last update:\s*(\d\d/\d\d/\d{4})", o.get("mvbox", ""))
    return {
        "name": sq["name"], "id": sq["id"], "href": sq["href"], "club": sq["club"], "position": sq.get("position") or "",
        "dob_age": dob, "nat": sq.get("nat", ""),
        "height": info.get("Height:", next((c for c in cells if c.endswith("m") and "," in c), "")),
        "foot": info.get("Foot:", next((c for c in cells if c in ("left", "right", "both")), "")),
        "joined": info.get("Joined:", dates[0] if dates else ""),
        "signed_from": sq.get("signed_from", ""),
        "contract": info.get("Contract expires:", dates[1] if len(dates) > 1 else ""),
        "mv": sq.get("mv", ""), "caps": caps, "intl_goals": goals, "intl_status": status, "intl_team": team,
        "league_level": lvl, "pos_full": info.get("Position:", ""),
        "last_ext": info.get("Last contract extension:", ""), "agent": info.get("Player agent:", ""),
        "outfitter": info.get("Outfitter:", ""), "mv_updated": mvu.group(1) if mvu else "",
        "mvh": o.get("mvh", []), "league": sq["league"],
    }


if __name__ == "__main__":
    raw = json.load(open(RAW / "tm_big5_raw.json"))
    P = raw["P"] if "P" in raw else raw
    rows, hdrs, trf, extra = [], {}, {}, {}
    for pid, o in P.items():
        if not o.get("sq") or not o["sq"].get("mv"):
            continue
        rows.append(player_row(o))
        hdrs[pid] = [h for h in o.get("hdr", []) if h]
        trf[pid] = o.get("trf", [])
        extra[pid] = {"inj": o.get("inj", []), "nat": o.get("nat", []), "mvh": o.get("mvh", [])}
    json.dump(rows, open(RAW / "transfermarkt_big5_2026.json", "w"))
    json.dump(hdrs, open(RAW / "transfermarkt_big5_headers_2026.json", "w"))
    json.dump(trf, open(RAW / "transfermarkt_big5_transfers_2026.json", "w"))
    json.dump(extra, open(RAW / "transfermarkt_big5_extra_2026.json", "w"))
    from collections import Counter
    print(len(rows), "players", Counter(r["league"] for r in rows),
          "with history", sum(len(r["mvh"]) > 1 for r in rows), "with profile", sum(bool(h) for h in hdrs.values()))
    fbp = RAW / "fbref_big5_raw.json"
    if fbp.exists():
        fb = json.load(open(fbp))
        fb = {k: v for k, v in fb.items() if not k.startswith("_")}
        json.dump(fb, open(RAW / "fbref_big5_2026.json", "w"))
        print("fbref", {k: len(v["rows"]) for k, v in fb.items()})
