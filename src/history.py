"""Past-season snapshots for training (v17 experiment).

For each summer S in 2024, 2025, 2026 we rebuild what the model sees today, as it stood then:
  squad   = Transfermarkt Premier League squads of season (S-1)/S
  target  = the player's first Transfermarkt valuation between 1 May and 31 Aug of S (the end-of-season re-rating)
  inputs  = valuation history strictly before that date, age/injuries/national-team career as of that date,
            FotMob stats for season (S-1)/S (full season) and (S-2)/(S-1)
Contract length and wages aren't available historically, so they are missing on these rows.
All euros are restated to 16 Nov 2026 money at the same 10%/yr as the live model.

Output: data/interim/hist_rows.csv with the AUG feature columns (see AUG in src/train.py), log target, snapshot,
player id. Current (Nov 2026) rows get the same columns in build_dataset.py.
"""
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extras import extra, injury_youth_features  # noqa: E402
from fotmob import fotmob_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, INTERIM = ROOT / "data/raw", ROOT / "data/interim"
TODAY = date(2026, 11, 16)
INFLATION = 0.10
POS_GROUP = {"Goalkeeper": "GK", "Centre-Back": "DF", "Left-Back": "DF", "Right-Back": "DF",
             "Defensive Midfield": "MF", "Central Midfield": "MF", "Attacking Midfield": "MF",
             "Left Midfield": "MF", "Right Midfield": "MF"}


def dmy(s):
    try:
        return datetime.strptime(str(s).strip()[:10], "%d/%m/%Y").date()
    except Exception:
        return None


def infl(v, d):
    return v * (1 + INFLATION) ** ((TODAY - d).days / 365.25)


def history_at(pts, cur_d):
    """Same definitions as build_dataset.history_features, anchored at cur_d (the target valuation date)."""
    before = [(d, v) for d, v in pts if d < cur_d]
    out = dict(prev_val=np.nan, months_since_prev=np.nan, val_12m_before=np.nan, peak_before=np.nan,
               momentum_12m=np.nan, n_valuations=len(before))
    if not before:
        return out
    d, v = before[-1]
    out["prev_val"] = infl(v, d)
    out["months_since_prev"] = (cur_d - d).days / 30.44
    out["peak_before"] = max(infl(vv, dd) for dd, vv in before)
    anchor = date(cur_d.year - 1, cur_d.month, min(cur_d.day, 28))
    old = [(dd, vv) for dd, vv in before if dd <= anchor]
    if old:
        out["val_12m_before"] = infl(old[-1][1], old[-1][0])
        out["momentum_12m"] = np.log(out["prev_val"] / out["val_12m_before"])
    return out


def senior_caps_at(nat, caps_now, as_of):
    """Approximate senior caps at a past date: find the national team whose total matches today's senior caps and
    scale it by the share of that player's senior career (debut -> last game) that had elapsed."""
    if not caps_now or caps_now != caps_now:
        return 0.0
    for club_id, games, goals, debut, last, state in nat:
        if games == caps_now:
            d0, d1 = dmy_iso(debut), dmy_iso(last)
            if not d0 or d0 > as_of:
                return 0.0
            if not d1 or d1 <= as_of:
                return float(caps_now)
            return float(caps_now) * (as_of - d0).days / max((d1 - d0).days, 1)
    return float(caps_now)


def dmy_iso(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def build_snapshot(S, fm, caps_now):
    E = extra()
    squad = E["squads"].get(str(S - 1), [])
    rows = []
    for r in squad:
        pid, name, club, position, dob_age, height, foot, joined, contract, mv, href = r
        mvh = E["players"].get(str(pid), {}).get("mvh", [])
        pts = sorted((dmy(d), v) for d, v, _ in mvh if dmy(d) and v)
        tgt = [(d, v) for d, v in pts if date(S, 5, 1) <= d <= date(S, 8, 31)]
        if not tgt:
            continue
        td, tv = tgt[0]
        dob = dmy((re.search(r"\d\d/\d\d/\d{4}", str(dob_age)) or [None])[0]) if re.search(r"\d\d/\d\d/\d{4}", str(dob_age)) else None
        h = history_at(pts, td)
        hc = re.search(r"([\d,]+)\s*m", str(height))
        rows.append(dict(snapshot=f"summer {S}", id=str(pid), name=name, club=club, position=position,
                         pos_group=POS_GROUP.get(position, "FW"), dob=dob, as_of=td,
                         market_value_eur=infl(tv, td), age=(td - dob).days / 365.25 if dob else np.nan,
                         height_cm=float(hc.group(1).replace(",", ".")) * 100 if hc else np.nan,
                         foot=str(foot) if foot else "unknown", months_since_valuation=0.0, season_share=1.0, **h))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    nat_all = {pid: E["players"].get(pid, {}).get("nat", []) for pid in df["id"]}
    df["international_caps"] = [senior_caps_at(nat_all[p], caps_now.get(p, np.nan), d) for p, d in zip(df["id"], df["as_of"])]
    iy = pd.concat([injury_youth_features([p], [dob], [c], d, date(d.year - 1, 8, 1))
                    for p, dob, c, d in zip(df["id"], df["dob"], df["international_caps"], df["as_of"])], ignore_index=True)
    df = pd.concat([df.reset_index(drop=True), iy], axis=1)
    fmf = fotmob_features(df.rename(columns={"name": "name"}), fm, cur_key=f"premier_league|{S-1}/{S}",
                          prev_seasons=(f"{S-2}/{S-1}", str(S - 1)))
    if fmf is not None:
        df = pd.concat([df, fmf.reset_index(drop=True)], axis=1)
    df["log_mv"] = np.log(df["market_value_eur"])
    g = df.groupby("club")["log_mv"]
    df["club_loo_log_mv"] = (g.transform("sum") - df["log_mv"]) / (g.transform("count") - 1)
    return df


if __name__ == "__main__":
    fm = json.load(open(RAW / "fotmob_2026.json"))
    hist_path = RAW / "fotmob_hist.json"
    if hist_path.exists():
        fm.update(json.load(open(hist_path)))
    cur = pd.read_csv(INTERIM / "players_all.csv", low_memory=False)
    caps_now = dict(zip(cur["id"].astype(str), cur["international_caps"]))
    # players who left the PL: take senior caps from the national-team career (largest team total that is a current
    # senior label is unknown, so fall back to 0 - they get youth features from the same career data)
    out = pd.concat([build_snapshot(S, fm, caps_now) for S in (2024, 2025, 2026)], ignore_index=True)
    out.to_csv(INTERIM / "hist_rows.csv", index=False)
    print(out.groupby("snapshot").agg(n=("id", "size"), with_prev=("prev_val", lambda s: s.notna().sum()),
                                      fm=("fm_matched", "sum"), med_val=("market_value_eur", "median")))
