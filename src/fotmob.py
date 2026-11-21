"""FotMob player stats (xG, xA, chance creation, defensive actions, match rating) -> per-player features.

Input : data/raw/fotmob_2026.json  {"<league>|<season>": {"sid": .., "players": {fotmob_id: {...}}}}
Output: fotmob_features(tm_df) -> DataFrame indexed like tm_df

Matching: a TM player is matched to the 2026-27 Premier League FotMob table by name + club. FotMob ids are
stable across leagues and seasons, so the same id then finds his 2025-26 row anywhere. Players with no
2026-27 PL minutes fall back to a strict name match in the 2025-26 tables.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

RAW = Path(__file__).resolve().parents[1] / "data/raw"
COUNT_STATS = {
    "expected_goals": "xg", "expected_assists": "xa", "big_chance_created": "bcc", "total_att_assist": "chances",
    "won_contest": "dribbles", "total_scoring_att": "fm_shots", "big_chance_missed": "bcm",
    "total_tackle": "tackles", "interception": "fm_int", "effective_clearance": "clearances",
    "outfielder_block": "blocks", "ball_recovery": "recoveries", "poss_won_att_3rd": "press_wins",
    "defensive_contributions": "def_contrib", "accurate_pass": "passes", "accurate_long_balls": "long_balls",
    "goals": "fm_goals", "goal_assist": "fm_assists",
}
# FotMob publishes these as per-90 rates (checked: Mazraoui total_tackle 2.7 over 233 min; Saka accurate_pass 27.4);
# the rest are season totals. Per-90 ones are turned back into totals (x minutes / 90) before pooling seasons.
PER90_KEYS = {"total_scoring_att", "ontarget_scoring_att", "won_contest", "accurate_pass", "accurate_long_balls",
              "total_tackle", "interception", "effective_clearance", "outfielder_block", "ball_recovery",
              "defensive_contributions", "poss_won_att_3rd"}
W_PREV, K = 0.75, 6.0


def norm(s):
    s = unidecode(str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def _frame(entry):
    rows = []
    for pid, p in entry["players"].items():
        r = {"fm_id": str(pid), "fm_name": p.get("n"), "fm_team": p.get("t"), "min": p.get("min") or 0,
             "rating": p.get("rating"), "rating_n": p.get("rating_n")}
        for k, v in COUNT_STATS.items():
            val = p.get(k, 0) or 0
            r[v] = val * r["min"] / 90 if k in PER90_KEYS else val
        rows.append(r)
    return pd.DataFrame(rows)


CALENDAR = {"allsvenskan", "brazil", "mls"}


def fotmob_features(tm, fm=None, cur_key="premier_league|2026/2027", prev_seasons=("2025/2026", "2025")):
    if fm is None:
        path = RAW / "fotmob_2026.json"
        if not path.exists():
            return None
        fm = json.load(open(path))
    if cur_key not in fm:
        return None
    cur = _frame(fm[cur_key])
    prev_parts = []
    for key, entry in fm.items():
        lg, season = key.split("|")
        if ("/" in season) == (lg in CALENDAR):   # calendar leagues use "2025", the rest "2025/2026"
            continue
        if season in prev_seasons:
            f = _frame(entry)
            f["league"] = lg
            prev_parts.append(f)
    prev = pd.concat(prev_parts, ignore_index=True)
    # a player who moved mid-season appears in two leagues: sum counts, minutes-weighted rating
    prev["rating_x_min"] = prev["rating"].fillna(0) * prev["min"]
    prev["rated_min"] = prev["min"].where(prev["rating"].notna(), 0)
    agg = {v: "sum" for v in COUNT_STATS.values()}
    agg.update({"min": "sum", "rating_x_min": "sum", "rated_min": "sum", "fm_name": "first", "league": lambda s: "/".join(dict.fromkeys(s))})
    prev = prev.groupby("fm_id").agg(agg).reset_index()
    prev["rating"] = prev["rating_x_min"] / prev["rated_min"].replace(0, np.nan)
    cur["n"] = cur["fm_name"].map(norm)
    prev["n"] = prev["fm_name"].map(norm)

    def club_ok(tm_club, fm_team):
        a = norm(tm_club).replace(" fc", "").replace("afc ", "")
        b = norm(fm_team)
        return max(fuzz.partial_ratio(a, b), fuzz.token_set_ratio(a, b))

    ids, how = [], []
    for r in tm.itertuples():
        n = norm(r.name)
        best, score = None, 0
        for c in cur.itertuples():
            s = max(fuzz.token_set_ratio(n, c.n), fuzz.token_sort_ratio(n, c.n))
            if s >= 80:
                s += 0.2 * club_ok(r.club, c.fm_team)
                if s > score:
                    best, score = c.fm_id, s
        if best is not None and score >= 96:
            ids.append(best); how.append("pl_2627")
            continue
        best, score = None, 0
        for c in prev.itertuples():
            s = fuzz.token_sort_ratio(n, c.n)
            if s > score:
                best, score = c.fm_id, s
        ids.append(best if score >= 95 else None); how.append("prev_name" if score >= 95 else "none")
    out = pd.DataFrame({"fm_id": ids, "fm_match": how}, index=tm.index)
    # a FotMob id claimed twice: drop both (ambiguous)
    dup = out["fm_id"].notna() & out["fm_id"].duplicated(keep=False)
    team_of = dict(zip(cur.fm_id, cur.fm_team))
    for fid in out.loc[dup, "fm_id"].unique():
        rows = out.index[out["fm_id"] == fid]
        scores = {i: club_ok(tm.loc[i, "club"], team_of.get(fid, "")) for i in rows}
        keep = max(scores, key=scores.get)
        for i in rows:
            if i != keep or list(scores.values()).count(scores[keep]) > 1:
                out.loc[i, ["fm_id", "fm_match"]] = [None, "ambiguous"]

    c = cur.set_index("fm_id"); p = prev.set_index("fm_id")
    get = lambda tbl, col: out["fm_id"].map(lambda i: tbl[col].get(i, np.nan) if isinstance(i, str) else np.nan)
    min_c, min_p = get(c, "min").fillna(0), get(p, "min").fillna(0)
    n90 = min_c / 90 + W_PREV * min_p / 90
    has = out["fm_id"].notna()
    feats = pd.DataFrame(index=tm.index)
    for col in COUNT_STATS.values():
        tot = get(c, col).fillna(0) + W_PREV * get(p, col).fillna(0)
        prior = tot[has].sum() / n90[has].sum()
        feats[f"{col}_p90"] = np.where(has, (tot + K * prior) / (n90 + K), np.nan)
    feats["xgxa_p90"] = feats["xg_p90"] + feats["xa_p90"]
    goals = get(c, "fm_goals").fillna(0) + W_PREV * get(p, "fm_goals").fillna(0)
    xg = get(c, "xg").fillna(0) + W_PREV * get(p, "xg").fillna(0)
    feats["finishing"] = np.where(has, (goals - xg) / (n90 + K), np.nan)           # goals above xG per 90, shrunk
    r_c, r_p = get(c, "rating"), get(p, "rating")
    w_c, w_p = min_c.where(r_c.notna(), 0), (W_PREV * min_p).where(r_p.notna(), 0)   # only minutes that carry a rating
    num = r_c.fillna(0) * w_c + r_p.fillna(0) * w_p
    feats["fm_rating"] = np.where(has, (num + 6.7 * 540) / (w_c + w_p + 540), np.nan)  # shrunk to a 6.7 average
    feats["fm_matched"] = has.astype(int)
    feats["fm_min_cur"] = min_c.where(has)
    feats["fm_min_prev"] = min_p.where(has)
    feats["fm_prev_league"] = get(p, "league")
    return pd.concat([out, feats], axis=1)
