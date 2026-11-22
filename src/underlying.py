"""Deep underlying-metric features (v20) for every player in data/interim/players_b5.csv.

Sources (all free):
  FotMob, 5 leagues, 2022-23 .. 2026-27 (the main source): xG, xA, xG on target, shots, chances created, big
  chances, dribbles, passes, long balls, tackles, interceptions, clearances, blocks, recoveries, high
  turnovers, penalties won, ratings, goalkeepers' goals prevented, team xG for/against and possession.
  FBref Big-5, 2023-24 .. 2026-27: minutes, goals, non-penalty goals, penalties taken. (FBref's passing, GCA,
  possession and defense tables were scraped too, but their advanced columns are now blank - FBref lost that
  data - so they are not used.)

Features (one row per Transfermarkt id), written to data/interim/underlying_b5.csv:
  u_<metric>      multi-season per-90 rate (weights: this season 1.0, then 0.75, 0.5, 0.3 by minutes), attacking
                  output converted to Premier League terms with league factors learned from players who moved,
                  shrunk toward the role average with a 6-match pseudo-sample
  p_<metric>      percentile of that rate within the player's role (GK, CB, FB, DM, CM, AM, W, CF)
  role_score      mean percentile of the metrics that matter for the role
  trend_*         slope of the role's key output per season (minutes-weighted) and this season vs the pooled past
  age_curve       expected change in that output next season at his age, learned from consecutive seasons
  team_xg/xga     team xG for/against per match (this season + last), involvement = player's xG+xA share
  proj_xgxa90     two-step model: next season's xG+xA per 90 predicted from everything above
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, INTERIM = ROOT / "data/raw", ROOT / "data/interim"
SEASONS = ["24", "25", "26", "27"]                       # FBref season keys (2023-24 .. 2026-27)
FM_SEASON = {"23": "2022/2023", "24": "2023/2024", "25": "2024/2025", "26": "2025/2026", "27": "2026/2027"}
W_SEASON = {"27": 1.0, "26": 0.75, "25": 0.5, "24": 0.3, "23": 0.2}
K = 6.0
ROLE = {"Goalkeeper": "GK", "Centre-Back": "CB", "Left-Back": "FB", "Right-Back": "FB", "Defensive Midfield": "DM",
        "Central Midfield": "CM", "Left Midfield": "CM", "Right Midfield": "CM", "Midfielder": "CM",
        "Attacking Midfield": "AM", "Second Striker": "AM", "Left Winger": "W", "Right Winger": "W", "Centre-Forward": "CF"}
FB_LEAGUE = {"eng Premier League": "GB1", "es La Liga": "ES1", "de Bundesliga": "L1", "it Serie A": "IT1", "fr Ligue 1": "FR1"}
FM_LEAGUE = {"premier_league": "GB1", "la_liga": "ES1", "bundesliga": "L1", "serie_a": "IT1", "ligue_1": "FR1"}

# metric -> (source table, column); all season totals
FB_METRICS = {"minutes": ("standard", "minutes"), "goals": ("standard", "goals"), "npg": ("standard", "goals_pens"),
              "pens_att": ("standard", "pens_att")}
# FotMob player keys (season files) -> our names; PER90 ones are published as per-90 and turned into totals
FM_BASE = {"expected_goals": "xg", "expected_assists": "xa", "total_scoring_att": "shots", "ontarget_scoring_att": "sot",
           "big_chance_created": "bcc", "total_att_assist": "chances", "won_contest": "dribbles", "accurate_pass": "passes",
           "accurate_long_balls": "long_balls", "big_chance_missed": "bcm", "total_tackle": "tackles", "interception": "interceptions",
           "effective_clearance": "clearances", "outfielder_block": "blocks", "ball_recovery": "recoveries",
           "defensive_contributions": "def_contrib", "poss_won_att_3rd": "press_wins"}
PER90 = {"total_scoring_att", "ontarget_scoring_att", "won_contest", "accurate_pass", "accurate_long_balls", "total_tackle",
         "interception", "effective_clearance", "outfielder_block", "ball_recovery", "defensive_contributions", "poss_won_att_3rd"}
FM_DEEP = {"xgot": "expected_goalsontarget", "pen_won": "penalty_won", "fm_goals_prevented": "_goals_prevented"}
FM_METRICS = list(FM_BASE.values()) + list(FM_DEEP)
# per-90 rates the model sees, and which are "attacking output" (converted between leagues)
RATES = ["npxg", "xa", "xgot_minus_xg", "npg", "shots", "sot", "chances", "bcc", "dribbles", "pen_won", "passes", "long_balls",
         "tackles", "interceptions", "clearances", "blocks", "recoveries", "def_contrib", "press_wins"]
ATTACK = {"npxg", "xa", "npg", "shots", "sot", "chances", "bcc"}
RATIOS = {"xg_per_shot": ("xg", "shots", 0.10, 8)}   # numerator, denominator, prior, pseudo-count
ROLE_KEYS = {   # what each role is judged on (percentiles averaged into role_score)
    "CF": ["npxg", "npg", "sot", "xgot_minus_xg", "xa", "xg_per_shot"],
    "W": ["npxg", "xa", "chances", "dribbles", "bcc", "sot"],
    "AM": ["xa", "chances", "bcc", "npxg", "dribbles", "passes"],
    "CM": ["passes", "long_balls", "chances", "xa", "recoveries", "tackles"],
    "DM": ["tackles", "interceptions", "recoveries", "passes", "long_balls", "def_contrib"],
    "FB": ["xa", "chances", "tackles", "interceptions", "recoveries", "dribbles"],
    "CB": ["clearances", "interceptions", "blocks", "def_contrib", "long_balls", "passes"],
    "GK": ["goals_prevented90", "long_balls", "passes"],
}
OUTPUT = "xgxa90"   # the composite used for trends, ageing curves and the projection (npxG + xA per 90)


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def fb_tables():
    """Per FBref id and season: summed totals of FB_METRICS, team, league."""
    srcs = {}
    for f in ["fbref_2026.json", "fbref_big5_2026.json", "fbref_deep.json"]:
        p = RAW / f
        if p.exists():
            srcs.update({k: v for k, v in json.load(open(p)).items() if isinstance(v, dict) and "rows" in v})
    srcs.setdefault("b526_standard", srcs.get("b5_standard")); srcs.setdefault("b526_shooting", srcs.get("b5_shooting"))
    out = []
    for s in SEASONS:
        merged = None
        for kind in sorted({t for t, _ in FB_METRICS.values()}):
            t = srcs.get(f"b5{s}_{kind}")
            if t is None:
                continue
            df = pd.DataFrame([r[:len(t["cols"])] for r in t["rows"]], columns=t["cols"][:len(t["rows"][0])])
            df["fbref_id"] = df["player"].str.split("|").str[1]
            cols = {m: c for m, (tk, c) in FB_METRICS.items() if tk == kind and c in df}
            keep = df[["fbref_id"] + list(cols.values())].copy()
            for c in cols.values():
                keep[c] = keep[c].map(num)
            keep = keep.rename(columns={v: k for k, v in cols.items()})
            if kind == "standard":
                keep["team"] = df["team"]; keep["league"] = df["comp_level"].map(FB_LEAGUE)
            agg = {c: "sum" for c in keep.columns if c not in ("fbref_id", "team", "league")}
            agg.update({c: "last" for c in ("team", "league") if c in keep})
            keep = keep.groupby("fbref_id").agg(agg).reset_index()
            merged = keep if merged is None else merged.merge(keep, on="fbref_id", how="outer")
        merged["season"] = s
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def fm_tables():
    """Per FotMob id and season: totals of FM_METRICS, minutes, rating, team, league; team xG/xGA/possession per match."""
    seasons = {}
    for f in ("fotmob_hist.json", "fotmob_2026.json"):
        for key, e in json.load(open(RAW / f)).items():
            lg, season = key.split("|")
            if lg in FM_LEAGUE and "/" in season:
                seasons[key] = e
    deep = json.load(open(RAW / "fotmob_deep.json"))
    rows, teams = [], []
    for key, e in seasons.items():
        lg, season = key.split("|"); s = season[-2:]
        dp = deep.get(key, {"p": {}, "t": {}})
        dstat = {m: {str(r[0]): r[3] for r in dp["p"].get(k, [])} for m, k in FM_DEEP.items()}
        for pid, p in e["players"].items():
            mn = p.get("min") or 0
            if mn <= 0:
                continue
            r = {"fm_id": str(pid), "season": s, "fm_min": mn, "fm_team": p.get("t"), "league": FM_LEAGUE[lg], "fm_rating_s": p.get("rating")}
            for k, m in FM_BASE.items():
                v = p.get(k, 0) or 0
                r[m] = v * mn / 90 if k in PER90 else v
            for m in FM_DEEP:
                r[m] = dstat[m].get(str(pid), 0.0) or 0.0
            rows.append(r)
        t = {k: {r[1]: (r[2], r[4]) for r in dp["t"].get(k, [])} for k in dp["t"]}
        for team, (xg, mp) in t.get("expected_goals_team", {}).items():
            teams.append({"season": s, "fm_team": team, "team_xg_pm": xg / max(mp, 1),
                          "team_xga_pm": t.get("expected_goals_conceded_team", {}).get(team, (np.nan, 1))[0] / max(mp, 1),
                          "team_poss": t.get("possession_percentage_team", {}).get(team, (np.nan, 1))[0]})
    return pd.DataFrame(rows), pd.DataFrame(teams)


def league_factors(fm):
    """Attacking output conversion between leagues, learned from players who changed league between consecutive
    seasons (>=900 minutes in both). log(xG+xA per 90 after / before) = f[to] - f[from] + age effect."""
    fm = fm.copy()
    fm["o90"] = (fm.xg + fm.xa) / (fm.fm_min / 90).clip(lower=1e-9)
    fm["sn"] = fm.season.astype(int)
    a = fm[fm.fm_min >= 900][["fm_id", "sn", "league", "o90"]]
    b = a.copy(); b["sn"] -= 1
    pairs = a.merge(b, on=["fm_id", "sn"], suffixes=("_0", "_1"))
    pairs = pairs[(pairs.o90_0 > 0.05) & (pairs.o90_1 > 0.05)]
    movers = pairs[pairs.league_0 != pairs.league_1]
    leagues = ["ES1", "L1", "IT1", "FR1"]                     # Premier League = 0
    X = np.array([[int(r.league_1 == l) - int(r.league_0 == l) for l in leagues] for r in movers.itertuples()], float)
    y = np.log(movers.o90_1 / movers.o90_0).to_numpy() - np.log(pairs.o90_1 / pairs.o90_0).median()   # net of the typical drift
    lam = 5.0                                                  # ridge toward 0 (few movers per pair)
    f = np.linalg.solve(X.T @ X + lam * np.eye(len(leagues)), X.T @ y)
    return {"GB1": 0.0, **dict(zip(leagues, f))}, len(movers)


def build():
    cur = pd.read_csv(INTERIM / "players_b5.csv", low_memory=False)
    cur["id"] = cur["id"].astype(str)
    cur["role"] = cur["position"].map(ROLE).fillna("CM")
    fb = fb_tables()
    fm, teams = fm_tables()
    lf, n_movers = league_factors(fm)
    print("league factors (log, attacking output in that league vs PL; >0 = easier to produce):",
          {k: round(v, 3) for k, v in lf.items()}, "from", n_movers, "moves")
    fm = fm.merge(teams, on=["season", "fm_team"], how="left")

    # long table: one row per (TM player, season) with FBref + FotMob totals
    link_fb = cur[["id", "fbref_id"]].dropna()
    link_fm = cur[["id", "fm_id"]].dropna(); link_fm["fm_id"] = link_fm["fm_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    L = link_fb.merge(fb, on="fbref_id").drop_duplicates(["id", "season"])
    F = link_fm.merge(fm, on="fm_id").rename(columns={"league": "fm_league"})
    # a player at two clubs in one season: sum totals, minutes-weight rating/team context, keep the main club
    F = F.sort_values("fm_min", ascending=False)
    for c in ("fm_rating_s", "team_xg_pm", "team_xga_pm", "team_poss"):
        F[c + "_x"] = F[c] * F["fm_min"]
    agg = {c: "sum" for c in list(FM_METRICS) + ["fm_min"] + [c + "_x" for c in ("fm_rating_s", "team_xg_pm", "team_xga_pm", "team_poss")]}
    agg.update({"fm_team": "first", "fm_league": "first", "fm_id": "first"})
    F["rated_min"] = F["fm_min"].where(F["fm_rating_s"].notna(), 0); agg["rated_min"] = "sum"
    F = F.groupby(["id", "season"]).agg(agg).reset_index()
    F["fm_rating_s"] = F.pop("fm_rating_s_x") / F["rated_min"].replace(0, np.nan)
    for c in ("team_xg_pm", "team_xga_pm", "team_poss"):
        F[c] = F.pop(c + "_x") / F["fm_min"].replace(0, np.nan)
    long = L.merge(F, on=["id", "season"], how="outer")
    long["minutes"] = long["minutes"].fillna(long["fm_min"])
    long = long[long.minutes.fillna(0) > 0].copy()
    for c in FB_METRICS:
        if c not in long:
            long[c] = np.nan
    long["league"] = long["league"].fillna(long["fm_league"])
    long["npxg"] = (long["xg"] - 0.76 * long["pens_att"].fillna(0)).clip(lower=0)
    long["xgot_minus_xg"] = long["xgot"] - long["xg"]
    # possession-adjusted defending: a defender on a team that has the ball 65% of the time faces 35% opposition
    # possession, so his raw tackles/interceptions/clearances undercount; scale to a 50/50 game (PAdj)
    padj = (50 / (100 - long["team_poss"])).clip(0.7, 1.5).fillna(1.0)
    for m in ("tackles", "interceptions", "clearances", "blocks", "recoveries", "def_contrib"):
        long[m] = long[m] * padj
    long["w"] = long["season"].map(W_SEASON)
    conv = np.exp(-long["league"].map(lf).fillna(0))           # PL-equivalent attacking output
    for m in ATTACK:
        if m in long:
            long[m + "_pl"] = long[m] * conv
    long["n90"] = long["minutes"] / 90

    # ---------- pooled, shrunk per-90 rates (toward role average)
    role_of = dict(zip(cur.id, cur.role))
    long["role"] = long["id"].map(role_of)
    g = long.assign(wn90=long.w * long.n90)
    feats = pd.DataFrame({"id": cur["id"]}).set_index("id")
    pooled_n90 = g.groupby("id").wn90.sum()
    feats["u_pooled_90s"] = pooled_n90
    for m in RATES:
        col = m + "_pl" if m in ATTACK else m
        if col not in g:
            continue
        tot = (g.w * g[col].fillna(0)).groupby(g.id).sum()
        has = g[col].notna().groupby(g.id).any()
        n = (g.wn90.where(g[col].notna(), 0)).groupby(g.id).sum()
        role_prior = (tot / n.replace(0, np.nan)).groupby(pd.Series(role_of)).median()
        prior = pd.Series(role_of).map(role_prior).reindex(tot.index)
        feats["u_" + m] = ((tot + K * prior) / (n + K)).where(has)
    for r, (a, b, pr0, k0) in RATIOS.items():
        na = (g.w * g[a].fillna(0)).groupby(g.id).sum(); nb = (g.w * g[b].fillna(0)).groupby(g.id).sum()
        feats["u_" + r] = ((na + pr0 * k0) / (nb + k0)).where(nb > 0)
    feats["u_xgxa"] = feats["u_npxg"] + feats["u_xa"]
    fmr = g.dropna(subset=["fm_rating_s"])
    feats["u_rating"] = ((fmr.fm_rating_s * fmr.w * fmr.n90).groupby(fmr.id).sum() + 6.7 * K) / ((fmr.w * fmr.n90).groupby(fmr.id).sum() + K)
    feats["u_goals_prevented90"] = (g.w * g.fm_goals_prevented.fillna(0)).groupby(g.id).sum() / (g.wn90.groupby(g.id).sum() + K)

    # ---------- percentiles within role, role score
    feats = feats.join(cur.set_index("id")[["role", "age", "pos_group"]] if "pos_group" in cur else cur.set_index("id")[["role", "age"]])
    pct_cols = [c for c in feats.columns if c.startswith("u_") and c not in ("u_pooled_90s",)]
    enough = feats["u_pooled_90s"] >= 5
    for c in pct_cols:
        feats["p_" + c[2:]] = feats[c].where(enough).groupby(feats["role"]).rank(pct=True)
    feats["role_score"] = [np.nanmean([feats.at[i, "p_" + k] for k in ROLE_KEYS[r] if "p_" + k in feats]) if enough[i] else np.nan
                           for i, r in zip(feats.index, feats.role)]

    # ---------- trends and ageing curve (composite: PL-equivalent npxG + xA per 90; role score proxy for defenders)
    long["o90"] = (long["npxg_pl"].fillna(0) + long["xa_pl"].fillna(0)) / long.n90.clip(lower=1e-9)
    ok = long[long.minutes >= 450]
    sn = ok.season.astype(int)
    def slope(df):
        if len(df) < 2:
            return np.nan
        x = df.season.astype(int).to_numpy(float); y = df.o90.to_numpy(); w = df.minutes.to_numpy()
        xm = np.average(x, weights=w); return float(np.sum(w * (x - xm) * (y - np.average(y, weights=w))) / max(np.sum(w * (x - xm) ** 2), 1e-9))
    feats["trend_xgxa90"] = ok.groupby("id").apply(slope)
    curr = ok[ok.season == "27"].set_index("id").o90
    past = ok[ok.season != "27"].groupby("id").apply(lambda d: np.average(d.o90, weights=d.minutes))
    feats["form_vs_past"] = curr - past
    # ageing curve: change in composite between consecutive full seasons by age band and role
    dob_age = dict(zip(cur.id, cur.age))
    a = long[(long.minutes >= 900) & (long.season != "27")][["id", "season", "o90"]].copy(); a["sn"] = a.season.astype(int)
    b = a.copy(); b["sn"] -= 1
    pr = a.merge(b, on=["id", "sn"], suffixes=("_0", "_1"))
    pr["age_then"] = pr["id"].map(dob_age) - (27 - pr["sn"])                       # approx age in season sn
    pr["role"] = pr["id"].map(role_of)
    pr["d"] = pr.o90_1 - pr.o90_0
    pr["ab"] = pr.age_then.clip(17, 35).round()
    curve = pr.groupby("ab").d.median().rolling(3, center=True, min_periods=1).mean()
    feats["age_curve"] = feats["age"].clip(17, 35).round().map(curve)
    print("ageing curve (median change in npxG+xA per 90 next season):", curve.round(3).to_dict())

    # ---------- team context (FotMob, this season + last)
    tm = long[long.season.isin(["27", "26"])].dropna(subset=["team_xg_pm"])
    tw = tm.w * tm.n90
    feats["team_xg_pm"] = (tm.team_xg_pm * tw).groupby(tm.id).sum() / tw.groupby(tm.id).sum()
    feats["team_xga_pm"] = (tm.team_xga_pm * tw).groupby(tm.id).sum() / tw.groupby(tm.id).sum()
    feats["team_poss"] = (tm.team_poss * tw).groupby(tm.id).sum() / tw.groupby(tm.id).sum()
    feats["involvement"] = feats["u_xgxa"] / feats["team_xg_pm"]

    # season-by-season numbers for the site's scouting card (actual, not league-converted)
    ser = {}
    for r in long.sort_values("season").itertuples():
        if r.minutes and r.minutes >= 90:
            ser.setdefault(r.id, []).append([int(r.season), int(r.minutes), round(float(r.npxg / r.n90), 3) if r.npxg == r.npxg else None,
                                             round(float(r.xa / r.n90), 3) if r.xa == r.xa else None, r.league if isinstance(r.league, str) else None])
    json.dump(ser, open(INTERIM / "underlying_series.json", "w"), separators=(",", ":"))

    # ---------- two-step: project next season's npxG+xA per 90 (trained on 23-24->24-25 and 24-25->25-26 pairs)
    proj = projection(long, dob_age, role_of)
    feats["proj_xgxa90"] = proj
    feats = feats.drop(columns=[c for c in ("role", "age", "pos_group") if c in feats])
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats.reset_index().to_csv(INTERIM / "underlying_b5.csv", index=False)
    print("players", len(feats), "with pooled minutes", int(feats.u_pooled_90s.notna().sum()),
          "| with xG", int(feats.u_npxg.notna().sum()), "| with defensive actions", int(feats.u_def_contrib.notna().sum()))
    return feats


def projection(long, dob_age, role_of):
    """LightGBM: next season's npxG+xA per 90 (>=900 min) from the previous seasons' per-90 profile, age, role."""
    import lightgbm as lgb
    long = long.copy(); long["sn"] = long.season.astype(int)
    prof = ["o90", "npxg_pl", "xa_pl", "chances_pl", "bcc_pl", "shots_pl", "sot_pl", "dribbles", "passes", "recoveries"]
    for c in prof[1:]:
        if c in long:
            long[c + "_90"] = long[c] / long.n90.clip(lower=1e-9)
    cols = ["o90"] + [c + "_90" for c in prof[1:] if c + "_90" in long]

    def snapshot(upto):
        h = long[long.sn <= upto]
        w = h.sn.map(lambda s: {0: 1.0, 1: 0.75, 2: 0.5}.get(upto - s, 0.3)) * h.n90
        X = pd.DataFrame({c: (h[c] * w).groupby(h.id).sum() / w.groupby(h.id).sum() for c in cols})
        X["n90"] = h[h.sn == upto].groupby("id").n90.sum().reindex(X.index).fillna(0)
        X["n90_all"] = h.groupby("id").n90.sum()
        X["age"] = X.index.map(dob_age) - (27 - upto)
        for r in ["CF", "W", "AM", "CM", "DM", "FB", "CB"]:
            X["r_" + r] = (X.index.map(role_of) == r).astype(int)
        return X
    Xs, ys = [], []
    for upto in (24, 25):
        X = snapshot(upto)
        nxt = long[(long.sn == upto + 1) & (long.minutes >= 900)].set_index("id").o90
        X = X.join(nxt.rename("y"), how="inner")
        Xs.append(X.drop(columns="y")); ys.append(X["y"])
    X, y = pd.concat(Xs), pd.concat(ys)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=20, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(X, y)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=20, verbose=-1), X, y, cv=5, scoring="r2")
    naive = 1 - np.mean((y - X["o90"]) ** 2) / np.var(y)
    print(f"projection: {len(X)} season pairs, CV R2 {cv.mean():.3f} (repeat-last-season R2 {naive:.3f})")
    now = snapshot(27)
    return pd.Series(m.predict(now[X.columns]), index=now.index)


if __name__ == "__main__":
    build()
