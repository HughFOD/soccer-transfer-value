"""Fit final models on all data and export everything the site needs.

out/site_data.json : every player (all positions) with OOF predictions, per-position metrics, importances,
                     model comparison, club context, calculator priors + both models (trees + ridge).
v14 (2026-11-20): all positions; OOF from src/allpos.py (folds over all players); stats-only model = stats_xg.
"""
import json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import train as T  # noqa: E402
import aug_eval as A  # noqa: E402   (v17 youth model: past-season rows + AUG features)

ROOT = Path(__file__).resolve().parents[1]
INTERIM, OUT = ROOT / "data/interim", ROOT / "out"
CUR, HIST = A.load()
df = CUR                                         # T.df plus the AUG columns (same rows, same order)
import os
# v19: env-driven so the five-league build (INPUT=players_b5.csv HIST=hist_rows_b5.csv) shares this script
FULL_FS = os.environ.get("FULL_FS", "full_iy")       # v17: + injury and youth-international features
STATS_FS = os.environ.get("STATS_FS", "stats_xg")
TAGS = "_" + os.environ["TAG"] if os.environ.get("TAG") else ""
oof = pd.read_pickle(INTERIM / f"oof_allpos_{FULL_FS}_change_et{os.environ.get('FULL_TAG', TAGS)}.pkl")
# v17: players aged 21 or under use the youth model (today + past summers weighted 0.3); OOF from src/aug_eval.py
V19 = os.environ.get("V19") == "1"
for k in ("lgbm", "ridge", "et", "blend"):
    oof[k] = oof[k].copy()
if V19:
    # v19: one model trained on all five leagues (past seasons and the youth model no longer helped: src/aug_eval.py).
    # Premier League players: 50/50 average with the v17 Premier League-only estimate (out-of-fold, same players):
    # PL median error 9.1% (v17) / 9.4% (five leagues) -> 8.5% averaged; mean 13.7 / 13.8 -> 13.4.
    YOUTH_AGE = -1
    v17 = pd.read_csv(INTERIM / "players_all.csv", low_memory=False)
    v17 = v17[v17.market_value_eur.notna()].reset_index(drop=True)
    o17 = pd.read_pickle(INTERIM / "oof_allpos_full_iy_change_et.pkl")["blend"].copy()
    _c, _y, _res = pd.read_pickle(INTERIM / "aug_eval.pkl")
    _yd = dict(zip(_c["id"].astype(str), _res["E1w0.3"]))
    _m = ((v17.age <= 21) & v17["id"].astype(str).isin(_yd)).to_numpy()
    o17[_m] = [_yd[i] for i in v17["id"].astype(str)[_m]]
    pl17 = dict(zip(v17["id"].astype(str), o17))
    _pl = (df.league == "GB1").to_numpy() & df["id"].astype(str).isin(pl17).to_numpy()
    oof["blend"][_pl] = 0.5 * oof["blend"][_pl] + 0.5 * np.array([pl17[i] for i in df["id"].astype(str)[_pl]])
    df["model_used"] = np.where(_pl, "full+pl", np.where(df.prev_val.notna(), "full", "level"))
else:
    _c, _y, _res = pd.read_pickle(INTERIM / f"aug_eval{TAGS}.pkl")
    _youth_oof = dict(zip(_c["id"].astype(str), _res["E1w0.3"]))
    YOUTH_AGE = 21
    _ym = ((df.age <= YOUTH_AGE) & df["id"].astype(str).isin(_youth_oof)).to_numpy()
    oof["blend"][_ym] = [_youth_oof[i] for i in df["id"].astype(str)[_ym]]
    df["model_used"] = np.where(_ym, "youth", np.where(df.prev_val.notna(), "full", "level"))
YOUTH_W = 0.3
if os.environ.get("SF_OOF"):   # v20: stats view = deep-metric stats-first model (src/stats_first.py OOF)
    _sf = pd.read_pickle(INTERIM / os.environ["SF_OOF"])
    _sfd = dict(zip(_sf["ids"], _sf["oof"]))
    oof_stats = {"blend": np.array([_sfd[i] for i in df["id"].astype(str)])}
else:
    oof_stats = pd.read_pickle(INTERIM / f"oof_allpos_{STATS_FS}{TAGS}.pkl")
fwd = np.arange(len(df))                      # every player is scored now
y = np.log(df.market_value_eur.to_numpy())
CHOSEN = "blend"   # 50/50 LightGBM + ridge on log value

POS3 = ["pos_DF", "pos_MF", "pos_GK"]
models = {}
_specs = [("full", T.FEATURE_SETS[FULL_FS] + POS3), ("youth", A.AUG), ("stats_only", T.FEATURE_SETS[STATS_FS] + POS3)]
if V19:
    _specs = [x for x in _specs if x[0] != "youth"]
for name, feats in _specs:
    X, yy = df[feats], np.log(df.market_value_eur.to_numpy())
    W = None
    if name in ("full", "youth"):   # learn log(value / previous valuation) on players that have one
        has = df.prev_val.notna().to_numpy()
        X, yy = X[has], yy[has] - np.log(df.prev_val.to_numpy()[has])
    if name == "youth":             # plus the past-season snapshots, weighted 0.3
        X = pd.concat([X, HIST[feats]])
        yy = np.concatenate([yy, np.log(HIST.market_value_eur.to_numpy()) - np.log(HIST.prev_val.to_numpy())])
        W = np.concatenate([np.ones(int(has.sum())), np.full(len(HIST), YOUTH_W)])
    p = T.tune(X.reset_index(drop=True), yy, seed=7)
    m = T.lgbm(p).fit(X, yy, **({} if W is None else {"sample_weight": W}))
    imp = pd.Series(m.booster_.feature_importance("gain"), index=feats)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    dump = m.booster_.dump_model()

    def slim(node):
        if "leaf_value" in node:
            return {"v": node["leaf_value"]}
        return {"f": node["split_feature"], "t": node["threshold"], "dl": node["default_left"], "iv": node.get("internal_value", 0.0),
                "mt": node["missing_type"], "l": slim(node["left_child"]), "r": slim(node["right_child"])}
    rp = T.ridge().fit(X, yy, **({} if W is None else {"ridgecv__sample_weight": W}))
    imp_, sc_, rg_ = rp.steps[0][1], rp.steps[1][1], rp.steps[2][1]
    ridge_exp = {"impute": [float(v) for v in imp_.statistics_], "ind": [int(i) for i in imp_.indicator_.features_],
                 "mean": sc_.mean_.tolist(), "scale": sc_.scale_.tolist(), "coef": rg_.coef_.tolist(),
                 "intercept": float(rg_.intercept_), "alpha": float(rg_.alpha_)}
    et_exp = None
    if name in ("full", "youth"):   # v16: third model, Extra Trees (same settings as src/allpos.py), exported as flat arrays
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        ep = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                           ExtraTreesRegressor(n_estimators=150, min_samples_leaf=4, max_features=0.5, n_jobs=2, random_state=0)).fit(X, yy, **({} if W is None else {"extratreesregressor__sample_weight": W}))
        ei, ef = ep.steps[0][1], ep.steps[1][1]
        trees = []
        for est in ef.estimators_:
            t = est.tree_
            trees.append({"f": t.feature.tolist(), "t": [round(float(v), 4) for v in t.threshold],
                          "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                          "v": [round(float(v), 4) for v in t.value[:, 0, 0]]})
        et_exp = {"impute": [float(v) for v in ei.statistics_], "ind": [int(i) for i in ei.indicator_.features_], "trees": trees,
                  "check": ep.predict(X.iloc[:5]).tolist()}
    models[name] = {"et": et_exp, "target": "change" if name in ("full", "youth") else "level", "features": feats, "params": p, "trees": [slim(t["tree_structure"]) for t in dump["tree_info"]], "ridge": ridge_exp,
                    "importance": [[k, round(float(v), 4)] for k, v in imp.items()],
                    "check": {"x": X.iloc[:5].to_numpy().tolist(), "pred": m.predict(X.iloc[:5]).tolist(),
                              "ridge": rp.predict(X.iloc[:5]).tolist()}}
    print(name, p, "top:", imp.head(8).round(3).to_dict())

f = df.loc[fwd].copy()
f["pred_full"] = np.exp(oof[CHOSEN])
f["pred_stats"] = np.exp(oof_stats[CHOSEN])
f["gap_pct"] = 100 * (f.pred_full - f.market_value_eur) / f.market_value_eur
cols = ["name", "club", "position", "age", "market_value_eur", "pred_full", "pred_stats", "gap_pct",
        "mv_1y_ago", "mv_peak_1y_ago", "annual_wage_eur", "contract_years_left", "international_caps",
        "min_cur", "min_prev", "prev_season_league", "c_goals", "c_assists", "p_goals", "p_assists",
        "ga_per90", "sh_per90", "height_cm", "foot", "id", "nat",
        "starts_cur", "starts_prev", "c_shots", "p_shots", "c_shots_on_target", "p_shots_on_target",
        "years_at_club", "years_since_extension", "senior_intl", "current_intl", "english", "intl_goals",
        "mv_2y_ago", "is_winger", "actual_value_reported", "prev_val", "months_since_prev", "val_12m_before",
        "peak_before", "mv_date", "model_used", "injured_now", "days_out_season", "youth_caps", "pos_group", "xg_p90", "xa_p90", "def_contrib_p90", "fm_rating", "league"]
players = f[cols].astype(object).where(f[cols].notna(), None)
players["age"] = players["age"].map(lambda a: round(a, 1))
full_feats = list(dict.fromkeys(sum((m["features"] for m in models.values()), [])))   # union for the calculator
# nominal (as-published) history for the calculator inputs
from datetime import date as _date
raw = {str(r["id"]): r["mvh"] for r in json.load(open(ROOT / "data/raw/transfermarkt_2026.json"))}
def nominal(pid):
    pts = sorted((datetime.strptime(d, "%d/%m/%Y").date(), v) for d, v, _ in raw.get(str(pid), []) if v)
    if len(pts) < 2:
        return None, None, None, None
    cur_d = pts[-1][0]; before = [(d, v) for d, v in pts[:-1] if d < cur_d]
    if not before:
        return None, None, None, None
    pd_, pv = before[-1]
    months = round((_date(2026, 11, 16) - pd_).days / 30.44, 1)
    anchor = _date(cur_d.year - 1, cur_d.month, min(cur_d.day, 28))
    old = [v for d, v in before if d <= anchor]
    return pv / 1e6, months, (old[-1] / 1e6 if old else None), max(v for _, v in before) / 1e6
nom = [nominal(i) for i in f["id"]]
players["prev_nom"], players["prev_months_ago"], players["v12_nom"], players["peak_nom"] = zip(*nom)
# v18: likely transfer fee = current TM value (today's money) x learned fee/value ratio
# (ratios were learned against the TM value at the time of each move, so the published value is the like-for-like base) (src/fee.py, 610 real transfers)
import pickle
FEE = pickle.load(open(ROOT / "data/interim/fee_model.pkl", "rb"))
def fee_ratio(val, age, tenure, seller_pl):
    x = np.c_[np.log(np.minimum(val, FEE["cap"])), age, (age - 25) ** 2, np.clip(tenure, 0, 8),
              np.ones(len(val)), seller_pl, np.full(len(val), 2026.9)]   # PL buyer, seller's league, today
    return (x - FEE["mu"]) / FEE["sd"] @ FEE["coef"] + FEE["b"]
_ten = f["years_at_club"].fillna(2.5).to_numpy()
_base = f.market_value_eur.to_numpy()
_lr = fee_ratio(_base, f.age.to_numpy(), _ten, (f.league == "GB1").astype(float).to_numpy())
_b = np.where(_base >= 20e6, 1, 0)
_q = np.array([FEE["bands"]["small"][:2], FEE["bands"]["big"][:2]])[_b]
players["fee_mid"] = _base * np.exp(_lr)
# ---- v20 scouting card: role percentiles, underlying rates, season series, projection, similar players
UND = INTERIM / "underlying_b5.csv"
if os.environ.get("DEEP") == "1" and UND.exists():
    import underlying as U
    from sklearn.neighbors import NearestNeighbors
    ROLE_OF = f["position"].map(U.ROLE).fillna("CM").to_numpy()
    players["role"] = ROLE_OF
    show_u = ["u_npxg", "u_xa", "u_xgot_minus_xg", "u_npg", "u_shots", "u_sot", "u_xg_per_shot", "u_chances", "u_bcc", "u_dribbles",
              "u_passes", "u_long_balls", "u_tackles", "u_interceptions", "u_clearances", "u_blocks", "u_recoveries", "u_def_contrib",
              "u_press_wins", "u_goals_prevented90", "u_rating"]
    for c in show_u + ["role_score", "proj_xgxa90", "trend_xgxa90", "team_xg_pm", "team_xga_pm", "involvement", "u_pooled_90s"]:
        players[c] = [None if pd.isna(v) else round(float(v), 4) for v in f[c]]
    pcols = sorted({"p_" + k for ks in U.ROLE_KEYS.values() for k in ks} | {"p_npxg", "p_xa", "p_chances", "p_dribbles", "p_passes", "p_tackles", "p_recoveries", "p_press_wins"})
    pcols = [c for c in pcols if c in f]
    players["pct"] = [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in f[pcols].to_numpy()]
    ser = json.load(open(INTERIM / "underlying_series.json"))
    players["series"] = [ser.get(str(i), []) for i in f["id"]]
    # similar players: same role, closest role-percentile profile + age (display only; not a model input)
    # similarity on the metrics that matter for the role (role keys), using the underlying per-90 rates (standardised
    # within role) so elite outliers match other elite players, plus age
    sim = [None] * len(f)
    for r in np.unique(ROLE_OF):
        ix = np.where(ROLE_OF == r)[0]
        cols = ["u_" + k for k in U.ROLE_KEYS[r] if "u_" + k in f]
        Z = f.iloc[ix][cols].astype(float)
        Z = ((Z - Z.mean()) / Z.std().replace(0, 1)).fillna(-1.0).to_numpy()
        Z = np.c_[Z, (f["age"].to_numpy()[ix] - 26) / 4]
        nn = NearestNeighbors(n_neighbors=min(6, len(ix))).fit(Z)
        _, nb = nn.kneighbors(Z)
        for a, row in zip(ix, nb):
            sim[a] = [int(ix[j]) for j in row if ix[j] != a][:5]
    players["similar"] = sim
    DEEP_META = {"pct_cols": pcols, "role_keys": U.ROLE_KEYS}
else:
    DEEP_META = None
players["fee_lo"] = _base * np.exp(_lr + _q[:, 0])
players["fee_hi"] = _base * np.exp(_lr + _q[:, 1])
players["years_at_club"] = f["years_at_club"].round(2).to_numpy()
players["fvec"] = [[None if pd.isna(v) else float(f"{float(v):.5g}") for v in row] for row in f[full_feats].to_numpy()]

# club context for the calculator (from every player at the club)
club = df.groupby("club").agg(league=("league", "first"), club_log_mv=("log_mv", "mean"), team_gd_per90=("team_gd_per90", "median"),
                              ppg_cur=("ppg_cur", "median"), n=("name", "size")).reset_index()
fw = df[df.is_forward == 1]
nine = fw.c_minutes_90s.fillna(0) + T_W if (T_W := 0.75) else None
priors = {}
for feat, col in {"fouled_per90": "fouled", "crs_per90": "crosses", "int_per90": "interceptions",
                  "tklw_per90": "tackles_won", "fls_per90": "fouls", "offsides_per90": "offsides",
                  "penwon_per90": "pens_won", "gls_per90": "goals", "ast_per90": "assists",
                  "npg_per90": "goals_pens", "sh_per90": "shots", "sot_per90": "shots_on_target"}.items():
    tot = fw["c_" + col].fillna(0) + 0.75 * fw["p_" + col].fillna(0)
    n90 = fw.c_minutes_90s.fillna(0) + 0.75 * fw.p_minutes_90s.fillna(0)
    priors[col] = float(tot.sum() / n90.sum())
med = {c: float(fw[c].median()) for c in ["years_since_extension", "height_cm", "ppg_cur", "team_gd_per90",
                                          "minutes_share_cur", "intl_goals", "cards_per90"]}
med_vec = {g: [None if pd.isna(v) else float(v) for v in df[df.pos_group == g][full_feats].median().to_numpy()]
           for g in ["FW", "MF", "DF", "GK"]}

comp = pd.read_csv(ROOT / "out/cmp_r11.csv")
y_eur = np.exp(y)
ape_naive = ((f.prev_val - f.market_value_eur).abs() / f.market_value_eur).dropna()
ape_naive1y = ((f.mv_1y_ago - f.market_value_eur).abs() / f.market_value_eur).dropna()
by_pos = []
for g, label in [("FW", "Forwards"), ("MF", "Midfielders"), ("DF", "Defenders"), ("GK", "Goalkeepers")]:
    m = (df.pos_group == g).to_numpy()
    nv = ((df.prev_val - df.market_value_eur).abs() / df.market_value_eur)[m].dropna()
    by_pos.append(dict(group=g, label=label, n=int(m.sum()), **T.metrics(y[m], oof[CHOSEN][m]),
                       stats=T.metrics(y[m], oof_stats[CHOSEN][m]), naive_median_ape=round(100 * nv.median(), 1)))
site = {
    "deep": DEEP_META,
    "meta": {"snapshot": "2026-11-16", "n_players": int(len(f)), "n_training": int(len(df)), "by_pos": by_pos,
             "features_union": full_feats, "logged": "2026-11-20", "youth_age": YOUTH_AGE,
             # walk-forward test (src/aug_eval.py): trained only on earlier summers, predicting a later summer
             "walk_forward": [
                 {"train": "summer 2024", "test": "summer 2025", "n": 695, "model": {"median": 10.5, "mean": 14.6, "within25": 83.2}, "naive": {"median": 14.8, "mean": 18.7, "within25": 71.1}},
                 {"train": "summers 2024-25", "test": "summer 2026", "n": 679, "model": {"median": 9.4, "mean": 13.5, "within25": 83.9}, "naive": {"median": 16.3, "mean": 20.4, "within25": 66.9}}],
             "metrics": T.metrics(y, oof[CHOSEN]), "metrics_stats": T.metrics(y, oof_stats[CHOSEN]),
             "inflation": 0.10, "naive1y_median_ape": round(100 * ape_naive1y.median(), 1), "naive1y_mean_ape": round(100 * ape_naive1y.mean(), 1),
             "naive_within25": round(100 * (ape_naive <= 0.25).mean(), 1),
             "naive_median_ape": round(100 * ape_naive.median(), 1), "naive_mean_ape": round(100 * ape_naive.mean(), 1),
             "max_cur_minutes": float(df.c_minutes.max()),
             "max_min_by_league": {k: float(v) for k, v in df.groupby("league").min_cur.max().items()},
             "league_gap": {"GB1": 0.0, "ES1": 5.6, "IT1": 5.6, "L1": 6.3, "FR1": 7.1},
             "by_league": [dict(league=lg, n=int((df.league == lg).sum()), **T.metrics(y[(df.league == lg).to_numpy()], oof[CHOSEN][(df.league == lg).to_numpy()]),
                                naive_median_ape=round(100 * ((df.prev_val - df.market_value_eur).abs() / df.market_value_eur)[df.league == lg].median(), 1))
                           for lg in ["GB1", "ES1", "L1", "IT1", "FR1"] if (df.league == lg).any()], "priors": priors, "medians": med,
             "max_minutes_cur": float(T.df.min_cur.max()), "median_fvec": med_vec,
             "fee": {"mu": FEE["mu"].tolist(), "sd": FEE["sd"].tolist(), "coef": FEE["coef"].tolist(), "b": float(FEE["b"]),
                     "cap": FEE["cap"], "bands": FEE["bands"], "n": FEE["n"], "year": 2026.9}},
    "comparison": comp.to_dict("records"),
    "players": players.to_dict("records"),
    "clubs": club.round(4).to_dict("records"),
    "models": models,
}
OUT.mkdir(exist_ok=True)
def clean(o):
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o
json.dump(clean(site), open(OUT / "site_data.json", "w"), separators=(",", ":"), allow_nan=False)
print("players", len(players), "bytes", (OUT / "site_data.json").stat().st_size)
print(f.sort_values("market_value_eur", ascending=False)[["name", "market_value_eur", "pred_full", "pred_stats", "gap_pct"]].head(25).round(0).to_string())
