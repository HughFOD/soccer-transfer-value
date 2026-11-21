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
FULL_FS, STATS_FS = "full_iy", "stats_xg"         # v17: + injury and youth-international features
oof = pd.read_pickle(INTERIM / f"oof_allpos_{FULL_FS}_change_et.pkl")
# v17: players aged 21 or under use the youth model (today + past summers weighted 0.3); OOF from src/aug_eval.py
_c, _y, _res = pd.read_pickle(INTERIM / "aug_eval.pkl")
_youth_oof = dict(zip(_c["id"].astype(str), _res["E1w0.3"]))
YOUTH_AGE = 21
for k in ("lgbm", "ridge", "et", "blend"):
    oof[k] = oof[k].copy()
_ym = ((df.age <= YOUTH_AGE) & df["id"].astype(str).isin(_youth_oof)).to_numpy()
oof["blend"][_ym] = [_youth_oof[i] for i in df["id"].astype(str)[_ym]]
df["model_used"] = np.where(_ym, "youth", np.where(df.prev_val.notna(), "full", "level"))
oof_stats = pd.read_pickle(INTERIM / f"oof_allpos_{STATS_FS}.pkl")
fwd = np.arange(len(df))                      # every player is scored now
y = np.log(df.market_value_eur.to_numpy())
CHOSEN = "blend"   # 50/50 LightGBM + ridge on log value

POS3 = ["pos_DF", "pos_MF", "pos_GK"]
models = {}
for name, feats in [("full", T.FEATURE_SETS[FULL_FS] + POS3), ("youth", A.AUG), ("stats_only", T.FEATURE_SETS[STATS_FS] + POS3)]:
    X, yy = df[feats], np.log(df.market_value_eur.to_numpy())
    W = None
    if name in ("full", "youth"):   # learn log(value / previous valuation) on players that have one
        has = df.prev_val.notna().to_numpy()
        X, yy = X[has], yy[has] - np.log(df.prev_val.to_numpy()[has])
    if name == "youth":             # plus the past-season snapshots, weighted 0.3
        X = pd.concat([X, HIST[feats]])
        yy = np.concatenate([yy, np.log(HIST.market_value_eur.to_numpy()) - np.log(HIST.prev_val.to_numpy())])
        W = np.concatenate([np.ones(int(has.sum())), np.full(len(HIST), 0.3)])
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
        "peak_before", "mv_date", "model_used", "injured_now", "days_out_season", "youth_caps", "pos_group", "xg_p90", "xa_p90", "def_contrib_p90", "fm_rating"]
players = f[cols].astype(object).where(f[cols].notna(), None)
players["age"] = players["age"].map(lambda a: round(a, 1))
full_feats = list(dict.fromkeys(models["full"]["features"] + models["stats_only"]["features"] + models["youth"]["features"]))   # union for the calculator
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
players["fvec"] = [[None if pd.isna(v) else float(f"{float(v):.5g}") for v in row] for row in f[full_feats].to_numpy()]

# club context for the calculator (from every player at the club)
club = df.groupby("club").agg(club_log_mv=("log_mv", "mean"), team_gd_per90=("team_gd_per90", "median"),
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
             "max_cur_minutes": float(df.c_minutes.max()), "priors": priors, "medians": med,
             "max_minutes_cur": float(T.df.min_cur.max()), "median_fvec": med_vec},
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
