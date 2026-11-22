"""Out-of-fold scores for every position (folds over all 547 players), for deciding which positions the
site can cover. Same nested scheme as train.run: outer 5-fold x N repeats, inner 3-fold grid search."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

sys.argv = ["x"]
import train as T  # noqa: E402

fs = os.environ.get("FS", "full_all")
n_rep = int(os.environ.get("REPEATS", "3"))
feats = T.FEATURE_SETS[fs] + ["pos_DF", "pos_MF", "pos_GK"]
df = T.df
SUBSET = os.environ.get("SUBSET")            # e.g. "FW,MF": train and score only these position groups
if SUBSET:
    df = df[df.pos_group.isin(SUBSET.split(","))].reset_index(drop=True)
X, y = df[feats], np.log(df.market_value_eur.to_numpy())
# TARGET=change: learn log(value / previous valuation) where a previous valuation exists, then add it back.
# Players without one fall back to the level model.
TARGET = os.environ.get("TARGET", "level")
base = np.log(df.prev_val.to_numpy()) if TARGET == "change" else np.zeros(len(df))
has_base = np.isfinite(base)
base = np.where(has_base, base, 0.0)
oof = {"lgbm": np.zeros(len(df)), "ridge": np.zeros(len(df)), "et": np.zeros(len(df))}
USE_ET = os.environ.get("ET", "0") == "1"
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
def et():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         ExtraTreesRegressor(n_estimators=150, min_samples_leaf=4, max_features=0.5, n_jobs=2, random_state=0))
for i, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=n_rep, random_state=42).split(df)):
    if TARGET == "change":
        trc = tr[has_base[tr]]; yc = y - base
        p = T.tune(X.iloc[trc].reset_index(drop=True), yc[trc], 42 + i)
        lg, rg = T.lgbm(p).fit(X.iloc[trc], yc[trc]), T.ridge().fit(X.iloc[trc], yc[trc])
        pl = T.lgbm(T.tune(X.iloc[tr].reset_index(drop=True), y[tr], 42 + i)).fit(X.iloc[tr], y[tr])
        pr = T.ridge().fit(X.iloc[tr], y[tr])
        tb = te[has_base[te]]; tn = te[~has_base[te]]
        if USE_ET:
            oof["et"][tb] += (base[tb] + et().fit(X.iloc[trc], yc[trc]).predict(X.iloc[tb])) / n_rep
            if len(tn):
                oof["et"][tn] += et().fit(X.iloc[tr], y[tr]).predict(X.iloc[tn]) / n_rep
        oof["lgbm"][tb] += (base[tb] + lg.predict(X.iloc[tb])) / n_rep
        oof["ridge"][tb] += (base[tb] + rg.predict(X.iloc[tb])) / n_rep
        if len(tn):
            oof["lgbm"][tn] += pl.predict(X.iloc[tn]) / n_rep
            oof["ridge"][tn] += pr.predict(X.iloc[tn]) / n_rep
        continue
    p = T.tune(X.iloc[tr].reset_index(drop=True), y[tr], 42 + i)
    oof["lgbm"][te] += T.lgbm(p).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]) / n_rep
    oof["ridge"][te] += T.ridge().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]) / n_rep
oof["blend"] = (oof["lgbm"] + oof["ridge"] + oof["et"]) / 3 if USE_ET else 0.5 * oof["lgbm"] + 0.5 * oof["ridge"]
rows = []
groups = [g for g in ["FW", "MF", "DF", "GK"] if (df.pos_group == g).any()] + ["ALL"]
if df.league.nunique() > 1:   # v19: also score the Premier League players alone, comparable with PL-only training
    groups += ["PL_" + g for g in ["FW", "MF", "DF", "GK", "ALL"]] + [lg for lg in ["ES1", "L1", "IT1", "FR1"]]
for grp in groups:
    gg = grp.replace("PL_", "")
    m = np.ones(len(df), bool) if gg == "ALL" else (df.pos_group == gg).to_numpy() if gg in ("FW", "MF", "DF", "GK") else (df.league == grp).to_numpy()
    if grp.startswith("PL_"):
        m = m & (df.league == "GB1").to_numpy()
    rows.append(dict(group=grp, n=int(m.sum()), **T.metrics(y[m], oof["blend"][m])))
    print(rows[-1], flush=True)
naive = ((df.prev_val - df.market_value_eur).abs() / df.market_value_eur)
for grp in [g for g in ["FW", "MF", "DF", "GK"] if (df.pos_group == g).any()]:
    print(grp, "naive prev-valuation median APE", round(100 * naive[df.pos_group == grp].median(), 1))
tag = fs + ("_change" if TARGET == "change" else "") + ("_et" if USE_ET else "") + ("_" + SUBSET.replace(",", "") if SUBSET else "") + ("_" + os.environ["TAG"] if os.environ.get("TAG") else "")
for nm in ["Declan Rice", "William Saliba", "Erling Haaland", "Junior Kroupi", "Florian Wirtz", "Bukayo Saka"]:
    i = np.where(df.name.values == nm)[0]
    if len(i):
        print(nm, "TM", round(df.market_value_eur.values[i[0]] / 1e6, 1), "model", round(np.exp(oof["blend"][i[0]]) / 1e6, 1), flush=True)
pd.DataFrame(rows).to_csv(f"../out/allpos_{tag}.csv", index=False)
pd.to_pickle(oof, f"../data/interim/oof_allpos_{tag}.pkl")
