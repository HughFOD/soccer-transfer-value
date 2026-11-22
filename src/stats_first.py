"""v20: stats-first models (no Transfermarkt value history) on all five leagues, with nested CV.

FS=<feature set from train.FEATURE_SETS>  COMPS=1 adds comparable players (fold-aware: neighbours and their values
come only from the training fold).  Run with INPUT=players_b5.csv DEEP=1.

Reports median/mean error by league and position, and a yardstick for "is the stats view right?":
  does log(stats estimate / previous TM valuation) predict log(current TM value / previous valuation)?
  (the estimate never sees either valuation; a positive slope means TM moved toward what the stats said).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline

sys.argv = ["x"]
import train as T  # noqa: E402

FS = os.environ.get("FS", "stats_deep")
REPS = int(os.environ.get("REPEATS", "2"))
COMPS = os.environ.get("COMPS") == "1"
df = T.df.copy()
feats = list(dict.fromkeys(T.FEATURE_SETS[FS] + ["pos_DF", "pos_MF", "pos_GK"]))
y = np.log(df.market_value_eur.to_numpy())
ROLE = {"Goalkeeper": "GK", "Centre-Back": "CB", "Left-Back": "FB", "Right-Back": "FB", "Defensive Midfield": "DM",
        "Central Midfield": "CM", "Left Midfield": "CM", "Right Midfield": "CM", "Attacking Midfield": "AM",
        "Second Striker": "AM", "Left Winger": "W", "Right Winger": "W", "Centre-Forward": "CF"}
role = df.position.map(ROLE).fillna("CM").to_numpy()
PROF = [c for c in T.DEEP if c.startswith("p_")] + ["age"]
P = df[PROF].copy()
P[[c for c in PROF if c != "age"]] = P[[c for c in PROF if c != "age"]].fillna(0.5)
P["age"] = (P["age"] - 26) / 8


def comps(tr, te, k=5):
    """Median log TM value of the k most similar players of the same role (profile = role percentiles + age),
    neighbours drawn from the training rows only; training rows exclude themselves."""
    out_tr, out_te = np.full(len(tr), np.nan), np.full(len(te), np.nan)
    for r in np.unique(role):
        a, b = np.where(role[tr] == r)[0], np.where(role[te] == r)[0]
        if len(a) <= k + 1:
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(P.iloc[tr[a]].to_numpy())
        _, ia = nn.kneighbors(P.iloc[tr[a]].to_numpy())
        out_tr[a] = np.median(y[tr[a]][ia[:, 1:]], axis=1)
        if len(b):
            _, ib = nn.kneighbors(P.iloc[te[b]].to_numpy(), n_neighbors=k)
            out_te[b] = np.median(y[tr[a]][ib], axis=1)
    return out_tr, out_te


def et():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         ExtraTreesRegressor(n_estimators=150, min_samples_leaf=4, max_features=0.5, n_jobs=2, random_state=0))


oof = np.zeros(len(df))
for i, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=REPS, random_state=42).split(df)):
    Xtr, Xte = df.iloc[tr][feats].copy(), df.iloc[te][feats].copy()
    if COMPS:
        Xtr["comp_log_value"], Xte["comp_log_value"] = comps(tr, te)
    p = T.tune(Xtr.reset_index(drop=True), y[tr], 42 + i)
    preds = [T.lgbm(p).fit(Xtr, y[tr]).predict(Xte), T.ridge().fit(Xtr, y[tr]).predict(Xte), et().fit(Xtr, y[tr]).predict(Xte)]
    oof[te] += np.mean(preds, axis=0) / REPS
    print("fold", i, flush=True)


def st(m):
    a = np.abs(np.exp(oof[m] - y[m]) - 1)
    return f"median {100*np.median(a):.1f}  mean {100*a.mean():.1f}  within25 {100*(a<=.25).mean():.1f}  n {m.sum()}"


tag = FS + ("_comps" if COMPS else "")
print("==", tag)
print("ALL", st(np.ones(len(df), bool)))
for lg in ["GB1", "ES1", "L1", "IT1", "FR1"]:
    print(lg, st((df.league == lg).to_numpy()))
for g in ["FW", "MF", "DF", "GK"]:
    print(g, st((df.pos_group == g).to_numpy()))
# yardstick: does the stats view predict Transfermarkt's latest move?
has = df.prev_val.notna().to_numpy()
gap = oof[has] - np.log(df.prev_val.to_numpy()[has])
move = y[has] - np.log(df.prev_val.to_numpy()[has])
slope = np.cov(gap, move)[0, 1] / np.var(gap)
big = np.abs(gap) > np.log(1.5)
hit = np.mean(np.sign(gap[big & (np.abs(move) > 0.05)]) == np.sign(move[big & (np.abs(move) > 0.05)]))
print(f"yardstick: corr(stats gap, TM move) {np.corrcoef(gap, move)[0,1]:.3f}, slope {slope:.3f}; "
      f"stats >50% away from prev value: TM moved that way {100*hit:.0f}% of the time (n {int((big & (np.abs(move) > 0.05)).sum())})")
pd.to_pickle({"oof": oof, "ids": df["id"].astype(str).to_numpy()}, T.INTERIM / f"oof_sf_{tag}.pkl")
