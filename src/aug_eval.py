"""Does training on past-season snapshots help? (v17 experiment)

E0: AUG features, today's 545 players only            (5-fold x R over today's players)
E1: AUG features, today's players + past-season rows   (past rows always in training, except the summer-2026 row of
                                                        any player in the test fold, which could share his target)
WF: walk-forward - train on earlier summers, predict a later summer that the model has never seen.
All models: LightGBM + ridge + Extra Trees, predicting the change from the previous valuation.
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

sys.argv = ["x"]
sys.path.insert(0, os.path.dirname(__file__))
import train as T  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402


def et():   # same settings as src/allpos.py (importing allpos would run it)
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         ExtraTreesRegressor(n_estimators=150, min_samples_leaf=4, max_features=0.5, n_jobs=2, random_state=0))

INTERIM = T.INTERIM
SEASON_SHARE_NOW = (date(2026, 11, 16) - date(2026, 8, 16)).days / (date(2027, 5, 24) - date(2026, 8, 16)).days


def add_common(df):
    df = df.copy()
    df["age_sq"] = df["age"] ** 2
    df["age_from_peak"] = (df["age"] - 26).abs()
    df["foot_left"] = (df["foot"] == "left").astype(int)
    df["foot_both"] = (df["foot"] == "both").astype(int)
    df["is_winger"] = df["position"].astype(str).str.contains("Winger").astype(int)
    for g in ("DF", "MF", "GK"):
        df[f"pos_{g}"] = (df["pos_group"] == g).astype(int)
    lg = df["fm_prev_league"].fillna("")
    df["aug_prev_pl"] = lg.str.contains("premier_league").astype(int)
    df["aug_prev_big5"] = lg.str.contains("premier_league|la_liga|bundesliga|serie_a|ligue_1").astype(int)
    df["aug_prev_champ"] = lg.str.contains("championship").astype(int)
    for c in ("prev_val", "val_12m_before", "peak_before"):
        df["log_" + c] = np.log(df[c])
    df["fm_min_total"] = df["fm_min_cur"].fillna(0) + df["fm_min_prev"].fillna(0)
    return df


AUG = ["age", "age_sq", "age_from_peak", "height_cm", "foot_left", "foot_both", "is_winger", "pos_DF", "pos_MF", "pos_GK",
       "log_prev_val", "months_since_prev", "log_val_12m_before", "momentum_12m", "log_peak_before", "n_valuations",
       "months_since_valuation", "season_share",
       "fm_min_cur", "fm_min_prev", "fm_min_total", "xg_p90", "xa_p90", "xgxa_p90", "bcc_p90", "chances_p90", "dribbles_p90",
       "fm_goals_p90", "fm_assists_p90", "fm_shots_p90", "tackles_p90", "fm_int_p90", "clearances_p90", "recoveries_p90",
       "def_contrib_p90", "passes_p90", "fm_rating", "fm_matched", "aug_prev_pl", "aug_prev_big5", "aug_prev_champ",
       "international_caps", "injured_now", "days_out_365", "days_out_season", "n_injuries_2y", "longest_injury_2y",
       "youth_caps", "youth_levels", "nat_debut_age", "recent_national", "club_loo_log_mv",
       "contract_years_left", "log_wage"]


def load():
    cur = T.df.copy()
    cur["season_share"] = SEASON_SHARE_NOW
    cur["snapshot"] = "now"
    cur["id"] = cur["id"].astype(str)
    hist = pd.read_csv(INTERIM / "hist_rows.csv", low_memory=False)
    hist["id"] = hist["id"].astype(str)
    hist = hist[hist["prev_val"].notna() & hist["market_value_eur"].notna()].copy()
    for c in ("contract_years_left", "log_wage"):
        hist[c] = np.nan
    cur, hist = add_common(cur), add_common(hist)
    return cur, hist


def fit_predict(Xtr, ytr, btr, Xte, bte, seed, w=None):
    """3-model blend on the change target; returns log predictions. w = optional sample weights."""
    yc = ytr - btr
    p = T.tune(Xtr.reset_index(drop=True), yc, seed)
    kw = {} if w is None else {"sample_weight": w}
    preds = [T.lgbm(p).fit(Xtr, yc, **kw).predict(Xte),
             T.ridge().fit(Xtr, yc, **({} if w is None else {"ridgecv__sample_weight": w})).predict(Xte),
             et().fit(Xtr, yc, **({} if w is None else {"extratreesregressor__sample_weight": w})).predict(Xte)]
    return bte + np.mean(preds, axis=0)


def ape_stats(y, p):
    a = np.abs(np.exp(p) - np.exp(y)) / np.exp(y)
    return dict(median=round(100 * np.median(a), 1), mean=round(100 * np.mean(a), 1), within25=round(100 * np.mean(a <= .25), 1))


if __name__ == "__main__":
    cur, hist = load()
    reps = int(os.environ.get("REPEATS", "2"))
    has = cur["prev_val"].notna().to_numpy()
    c = cur[has].reset_index(drop=True)
    y = np.log(c.market_value_eur.to_numpy()); b = np.log(c.prev_val.to_numpy())
    Xc = c[AUG]
    hy = np.log(hist.market_value_eur.to_numpy()); hb = np.log(hist.prev_val.to_numpy())
    res = {}
    for mode in os.environ.get("MODES", "E0,E1").split(","):
        oof = np.zeros(len(c))
        for i, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=reps, random_state=42).split(c)):
            Xtr, ytr, btr = Xc.iloc[tr], y[tr], b[tr]
            wts = None
            if mode.startswith("E1"):
                test_ids = set(c.id.iloc[te])
                keep = ~(hist.id.isin(test_ids) & (hist.snapshot == "summer 2026")).to_numpy()
                hw = float(mode[3:]) if mode.startswith("E1w") else 1.0     # e.g. E1w0.3 = past rows weighted 0.3
                wts = np.concatenate([np.ones(len(tr)), np.full(int(keep.sum()), hw)])
                Xtr = pd.concat([Xtr, hist.loc[keep, AUG]]); ytr = np.concatenate([ytr, hy[keep]]); btr = np.concatenate([btr, hb[keep]])
            oof[te] += fit_predict(Xtr, ytr, btr, Xc.iloc[te], b[te], 42 + i, wts) / reps
        res[mode] = oof
        print(mode, "today's players (with a previous valuation)", ape_stats(y, oof), flush=True)
        for g in ["FW", "MF", "DF", "GK"]:
            m = (c.pos_group == g).to_numpy(); print("   ", g, ape_stats(y[m], oof[m]), flush=True)
        m = (c.age <= 21).to_numpy(); print("    age<=21", ape_stats(y[m], oof[m]), flush=True)
    print("naive (previous valuation)", ape_stats(y, b))
    pd.to_pickle((c[["id", "name"]], y, res), INTERIM / "aug_eval.pkl")
    if os.environ.get("WF", "1") == "1":
        for train_s, test_s in [(["summer 2024"], "summer 2025"), (["summer 2024", "summer 2025"], "summer 2026")]:
            tr = hist[hist.snapshot.isin(train_s)]; te = hist[hist.snapshot == test_s]
            p = fit_predict(tr[AUG], np.log(tr.market_value_eur.to_numpy()), np.log(tr.prev_val.to_numpy()),
                            te[AUG], np.log(te.prev_val.to_numpy()), 7)
            yt = np.log(te.market_value_eur.to_numpy())
            print("walk-forward", "+".join(s[-4:] for s in train_s), "->", test_s[-4:], "n", len(te),
                  "model", ape_stats(yt, p), "naive", ape_stats(yt, np.log(te.prev_val.to_numpy())), flush=True)
