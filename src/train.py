"""Train + evaluate forward market-value models (2026-27 snapshot).

Evaluation is nested: outer 5-fold x 5 repeats for out-of-fold (OOF) predictions,
inner 3-fold grid search per outer fold, so the reported metrics never see data the
hyperparameters were tuned on (the old pipeline tuned on the same folds it scored).
"""
import json, sys, warnings
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"

import os
df = pd.read_csv(INTERIM / os.environ.get("INPUT", "players_all.csv"), low_memory=False)
df = df[df.market_value_eur.notna()].reset_index(drop=True)
df["foot_left"] = (df.foot == "left").astype(int)
df["foot_both"] = (df.foot == "both").astype(int)
df["prev_pl"] = (df.prev_season_league == "PL").astype(int)
df["prev_big5"] = df.prev_season_league.isin(["PL", "other_big5"]).astype(int)
df["prev_champ"] = (df.prev_season_league == "championship").astype(int)
df["prev_other"] = (df.prev_season_league == "other_league").astype(int)
df["has_wage"] = df.annual_wage_eur.notna().astype(int)
POS_GROUP = {"Goalkeeper": "GK", "Centre-Back": "DF", "Left-Back": "DF", "Right-Back": "DF",
             "Defensive Midfield": "MF", "Central Midfield": "MF", "Attacking Midfield": "MF",
             "Left Midfield": "MF", "Right Midfield": "MF"}
df["pos_group"] = df.position.map(POS_GROUP).fillna("FW")
for g in ("DF", "MF", "GK"):
    df[f"pos_{g}"] = (df.pos_group == g).astype(int)

PROFILE = ["age", "age_sq", "age_from_peak", "height_cm", "foot_left", "foot_both",
           "international_caps", "intl_goals", "caps_per_age", "senior_intl", "current_intl", "former_intl", "youth_intl", "english",
           "is_winger", "contract_years_left", "years_at_club", "new_signing", "years_since_extension"]
CLUB = ["club_loo_log_mv", "team_gd_per90", "ppg_cur"]
STATS = ["min_cur", "min_prev", "starts_cur", "starts_prev", "minutes_share_cur", "pooled_90s",
         "prev_pl", "prev_big5", "prev_champ", "prev_other", "gls_per90", "ast_per90", "ga_per90", "npga_per90", "sh_per90",
         "sot_per90", "g_per_sh", "fouled_per90", "crs_per90", "int_per90", "tklw_per90",
         "fls_per90", "offsides_per90", "cards_per90", "ga_cur", "ga_prev", "experience_score"]
WAGE = ["log_wage", "has_wage"]
INJ = ["injured_now", "days_out_365", "days_out_season", "n_injuries_2y", "longest_injury_2y"]
YOUTH = ["youth_caps", "youth_levels", "nat_debut_age", "recent_national"]
HIST2 = ["log_prev2_val", "months_prev_to_prev2", "last_step", "n_vals_24m", "vol_3y", "months_since_peak", "prev_vs_peak"]
JUMP = ["log_last_fee", "log_max_fee", "fee_vs_prev_val", "months_since_move", "recent_big_move", "last_move_loan",
        "n_moves_3y", "youth_promotion", "form_jump", "recent_extension"]
XG = ["xg_p90", "xa_p90", "xgxa_p90", "bcc_p90", "chances_p90", "dribbles_p90", "finishing", "fm_rating", "fm_matched"]
DEF = ["tackles_p90", "fm_int_p90", "clearances_p90", "blocks_p90", "recoveries_p90", "press_wins_p90",
       "def_contrib_p90", "passes_p90", "long_balls_p90"]
HISTORY_CAL = ["log_mv_1y_ago", "log_mv_peak_1y_ago", "mv_trend_prev_year", "n_valuations"]   # Round 7 v11: anchored to today
HISTORY = ["months_since_valuation", "log_prev_val", "months_since_prev", "log_val_12m_before", "momentum_12m", "log_peak_before", "n_valuations"]

# v19: league of the current club (only varies when trained on all Big-5 leagues; BIG5=1 dataset)
LEAGUE = ["lg_ES1", "lg_L1", "lg_IT1", "lg_FR1", "league_gap"]
for _c in LEAGUE:
    if _c not in df:
        df[_c] = 0.0
if "league" not in df:
    df["league"] = "GB1"

FEATURE_SETS = {
    "stats_only": PROFILE + CLUB + STATS,
    "stats_wage": PROFILE + CLUB + STATS + WAGE,
    "full": PROFILE + CLUB + STATS + WAGE + HISTORY,
    "full_calendar": PROFILE + CLUB + STATS + WAGE + HISTORY_CAL,
    "full_iy": PROFILE + CLUB + STATS + WAGE + HISTORY + INJ + YOUTH,
    "full_iy_lg": PROFILE + CLUB + LEAGUE + STATS + WAGE + HISTORY + INJ + YOUTH,
    "stats_xg_lg": PROFILE + CLUB + LEAGUE + STATS + XG + DEF,
    "full_inj": PROFILE + CLUB + STATS + WAGE + HISTORY + INJ,
    "full_hist2": PROFILE + CLUB + STATS + WAGE + HISTORY + HIST2,
    "full_jump": PROFILE + CLUB + STATS + WAGE + HISTORY + JUMP,
    "full_xg": PROFILE + CLUB + STATS + WAGE + HISTORY + XG + DEF,
    "full_all": PROFILE + CLUB + STATS + WAGE + HISTORY + JUMP + XG + DEF,
    "stats_xg": PROFILE + CLUB + STATS + XG + DEF,
    "full_no_staleness": PROFILE + CLUB + STATS + WAGE + [h for h in HISTORY if h != "months_since_valuation"],
}
GRID = {"n_estimators": [300, 600], "learning_rate": [0.03, 0.06],
        "num_leaves": [4, 8], "min_child_samples": [6, 12]}
if os.environ.get("BIGGRID") == "1":   # v19: more data supports bigger trees
    GRID = {"n_estimators": [600, 1200], "learning_rate": [0.03], "num_leaves": [8, 16, 31], "min_child_samples": [10, 25]}
BASE = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1)


def lgbm(params):
    return lgb.LGBMRegressor(**BASE, **params, random_state=0)


def tune(X, y, seed):
    best, best_err = None, 1e9
    inner = KFold(3, shuffle=True, random_state=seed)
    for vals in product(*GRID.values()):
        p = dict(zip(GRID, vals))
        err = 0
        for tr, va in inner.split(X):
            m = lgbm(p).fit(X.iloc[tr], y[tr])
            err += np.mean(np.abs(m.predict(X.iloc[va]) - y[va]))
        if err < best_err:
            best, best_err = p, err
    return best


def ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 3, 30)))


def metrics(y_log, p_log):
    y, p = np.exp(y_log), np.exp(p_log)
    ape = np.abs(p - y) / y
    r2_log = 1 - np.sum((y_log - p_log) ** 2) / np.sum((y_log - y_log.mean()) ** 2)
    r2_eur = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(r2_log=round(r2_log, 3), r2_eur=round(r2_eur, 3),
                median_ape=round(100 * np.median(ape), 1), mean_ape=round(100 * np.mean(ape), 1),
                within_25pct=round(100 * np.mean(ape <= 0.25), 1))


def run(feats, train_all_positions=False, n_repeats=5, seed=42):
    """Returns OOF log predictions for forwards (averaged over repeats) for lgbm, ridge, blend."""
    fwd = df.index[df.is_forward == 1].to_numpy()
    others = df.index[df.is_forward == 0].to_numpy()
    feats = feats + (["pos_DF", "pos_MF", "pos_GK"] if train_all_positions else [])
    X, y = df[feats], np.log(df.market_value_eur.to_numpy())
    oof = {k: np.zeros(len(fwd)) for k in ("lgbm", "ridge")}
    params_used = []
    rkf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(rkf.split(fwd)):
        tr_idx = np.concatenate([fwd[tr], others]) if train_all_positions else fwd[tr]
        p = tune(X.iloc[tr_idx].reset_index(drop=True), y[tr_idx], seed + i)
        params_used.append(p)
        m = lgbm(p).fit(X.iloc[tr_idx], y[tr_idx])
        oof["lgbm"][te] += m.predict(X.iloc[fwd[te]]) / n_repeats
        r = ridge().fit(X.iloc[tr_idx], y[tr_idx])
        oof["ridge"][te] += r.predict(X.iloc[fwd[te]]) / n_repeats
    oof["blend"] = 0.5 * oof["lgbm"] + 0.5 * oof["ridge"]
    return fwd, y[fwd], oof, params_used


if __name__ == "__main__":
    results, store = [], {}
    configs = [("full", True), ("full_no_staleness", True), ("full", False), ("stats_wage", True), ("stats_only", True)]
    for fs, allpos in configs:
        fwd, y, oof, params = run(FEATURE_SETS[fs], allpos)
        for model, pred in oof.items():
            row = dict(features=fs, train_on="all positions" if allpos else "forwards", model=model,
                       **metrics(y, pred))
            results.append(row)
            print(row, flush=True)
        store[(fs, allpos)] = (fwd, y, oof, params)
        pd.to_pickle(store, INTERIM / "oof_store.pkl")
    res = pd.DataFrame(results)
    res.to_csv(INTERIM / "model_comparison.csv", index=False)
    print(res.sort_values("median_ape").to_string())
    pd.to_pickle(store, INTERIM / "oof_store.pkl")
