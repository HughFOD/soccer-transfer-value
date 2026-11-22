"""Build the 2026-27 forwards dataset from the raw FBref + Transfermarkt pulls.

Inputs : data/raw/fbref_2026.json, data/raw/transfermarkt_2026.json
Outputs: data/interim/players_all.csv  (all 547 TM players, joined + engineered)
         data/interim/match_report.csv (TM -> FBref name-match audit trail)
"""
import json, re, math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

ROOT = Path(__file__).resolve().parents[1]
RAW, INTERIM = ROOT / "data/raw", ROOT / "data/interim"
INTERIM.mkdir(parents=True, exist_ok=True)
TODAY = date(2026, 11, 16)
ONE_YEAR_AGO = date(2025, 11, 16)
TWO_YEARS_AGO = date(2024, 11, 16)
FORWARD_POSITIONS = {"Centre-Forward", "Left Winger", "Right Winger", "Second Striker"}


def norm(s):
    s = unidecode(str(s)).lower()
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def num(x):
    if x is None:
        return np.nan
    x = str(x).replace(",", "").replace("%", "").strip()
    if x in ("", "-"):
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def dmy(s):
    try:
        return datetime.strptime(s.strip(), "%d/%m/%Y").date()
    except Exception:
        return None


def money(s):
    """'€75.00M' / '€800.0K' -> euros"""
    m = re.search(r"€\s*([\d.]+)\s*([MK]?)", str(s))
    if not m:
        return np.nan
    v = float(m.group(1))
    return v * (1e6 if m.group(2) == "M" else 1e3 if m.group(2) == "K" else 1)


# ---------------------------------------------------------------- FBref
fb = json.load(open(RAW / "fbref_2026.json"))
import os
# v19: BIG5=1 adds La Liga, Bundesliga, Serie A and Ligue 1 players (src/convert_big5.py makes the inputs)
BIG5 = os.environ.get("BIG5", "0") == "1" and (RAW / "fbref_big5_2026.json").exists()
if BIG5:
    fb.update(json.load(open(RAW / "fbref_big5_2026.json")))


def fb_frame(key):
    t = fb[key]
    df = pd.DataFrame(t["rows"], columns=t["cols"][: len(t["rows"][0])])
    df[["player", "fbref_id"]] = df["player"].str.split("|", n=1, expand=True)
    return df


NONNUM = {"ranker", "player", "fbref_id", "nationality", "position", "team", "comp_level",
          "age", "matches", "weekly_wages", "notes"}


def season_table(prefix, kinds):
    """Merge standard/shooting/misc/(playingtime) for one season and sum
    split-season stints (a player with two club rows) into one row per player."""
    merged = None
    for kind in kinds:
        df = fb_frame(f"{prefix}_{kind}")
        keep = ["fbref_id", "player", "team", "position", "birth_year"] + (
            ["comp_level"] if "comp_level" in df else [])
        numcols = [c for c in df.columns if c not in NONNUM and c not in keep]
        for c in numcols:
            df[c] = df[c].map(num)
        df["stint"] = df.groupby("fbref_id").cumcount()
        df = df[keep + ["stint"] + numcols]
        if merged is None:
            merged = df
        else:
            new = [c for c in df.columns if c not in merged.columns]
            merged = merged.merge(df[["fbref_id", "stint"] + new], on=["fbref_id", "stint"], how="left")
    sum_cols = [c for c in merged.columns if c not in NONNUM | {"stint", "birth_year"}
                and not c.endswith(("per90", "_pct", "per_shot", "per_shot_on_target",
                                    "per_game", "per_start", "per_sub", "points_per_game", "wowy"))]
    agg = {c: "sum" for c in sum_cols}
    agg.update({"player": "first", "birth_year": "first", "position": "first",
                "team": lambda s: " / ".join(dict.fromkeys(s))})
    if "comp_level" in merged:
        agg["comp_level"] = lambda s: " / ".join(dict.fromkeys(s))
    if "points_per_game" in merged:  # minutes-weighted
        merged["ppg_x_min"] = merged["points_per_game"] * merged["minutes"]
        agg["ppg_x_min"] = "sum"
    out = merged.groupby("fbref_id").agg(agg).reset_index()
    if "ppg_x_min" in out:
        out["points_per_game"] = out["ppg_x_min"] / out["minutes"].replace(0, np.nan)
    out["n_stints"] = merged.groupby("fbref_id").size().values
    return out


cur = season_table("b527" if BIG5 else "pl27", ["standard", "shooting", "misc", "playingtime"])
prev = season_table("b5", ["standard", "shooting", "misc"])          # all Big-5 leagues 2025-26
# Last-season stats from leagues outside the Big 5 (Championship for the promoted clubs, Portugal, NL, ...)
OTHER = RAW / "fbref_other_2026.json"
other_prev = None
if OTHER.exists():
    oth = json.load(open(OTHER))
    frames = []
    for lg in sorted({k.rsplit("_", 1)[0] for k in oth}):
        for kind in ("stats", "shooting", "misc"):
            if f"{lg}_{kind}" in oth:
                fb[f"{lg}_{'standard' if kind == 'stats' else kind}"] = oth[f"{lg}_{kind}"]
        kinds = [k for k in ("standard", "shooting", "misc") if f"{lg}_{k}" in fb]
        if "standard" in kinds:
            t = season_table(lg, kinds)
            t["comp_level"] = lg
            frames.append(t)
    other_prev = pd.concat(frames, ignore_index=True)
    num_cols = [c for c in other_prev.columns if c not in ("fbref_id", "player", "team", "position", "birth_year", "comp_level")]
    other_prev = other_prev.groupby("fbref_id").agg({**{c: "sum" for c in num_cols}, "player": "first", "team": "first",
                                                     "position": "first", "birth_year": "first",
                                                     "comp_level": lambda s: " / ".join(dict.fromkeys(s))}).reset_index()
prev_pt = season_table("b526" if BIG5 else "pl26", ["playingtime"])[["fbref_id", "on_goals_for", "on_goals_against", "minutes"]]
prev_pt = prev_pt.rename(columns={"on_goals_for": "pl26_on_gf", "on_goals_against": "pl26_on_ga", "minutes": "pl26_pt_min"})

wages = pd.concat([fb_frame(k) for k in fb if k.endswith("_wages")], ignore_index=True)
wages["annual_wage_eur"] = wages["annual_wages"].map(num)
wages = wages.groupby("fbref_id", as_index=False)["annual_wage_eur"].max()

# Identity table for matching: every FBref player we saw anywhere
ident = []
for key in fb:
    df = fb_frame(key)
    by = df["birth_year"].map(num) if "birth_year" in df else np.nan
    ident.append(pd.DataFrame({"fbref_id": df["fbref_id"], "fb_name": df["player"],
                               "fb_team": df["team"], "birth_year": by, "src": key}))
ident = pd.concat(ident)
ident_by_id = (ident.sort_values("birth_year").groupby("fbref_id")
               .agg(fb_name=("fb_name", "first"), birth_year=("birth_year", "first"),
                    fb_teams=("fb_team", lambda s: set(s)), in_pl27=("src", lambda s: any(x.startswith(("pl27", "b527")) for x in s)))
               .reset_index())
ident_by_id["n"] = ident_by_id["fb_name"].map(norm)

# ---------------------------------------------------------------- Transfermarkt
tm = pd.DataFrame(json.load(open(RAW / "transfermarkt_2026.json")))
tm["league"] = "GB1"
if BIG5:
    tm = pd.concat([tm, pd.DataFrame(json.load(open(RAW / "transfermarkt_big5_2026.json")))], ignore_index=True)
    tm = tm.drop_duplicates("id", keep="first").reset_index(drop=True)
tm["dob"] = tm["dob_age"].str.extract(r"(\d\d/\d\d/\d{4})")[0].map(dmy)
tm["birth_year"] = tm["dob"].map(lambda d: d.year if d else np.nan)
tm["n"] = tm["name"].map(norm)


def club_match(tm_club, fb_teams):
    a = norm(tm_club).replace(" fc", "").replace("afc ", "")
    return max((fuzz.partial_ratio(a, norm(t)) for t in fb_teams if isinstance(t, str)), default=0)


def match_player(row):
    cands = ident_by_id
    if not np.isnan(row.birth_year):
        cands = cands[(cands.birth_year - row.birth_year).abs() <= 1]
    best, best_s = None, 0
    for c in cands.itertuples():
        s = max(fuzz.token_set_ratio(row.n, c.n), fuzz.token_sort_ratio(row.n, c.n))
        # mononyms ("Estevao") vs full names ("Estevao Willian") - token_set handles
        if c.birth_year != row.birth_year:
            s -= 8
        s += 0.15 * club_match(row.club, c.fb_teams) if c.in_pl27 else 0
        if s > best_s:
            best, best_s = c, s
    return (best.fbref_id, best.fb_name, best_s) if best is not None and best_s >= 82 else (None, None, best_s)


res = tm.apply(match_player, axis=1, result_type="expand")
tm[["fbref_id", "fb_name", "match_score"]] = res
# guard: a single fbref id claimed twice -> keep best score only
dup = tm[tm.fbref_id.notna()].sort_values("match_score", ascending=False).duplicated("fbref_id")
tm.loc[dup[dup].index, ["fbref_id", "fb_name"]] = None
tm[["name", "club", "position", "birth_year", "fbref_id", "fb_name", "match_score"]].to_csv(
    INTERIM / "match_report.csv", index=False)

# ---------------------------------------------------------------- Feature engineering
df = tm.copy()
df["market_value_eur"] = df["mv"].map(money)
df["age"] = df["dob"].map(lambda d: (TODAY - d).days / 365.25 if d else np.nan)
df["age_sq"] = df["age"] ** 2
df["age_from_peak"] = (df["age"] - 26).abs()
df["height_cm"] = df["height"].str.replace(",", ".").str.extract(r"([\d.]+)")[0].astype(float) * 100
df["foot"] = df["foot"].replace({"": "unknown", "-": "unknown"}).fillna("unknown")
df["contract_end"] = df["contract"].map(dmy)
df["contract_years_left"] = df["contract_end"].map(lambda d: (d - TODAY).days / 365.25 if d else np.nan)
df["joined_date"] = df["joined"].map(dmy)
df["years_at_club"] = df["joined_date"].map(lambda d: (TODAY - d).days / 365.25 if d else np.nan)
df["new_signing"] = (df["years_at_club"] < 0.5).astype(int)
df["ext_date"] = df["last_ext"].map(dmy)
df["years_since_extension"] = df["ext_date"].map(lambda d: (TODAY - d).days / 365.25 if d else np.nan)
# TM uses three labels: "Current international", "National player" (both active) and "Former International".
# v11/v12 only parsed the first, so Saka, Rashford, Semenyo etc. were flagged as non-internationals.
HDR = json.load(open(RAW / "transfermarkt_headers_2026.json"))
if BIG5:
    HDR.update(json.load(open(RAW / "transfermarkt_big5_headers_2026.json")))
def intl_info(pid):
    for item in HDR.get(str(pid), []):
        m = re.match(r"(Current international|National player|Former International):\s*(.+)", item)
        if m:
            return m.group(1), m.group(2).strip()
    return None, ""
_ii = df["id"].map(intl_info)
df["intl_label"] = _ii.str[0]
df["intl_team"] = _ii.str[1]
_youth = df["intl_team"].str.contains(r"U\d|Olympic", regex=True)
df["senior_intl"] = ((~_youth) & (df["caps"] > 0) & df["intl_label"].notna()).astype(int)
df["current_intl"] = (df["intl_label"].isin(["Current international", "National player"]) & ~_youth).astype(int)
df["former_intl"] = ((df["intl_label"] == "Former International") & ~_youth).astype(int)
df["youth_intl"] = _youth.astype(int)
df["international_caps"] = df["caps"].astype(float)
df["intl_goals"] = df["intl_goals"].astype(float)
df["caps_per_age"] = df["international_caps"] / (df["age"] - 16).clip(lower=1)
df["english"] = df["nat"].fillna("").str.split("|").str[0].eq("England").astype(int)
df["is_winger"] = df["position"].str.contains("Winger").astype(int)

# club context: leave-one-out mean log market value of the rest of the squad
df["log_mv"] = np.log(df["market_value_eur"])
g = df.groupby("club")["log_mv"]
df["club_loo_log_mv"] = (g.transform("sum") - df["log_mv"]) / (g.transform("count") - 1)

# ------------------------------------------------ historical valuations (TM history), in today's money
# Football inflation: PL prices rose ~12.2%/yr over the last decade (CIES) and PL summer spend grew
# ~11% from 2025 to 2026, so every past valuation is restated at INFLATION per year to 16 Nov 2026.
import os
INFLATION = float(os.environ.get("INFLATION", "0.10"))


def infl(v, d):
    return v * (1 + INFLATION) ** ((TODAY - d).days / 365.25)


def hist_points(h):
    return sorted((dmy(d), v) for d, v, _ in h if dmy(d) and v)


def history_features(h):
    pts = hist_points(h)
    out = dict(mv_date=None, prev_val=np.nan, months_since_prev=np.nan, val_12m_before=np.nan,
               peak_before=np.nan, momentum_12m=np.nan, mv_1y_ago=np.nan, mv_2y_ago=np.nan, mv_peak_1y_ago=np.nan,
               prev2_val=np.nan, months_prev_to_prev2=np.nan, last_step=np.nan, n_vals_24m=0, vol_3y=np.nan,
               months_since_peak=np.nan, prev_vs_peak=np.nan)
    if not pts:
        return out
    cur_d, _ = pts[-1]
    out["mv_date"] = cur_d
    before = [(d, v) for d, v in pts[:-1] if d < cur_d]
    if before:
        d, v = before[-1]
        out["prev_val"] = infl(v, d)
        out["months_since_prev"] = (cur_d - d).days / 30.44
        out["peak_before"] = max(infl(v, d) for d, v in before)
        # richer history (v16 experiment): the step before last, update cadence, volatility, distance from peak
        if len(before) >= 2:
            d2, v2 = before[-2]
            out["prev2_val"] = infl(v2, d2)
            out["months_prev_to_prev2"] = (d - d2).days / 30.44
            out["last_step"] = np.log(out["prev_val"] / out["prev2_val"])
        out["n_vals_24m"] = sum(1 for dd, _ in before if (cur_d - dd).days <= 730)
        recent = [(dd, infl(vv, dd)) for dd, vv in before if (cur_d - dd).days <= 3 * 365]
        steps = [np.log(b2 / b1) / max((d2_ - d1_).days / 365.25, 0.25) for (d1_, b1), (d2_, b2) in zip(recent, recent[1:])]
        out["vol_3y"] = float(np.std(steps)) if len(steps) >= 2 else np.nan
        pk_d, pk_v = max(((dd, infl(vv, dd)) for dd, vv in before), key=lambda t: t[1])
        out["months_since_peak"] = (cur_d - pk_d).days / 30.44
        out["prev_vs_peak"] = np.log(out["prev_val"] / pk_v)
        anchor = cur_d.replace(year=cur_d.year - 1) if not (cur_d.month == 2 and cur_d.day == 29) else cur_d.replace(year=cur_d.year - 1, day=28)
        old = [(d, v) for d, v in before if d <= anchor]
        if old:
            d12, v12 = old[-1]
            out["val_12m_before"] = infl(v12, d12)
            out["momentum_12m"] = np.log(out["prev_val"] / out["val_12m_before"])
    # calendar-anchored versions (kept for the naive baseline / calculator)
    for key, when in (("mv_1y_ago", ONE_YEAR_AGO), ("mv_2y_ago", TWO_YEARS_AGO)):
        old = [(d, v) for d, v in pts if d <= when]
        if old:
            out[key] = infl(old[-1][1], old[-1][0])
    old = [infl(v, d) for d, v in pts if d <= ONE_YEAR_AGO]
    out["mv_peak_1y_ago"] = max(old) if old else np.nan
    return out


H = pd.DataFrame([history_features(h) for h in df["mvh"]], index=df.index)
df = pd.concat([df, H], axis=1)
# target: current TM value restated to today's money (most values were set Jun-Oct 2026)
df["actual_value_reported"] = df["market_value_eur"]
df["market_value_eur"] = [infl(v, d) if d else v for v, d in zip(df["actual_value_reported"], df["mv_date"])]
df["log_mv"] = np.log(df["market_value_eur"])
g = df.groupby("club")["log_mv"]
df["club_loo_log_mv"] = (g.transform("sum") - df["log_mv"]) / (g.transform("count") - 1)
# How stale TM's current value is: a value set in June predates every 2026-27 minute (the Kroupi case).
# Known for training rows; the calculator sets it to 0 ("valued today").
df["months_since_valuation"] = [((TODAY - d).days / 30.44) if d else np.nan for d in df["mv_date"]]
# ---- Transfer history (Transfermarkt): signals that tend to precede big re-ratings
TRF = RAW / "transfermarkt_transfers_2026.json"
if TRF.exists():
    trf = json.load(open(TRF))
    if BIG5:
        trf.update(json.load(open(RAW / "transfermarkt_big5_transfers_2026.json")))

    def transfer_features(pid):
        rows = []
        for d, frm, to, fee, mv, upcoming in trf.get(str(pid), []):
            try:
                dd = datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                continue
            if upcoming or dd > TODAY:
                continue
            f = str(fee or "")
            loan = "loan" in f.lower()
            end_loan = f.lower().startswith("end of loan")
            eur = money(f) if "€" in f else (0.0 if "free" in f.lower() else np.nan)
            rows.append((dd, eur, loan, end_loan, frm))
        rows.sort(reverse=True)
        out = dict(last_fee=np.nan, months_since_move=np.nan, last_move_loan=0, n_moves_3y=0,
                   max_fee=np.nan, youth_promotion=0)
        perm = [r for r in rows if not r[3]]                      # ignore "end of loan" returns
        if perm:
            d0, e0, l0, _, frm0 = perm[0]
            out["months_since_move"] = (TODAY - d0).days / 30.44
            out["last_move_loan"] = int(l0)
            out["last_fee"] = infl(e0, d0) if e0 == e0 else np.nan
            out["youth_promotion"] = int(bool(re.search(r"U1\d|U2\d|Youth|Academy|II$| B$", str(frm0))))
        fees = [infl(e, d) for d, e, l, el, _ in perm if e == e and e > 0 and not l]
        out["max_fee"] = max(fees) if fees else np.nan
        out["n_moves_3y"] = sum(1 for r in perm if (TODAY - r[0]).days < 3 * 365)
        return out

    TF = pd.DataFrame([transfer_features(i) for i in df["id"]], index=df.index)
    df = pd.concat([df, TF], axis=1)
    df["log_last_fee"] = np.log1p(df["last_fee"])
    df["log_max_fee"] = np.log1p(df["max_fee"])
    df["fee_vs_prev_val"] = np.log((df["last_fee"] + 1e5) / (df["prev_val"] + 1e5))
    df["recent_big_move"] = ((df["months_since_move"] < 18) & (df["last_fee"] > 0)).astype(int)
df["log_prev_val"] = np.log(df["prev_val"])
df["log_prev2_val"] = np.log(df["prev2_val"])
df["log_val_12m_before"] = np.log(df["val_12m_before"])
df["log_peak_before"] = np.log(df["peak_before"])
df["log_mv_1y_ago"] = np.log(df["mv_1y_ago"])
df["log_mv_peak_1y_ago"] = np.log(df["mv_peak_1y_ago"])
df["mv_trend_prev_year"] = df["log_mv_1y_ago"] - np.log(df["mv_2y_ago"])
df["n_valuations"] = df["mvh"].map(len)

# ------------------------------------------------ FBref stats
df = df.merge(wages, on="fbref_id", how="left")
df["log_wage"] = np.log(df["annual_wage_eur"])
c = cur.add_prefix("c_").rename(columns={"c_fbref_id": "fbref_id"})
p = prev.add_prefix("p_").rename(columns={"p_fbref_id": "fbref_id"})
if other_prev is not None:
    po = other_prev[~other_prev.fbref_id.isin(prev.fbref_id)].add_prefix("p_").rename(columns={"p_fbref_id": "fbref_id"})
    p = pd.concat([p, po], ignore_index=True)
df = df.merge(c, on="fbref_id", how="left").merge(p, on="fbref_id", how="left").merge(prev_pt, on="fbref_id", how="left")

count_cols = ["minutes", "minutes_90s", "games", "games_starts", "goals", "assists", "goals_pens", "pens_made",
              "shots", "shots_on_target", "fouled", "offsides", "crosses", "interceptions",
              "tackles_won", "fouls", "cards_yellow", "cards_red", "pens_won"]
for col in count_cols:
    for pre in ("c_", "p_"):
        if pre + col not in df:
            df[pre + col] = np.nan
# ---- League strength (Opta Power Rankings, average club rating; gap to the Premier League of the same release)
# Big-5 from the June 2025 release (PL 92.6); the rest from the Oct 2023 release (PL 87.3), which is the
# latest with their numbers; Scottish from a 2023 table (PL 87.6). League One isn't published: assumed -20.
LEAGUE_GAP = {
    "Premier League": 0.0, "La Liga": 92.6 - 87.0, "Serie A": 92.6 - 87.0, "Bundesliga": 92.6 - 86.3,
    "Ligue 1": 92.6 - 85.5, "championship": 87.3 - 78.58, "eredivisie": 87.3 - 77.24, "primeira": 87.3 - 77.20,
    "belgian": 87.3 - 77.19, "brazil": 87.3 - 76.43, "super_lig": 87.3 - 75.65, "danish": 87.3 - 75.00,
    "mls": 87.3 - 74.77, "allsvenskan": 87.3 - 72.30, "scottish": 87.59 - 70.25, "league_one": 20.0,
}
# Tested 2026-11-19 (nested CV, b = 0, .02, .03, .05, .08): no gain for the full model (12.4-12.8% median APE
# vs 12.3%) and the stats-only model got slightly worse as b rose (25.5% -> 26.7%). Off by default.
LEAGUE_B = float(os.environ.get("LEAGUE_B", "0.0"))   # output discount per rating point: factor = exp(-b * gap)


def league_gap(cl):
    if not isinstance(cl, str) or not cl:
        return np.nan
    gaps = [g for name, g in LEAGUE_GAP.items() for part in cl.split(" / ") if name.lower() in part.lower()]
    return float(np.mean(gaps)) if gaps else np.nan


df["prev_league_gap"] = df["p_comp_level"].map(league_gap)
_factor = np.exp(-LEAGUE_B * df["prev_league_gap"].fillna(0))
df["prev_league_factor"] = _factor
for col in ["goals", "assists", "goals_pens", "shots", "shots_on_target", "fouled", "crosses", "offsides", "pens_won"]:
    if "p_" + col in df:
        df["p_" + col] = df["p_" + col] * _factor   # last season's attacking output in Premier League terms
_cl = df["p_comp_level"].fillna("")
df["prev_season_league"] = np.select(
    [_cl.str.contains("Premier League"), _cl.str.contains("La Liga|Serie A|Bundesliga|Ligue 1"),
     _cl.str.contains("championship"), _cl.ne("")],
    ["PL", "other_big5", "championship", "other_league"], default="none")
df["min_cur"] = df["c_minutes"].fillna(0)
df["min_prev"] = df["p_minutes"].fillna(0)
df["starts_cur"] = df["c_games_starts"].fillna(0)
df["starts_prev"] = df["p_games_starts"].fillna(0)
# share of the most minutes anyone in his league has played so far (leagues are at different matchdays)
df["minutes_share_cur"] = df["min_cur"] / df.groupby("league")["min_cur"].transform("max").clip(lower=1)
for lg in ["GB1", "ES1", "L1", "IT1", "FR1"]:
    df["lg_" + lg] = (df["league"] == lg).astype(int)
# Opta Power Rankings gap to the Premier League (same table as LEAGUE_GAP below)
df["league_gap"] = df["league"].map({"GB1": 0.0, "ES1": 5.6, "IT1": 5.6, "L1": 6.3, "FR1": 7.1})
df["ppg_cur"] = df["c_points_per_game"]

# Two-season pooled, shrunk per-90 rates. Current season weighted 1.0, last season 0.75
# (recency), and every rate shrunk toward the cohort mean with a 6x90 pseudo-sample so
# a forward with 120 minutes can't post an extreme rate (plan.md next step #1).
W_PREV, K = 0.75, 6.0
nineties = df["c_minutes_90s"].fillna(0) + W_PREV * df["p_minutes_90s"].fillna(0)
df["pooled_90s"] = nineties
fwd_mask = df["position"].isin(FORWARD_POSITIONS)
rate_defs = {
    "gls_per90": "goals", "ast_per90": "assists", "npg_per90": "goals_pens", "sh_per90": "shots",
    "sot_per90": "shots_on_target", "fouled_per90": "fouled", "crs_per90": "crosses",
    "int_per90": "interceptions", "tklw_per90": "tackles_won", "fls_per90": "fouls",
    "offsides_per90": "offsides", "penwon_per90": "pens_won",
}
for feat, col in rate_defs.items():
    tot = df["c_" + col].fillna(0) + W_PREV * df["p_" + col].fillna(0)
    prior = (tot[fwd_mask].sum() / nineties[fwd_mask].sum())
    df[feat] = (tot + K * prior) / (nineties + K)
df["ga_per90"] = df["gls_per90"] + df["ast_per90"]
df["npga_per90"] = df["npg_per90"] + df["ast_per90"]
df["g_per_sh"] = (df["c_goals"].fillna(0) + W_PREV * df["p_goals"].fillna(0) + 0.5) / (
    df["c_shots"].fillna(0) + W_PREV * df["p_shots"].fillna(0) + 5)
df["cards_per90"] = (df["c_cards_yellow"].fillna(0) + 3 * df["c_cards_red"].fillna(0)
                     + W_PREV * (df["p_cards_yellow"].fillna(0) + 3 * df["p_cards_red"].fillna(0))) / (nineties + K)
# this-season-only signals (form)
df["ga_cur"] = df["c_goals"].fillna(0) + df["c_assists"].fillna(0)
df["ga_prev"] = df["p_goals"].fillna(0) + df["p_assists"].fillna(0)
df["team_gd_per90"] = ((df["c_on_goals_for"].fillna(0) - df["c_on_goals_against"].fillna(0)
                        + W_PREV * (df["pl26_on_gf"].fillna(0) - df["pl26_on_ga"].fillna(0)))
                       / (df["c_minutes"].fillna(0) / 90 + W_PREV * df["pl26_pt_min"].fillna(0) / 90 + K))
# breakout form: this season's G+A per 90 vs last season's (both shrunk), and fresh extensions
_n_c = df["c_minutes_90s"].fillna(0); _n_p = df["p_minutes_90s"].fillna(0)
_pr = df.loc[fwd_mask, "ga_per90"].median()
df["ga90_cur_shrunk"] = (df["c_goals"].fillna(0) + df["c_assists"].fillna(0) + K * _pr) / (_n_c + K)
df["ga90_prev_shrunk"] = (df["p_goals"].fillna(0) + df["p_assists"].fillna(0) + K * _pr) / (_n_p + K)
df["form_jump"] = df["ga90_cur_shrunk"] - df["ga90_prev_shrunk"]
df["recent_extension"] = (df["years_since_extension"] < 0.75).astype(int)
df["experience_score"] = np.log1p(df["min_cur"] + df["min_prev"]) * (1 + df["international_caps"] / 50)

# FotMob: xG/xA, chance creation, defensive actions, match rating (see src/fotmob.py)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from fotmob import fotmob_features  # noqa: E402
_fm = fotmob_features(df, cur_key=["premier_league|2026/2027", "la_liga|2026/2027", "bundesliga|2026/2027",
                                    "serie_a|2026/2027", "ligue_1|2026/2027"] if BIG5 else "premier_league|2026/2027")
if _fm is not None:
    df = pd.concat([df, _fm], axis=1)
# Injuries and youth internationals (Transfermarkt; see src/extras.py), as of today, season from 1 Aug 2026
from extras import injury_youth_features  # noqa: E402
_iy = injury_youth_features(df["id"], df["dob"], df["international_caps"], TODAY, date(2026, 8, 1))
_iy.index = df.index
df = pd.concat([df, _iy], axis=1)
df["is_forward"] = fwd_mask.astype(int)
drop = ["mvh", "href", "dob_age", "info"]
df.drop(columns=[c for c in drop if c in df], inplace=True)
df.to_csv(INTERIM / os.environ.get("OUTFILE", "players_all.csv"), index=False)

f = df[fwd_mask]
print(f"players {len(df)}, forwards {len(f)}")
print(f"forwards matched to FBref: {f.fbref_id.notna().sum()}  | with wages: {f.annual_wage_eur.notna().sum()}"
      f"  | with current-season minutes: {(f.min_cur>0).sum()}  | prev-season big5: {(f.min_prev>0).sum()}"
      f"  | with 1y-ago value: {f.mv_1y_ago.notna().sum()}")
print("unmatched forwards:", f[f.fbref_id.isna()][["name", "club", "match_score"]].to_string())
