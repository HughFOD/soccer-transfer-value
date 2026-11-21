"""Likely transfer fee: how real fees compare with the Transfermarkt value at the time.

Training rows: permanent transfers since 2021 with a known fee and TM value at the
time of the move, taken from the transfer histories of the players in the dataset.
Target: log(fee / TM value). Features known both at transfer time and today:
  log TM value, age, tenure at selling club (proxy for contract left),
  buyer is an English top-flight club, seller is an English top-flight club, year.
"""
import json, re, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"

def money(s):
    m = re.search(r"€\s*([\d.]+)\s*([MK]?)", str(s))
    if not m: return np.nan
    v = float(m.group(1)); return v * (1e6 if m.group(2) == "M" else 1e3 if m.group(2) == "K" else 1)

def pl_clubs(t):
    """Short club names TM uses for English top-flight clubs in transfer lists."""
    return {"Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Burnley","Chelsea",
            "Crystal Palace","Everton","Fulham","Leeds","Liverpool","Man City","Man United",
            "Newcastle","Nott'm Forest","Sunderland","Spurs","Tottenham","West Ham","Wolves",
            "Leicester","Southampton","Ipswich","Luton","Sheff Utd","Watford","Norwich"}

CAP = 10e6  # value effect levels off above €10m (best fit for stars in CV)
FEATS = ["log_mv","age","age2","tenure","buyer_pl","seller_pl","year"]

def rows():
    t = json.load(open(RAW / "transfermarkt_transfers_2026.json"))
    tm = {p["id"]: p for p in json.load(open(RAW / "transfermarkt_2026.json"))}
    PL = pl_clubs(t)
    out = []
    for pid, v in t.items():
        p = tm.get(pid)
        if not p: continue
        dob = pd.to_datetime(p["dob_age"][:10], dayfirst=True, errors="coerce")
        hist = sorted(v, key=lambda r: r[0])
        for i, (d, frm, to, fee, mv, up) in enumerate(hist):
            if up or "loan" in str(fee).lower() or d < "2021-01-01": continue
            f, m = money(fee), money(mv)
            if not (f > 0 and m > 0) or m < 1e6: continue
            prev = [r[0] for r in hist[:i] if r[2] == frm]
            dt = pd.Timestamp(d)
            ten = (dt - pd.Timestamp(prev[-1])).days / 365.25 if prev else np.nan
            age = (dt - dob).days / 365.25
            out.append(dict(id=pid, date=d, frm=frm, to=to, fee=f, mv=m, age=age, tenure=ten,
                            buyer_pl=int(to in PL), seller_pl=int(frm in PL), year=dt.year + dt.dayofyear / 366))
    r = pd.DataFrame(out)
    r["tenure"] = r.tenure.fillna(r.tenure.median()).clip(0, 8)
    r["log_mv"] = np.log(r.mv.clip(upper=CAP)); r["age2"] = (r.age - 25) ** 2
    r["y"] = np.log(r.fee / r.mv).clip(-1.5, 2)
    return r

def fit(r, alpha=10.0):
    X = r[FEATS].values; mu, sd = X.mean(0), X.std(0) + 1e-9
    m = Ridge(alpha=alpha).fit((X - mu) / sd, r.y)
    return dict(mu=mu, sd=sd, coef=m.coef_, b=m.intercept_)

def bands(r, alpha=100.0):
    """Out-of-fold residual quartiles, separately for players under/over €20m."""
    res = np.zeros(len(r))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(r):
        M = fit(r.iloc[tr], alpha); res[te] = r.y.values[te] - predict(M, r.iloc[te][FEATS].values)
    big = r.mv.values >= 20e6
    return {k: (float(np.quantile(res[m], .25)), float(np.quantile(res[m], .75)), int(m.sum()))
            for k, m in [("small", ~big), ("big", big)]}

def predict(M, X):
    return (np.asarray(X) - M["mu"]) / M["sd"] @ M["coef"] + M["b"]

if __name__ == "__main__":
    r = rows()
    errs, base, eb, bb = [], [], [], []
    for rep in range(5):
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(r):
            M = fit(r.iloc[tr], 100); p = predict(M, r.iloc[te][FEATS].values)
            y = r.y.values[te]; big = r.mv.values[te] >= 30e6
            e = np.abs(y - p); f = np.abs(y - r.y.iloc[tr].median())
            errs += list(e); base += list(f); eb += list(e[big]); bb += list(f[big])
    print(f"{len(r)} transfers. median |log err|: model {np.median(errs):.3f} vs flat {np.median(base):.3f}; "
          f"€30m+: model {np.median(eb):.3f} vs flat {np.median(bb):.3f}")
    M = fit(r, 100); M["bands"] = bands(r); M["feats"] = FEATS; M["cap"] = CAP; M["n"] = len(r)
    print("per-unit effects", dict(zip(FEATS, np.round(M["coef"] / M["sd"], 4))))
    print("bands", M["bands"])
    pickle.dump(M, open(ROOT / "data/interim/fee_model.pkl", "wb"))
