"""Injury and youth-international features from Transfermarkt (data/raw/transfermarkt_extra_2026.json).

injury_youth_features(ids, dobs, senior_caps, as_of, season_start) -> DataFrame (one row per id, same order)
Everything is computed "as of" a date, so the same code serves today's rows and past-season snapshots.
"""
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data/raw"
_EXTRA = None


def extra():
    global _EXTRA
    if _EXTRA is None:
        p = RAW / "transfermarkt_extra_2026.json"
        _EXTRA = json.load(open(p)) if p.exists() else {"players": {}, "squads": {}}
    return _EXTRA


def _d(s):
    try:
        return datetime.strptime(str(s).strip(), "%d/%m/%Y").date()
    except Exception:
        try:
            return datetime.strptime(str(s).strip()[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def _overlap(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0, (hi - lo).days + 1) if hi >= lo else 0


def injury_youth_features(ids, dobs, senior_caps, as_of, season_start):
    P = extra()["players"]
    rows = []
    for pid, dob, caps in zip(ids, dobs, senior_caps):
        o = P.get(str(pid), {})
        r = dict(injured_now=0, days_out_365=0, days_out_season=0, n_injuries_2y=0, longest_injury_2y=0,
                 youth_caps=np.nan, youth_levels=np.nan, nat_debut_age=np.nan, recent_national=0, has_tm_extra=int(bool(o)))
        for row in o.get("inj", []):
            if len(row) < 6:
                continue
            f, u = _d(row[2]), _d(row[3])
            if not f or f > as_of:
                continue
            if u is None:                       # open-ended injury ("-" / "?")
                u = as_of
            r["injured_now"] |= int(f <= as_of <= u)
            u_c = min(u, as_of)
            r["days_out_365"] += _overlap(f, u_c, as_of.replace(year=as_of.year - 1), as_of)
            r["days_out_season"] += _overlap(f, u_c, season_start, as_of)
            if (as_of - f).days <= 730:
                r["n_injuries_2y"] += 1
                r["longest_injury_2y"] = max(r["longest_injury_2y"], (u_c - f).days)
        nat = [n for n in o.get("nat", []) if _d(n[3]) and _d(n[3]) <= as_of]
        if o.get("nat") is not None:
            total = sum((n[1] or 0) for n in nat)
            sc = 0 if caps != caps else caps
            r["youth_caps"] = max(0, total - sc)
            r["youth_levels"] = max(0, len(nat) - (1 if sc > 0 else 0))
            debuts = [_d(n[3]) for n in nat]
            if debuts and dob:
                r["nat_debut_age"] = (min(debuts) - dob).days / 365.25
            r["recent_national"] = int(any(_d(n[4]) and (as_of - _d(n[4])).days <= 400 for n in nat))
        rows.append(r)
    return pd.DataFrame(rows)
