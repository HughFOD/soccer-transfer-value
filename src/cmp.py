import os, sys, numpy as np, pandas as pd
sys.argv = ['x']
import train as T
rows = []
for fs in os.environ["FS"].split(","):
    fwd, y, oof, _ = T.run(T.FEATURE_SETS[fs], True)
    d = T.df.loc[fwd]
    for m in ("lgbm", "blend"):
        ape = np.abs(np.exp(oof[m]) - np.exp(y)) / np.exp(y)
        r = dict(features=fs, model=m, **T.metrics(y, oof[m]))
        jump = (d.market_value_eur / d.prev_val > 1.6).to_numpy()   # players TM re-rated up >60% at the latest update
        r["big_risers_median"] = round(100 * np.median(ape[jump]), 1); r["big_risers_n"] = int(jump.sum())
        rows.append(r); print(r, flush=True)
    pd.to_pickle((fwd, y, oof), f"../data/interim/oof_{fs}.pkl")
pd.DataFrame(rows).to_csv(f"../out/cmp_{os.environ['TAG']}.csv", index=False)
