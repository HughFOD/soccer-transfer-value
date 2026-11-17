import os, sys, numpy as np, pandas as pd
sys.argv=['x']
import train as T
rows=[]
cfg=[(os.environ["INPUT"], fs) for fs in os.environ["FS"].split(",")]
for inp, fs in cfg:
    fwd,y,oof,_=T.run(T.FEATURE_SETS[fs], True)
    for m in ("lgbm","blend"):
        r=dict(inflation=os.environ["INFLATION"], features=fs, model=m, **T.metrics(y,oof[m])); rows.append(r); print(r, flush=True)
    d=T.df.loc[fwd]
    for base in ("prev_val","mv_1y_ago"):
        ape=((d[base]-d.market_value_eur).abs()/d.market_value_eur).dropna()
        print("naive", base, round(100*ape.median(),1), round(100*ape.mean(),1), flush=True)
pd.DataFrame(rows).to_csv(f"out/sweep_{os.environ['INFLATION']}_{os.environ['FS'].replace(',','-')}.csv", index=False)
