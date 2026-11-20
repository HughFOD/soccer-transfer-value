# Prediction log

`estimates_2026-11-20.csv` freezes every out-of-sample estimate the site showed on 20 Nov 2026 (model v14),
before Transfermarkt's next update round. `call` is the model's verdict: **undervalued** (estimate at least 25%
above Transfermarkt), **overvalued** (at least 25% below) or **fair**.

After Transfermarkt's next update, re-pull current values and compare: did the "undervalued" group rise more
than the "fair" group, and the "overvalued" group fall? That is the real test of the model, and the result goes
on the site's Track record section. Values are euros; `tm_value_nov2026_money` is the published value restated
to 16 Nov 2026 money at 10% a year.
