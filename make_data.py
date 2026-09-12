"""Generate the deterministic demo CSVs.

churn.csv  - the Milestone 1 dataset: imbalanced, non-linear, missing values
fraud.csv  - numeric + categorical + missing (both kinds) + imbalance, target auto-detected
houses.csv - a regression-shaped file, to demonstrate the needs_clarification path
"""

from pathlib import Path

import numpy as np
import pandas as pd

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)

# --- churn.csv -----------------------------------------------------------------------
rng = np.random.default_rng(7)
n = 1200

tenure = rng.integers(1, 72, n)
monthly = rng.normal(70, 25, n).clip(15, 150)
support = rng.poisson(1.4, n)
plan = rng.choice(["basic", "plus", "premium"], n, p=[0.5, 0.3, 0.2])
contract = rng.choice(["monthly", "yearly"], n, p=[0.65, 0.35])

# Non-linear: churn risk peaks for short tenure AND high spend -> trees should beat logreg.
risk = (
    2.2 * (tenure < 12)
    + 1.6 * (monthly > 90) * (tenure < 24)
    + 0.55 * support
    + 1.1 * (contract == "monthly")
    - 0.9 * (plan == "premium")
    - 3.4
)
churn = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)

df = pd.DataFrame(
    {
        "tenure_months": tenure,
        "monthly_charge": monthly.round(2),
        "support_calls": support,
        "plan": plan,
        "contract": contract,
        "churned": churn,
    }
)
df.loc[rng.choice(n, 60, replace=False), "monthly_charge"] = np.nan  # real missing values
df.to_csv(out / "churn.csv", index=False)
print(f"churn.csv  rows={len(df)} churn_rate={churn.mean():.3f} ratio={(churn == 0).sum() / churn.sum():.2f}:1")

# --- fraud.csv -----------------------------------------------------------------------
# Same four properties as churn, plus missing values in a CATEGORICAL column and no
# explicit target, so the Data Engineer has to infer one.
rng = np.random.default_rng(3)
n = 400
fraud = pd.DataFrame(
    {
        "amount": rng.normal(200, 60, n).round(2),
        "visits": rng.integers(1, 30, n),
        "channel": rng.choice(["web", "mobile", "store"], n),
        "segment": rng.choice(["consumer", "business"], n, p=[0.7, 0.3]),
    }
)
risk = 1.8 * (fraud.visits < 5) + 1.2 * (fraud.amount > 260) + 0.9 * (fraud.channel == "store") - 2.9
fraud["fraud"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
fraud.loc[rng.choice(n, 40, replace=False), "amount"] = np.nan
fraud.loc[rng.choice(n, 25, replace=False), "channel"] = np.nan
fraud.to_csv(out / "fraud.csv", index=False)
counts = fraud.fraud.value_counts()
print(f"fraud.csv  rows={len(fraud)} classes={dict(counts)} ratio={counts.max() / counts.min():.2f}:1")

# --- houses.csv ----------------------------------------------------------------------
houses = pd.DataFrame({"sqft": range(300, 400), "price": [s * 312.5 for s in range(300, 400)]})
houses.to_csv(out / "houses.csv", index=False)
print(f"houses.csv rows={len(houses)} continuous target - ATLAS should refuse to guess")

