"""Creates the small CSV datasets used in Lab 05."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)

# 1) Position_Salaries.csv : the exact 10-row table from the lab manual
pd.DataFrame({
    "Position": ["Business Analyst", "Junior Consultant", "Senior Consultant", "Manager",
                 "Country Manager", "Region Manager", "Partner", "Senior Partner", "C-level", "CEO"],
    "Level": range(1, 11),
    "Salary": [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000],
}).to_csv("Position_Salaries.csv", index=False)

# 2) Braking_Distance.csv : 80 rows, single feature (Speed) -> target (Distance); curved (quadratic)
speed = np.round(rng.uniform(20, 120, 80), 1)
dist = 0.006 * speed**2 + 0.25 * speed + rng.normal(0, 2.0, 80)
pd.DataFrame({"Speed_kmh": speed, "Braking_Distance_m": np.round(dist, 2)}) \
  .sort_values("Speed_kmh").to_csv("Braking_Distance.csv", index=False)

# 3) Car_MPG.csv : 200 rows, 4 features -> MPG ; non-linear in Horsepower
n = 200
hp = np.round(rng.uniform(60, 250, n))
weight = np.round(800 + 5.5 * hp + rng.normal(0, 120, n))
engine = np.round(np.clip(hp / 70 + rng.normal(0, 0.3, n), 0.9, 5.5), 1)
age = rng.integers(0, 21, n)
mpg = 6 + 3000 / (hp + 30) - 0.0025 * weight - 0.15 * age + rng.normal(0, 1.0, n)
pd.DataFrame({"Horsepower": hp, "Weight_kg": weight, "EngineSize_L": engine,
              "Age_years": age, "MPG": np.round(mpg, 2)}).to_csv("Car_MPG.csv", index=False)
print("CSV files created")
