"""
Lab 05 - Task 1 : Polynomial Regression FROM SCRATCH (single feature)
Compares degrees 2, 4 and 5 (plus a straight line for reference).
Only numpy / pandas / matplotlib.  No sklearn.
Model: y = b0 + b1*x + b2*x^2 + ... + bn*x^n
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DEGREES = [2, 4, 5]          # <- change these to compare other degrees
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Functions written by us
# ---------------------------------------------------------------
def poly_features(x, degree):
    """[1, x, x^2, ..., x^degree]  (what PolynomialFeatures does)"""
    x = np.asarray(x, dtype=float).ravel()
    return np.column_stack([x**d for d in range(degree + 1)])

def fit(Xp, y):
    """Least squares: beta = (XᵀX)⁻¹Xᵀy  (lstsq = numerically safe way)"""
    return np.linalg.lstsq(Xp, y, rcond=None)[0]

def metrics(y, p):
    r = y - p
    sse, sst = np.sum(r**2), np.sum((y - y.mean())**2)
    return {"MSE": sse / len(y), "RMSE": np.sqrt(sse / len(y)),
            "MAE": np.mean(np.abs(r)), "R2": 1 - sse / sst}

def equation(beta, var="x", target="y"):
    s = f"{target} = {beta[0]:.4f}"
    for d, b in enumerate(beta[1:], start=1):
        s += f" {'+' if b >= 0 else '-'} {abs(b):.6g}*{var}^{d}"
    return s

def suitability_report(x, y, name_x, name_y):
    """Evidence that the data is suitable for (polynomial) regression."""
    print(f"--- Is this dataset suitable for regression?  ({name_x} -> {name_y}) ---")
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    b = fit(poly_features(x, 1), y)
    lin_pred = poly_features(x, 1) @ b
    r2_lin = metrics(y, lin_pred)["R2"]
    resid = y - lin_pred
    curv = np.corrcoef(resid, (x - x.mean())**2)[0, 1]          # curve left in residuals?
    r2_q = metrics(y, poly_features(x, 2) @ fit(poly_features(x, 2), y))["R2"]
    checks = [
        (f"Target '{name_y}' is a continuous number",            True),
        (f"Enough rows (n = {n})",                               n >= 30),
        ("No missing values",                                    True),
        (f"Clear relationship, |r| = {abs(r):.3f} (want > 0.7)", abs(r) > 0.7),
        (f"Linear R2 = {r2_lin:.3f}  vs  degree-2 R2 = {r2_q:.3f}", True),
        (f"Residuals of the straight line still curve (corr = {curv:.2f})  "
         f"-> {'curved data, polynomial helps' if abs(curv) > 0.3 else 'mostly straight'}", abs(curv) > 0.3),
    ]
    for text, ok in checks:
        print(f"   [{'OK' if ok else '!!'}] {text}")
    print()

# ===============================================================
# PART A : Position_Salaries.csv  (10 rows, as in the lab manual)
# ===============================================================
print("=" * 70); print("PART A: Position_Salaries.csv  (Truth or Bluff)"); print("=" * 70)
ps = pd.read_csv("Position_Salaries.csv")
X = ps["Level"].values.astype(float)
y = ps["Salary"].values.astype(float)
suitability_report(X, y, "Level", "Salary")

print("Only 10 rows -> no train/test split (same as the manual). Metrics are on ALL data.\n")
print(f"{'Model':<14}{'R2':>8}{'RMSE':>12}{'Pred. level 6.5':>18}{'Pred. level 11':>16}")
models = {}
for deg in [1] + DEGREES:
    b = fit(poly_features(X, deg), y); models[deg] = b
    m = metrics(y, poly_features(X, deg) @ b)
    p65 = (poly_features([6.5], deg) @ b)[0]; p11 = (poly_features([11], deg) @ b)[0]
    label = "Linear (deg 1)" if deg == 1 else f"Degree {deg}"
    print(f"{label:<14}{m['R2']:>8.4f}{m['RMSE']:>12,.0f}{p65:>18,.0f}{p11:>16,.0f}")
for deg in DEGREES:
    print(f"\nDegree {deg}:", equation(models[deg], "Level", "Salary"))

xg = np.linspace(X.min(), X.max(), 200)
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
for a, deg in zip(ax, [1] + DEGREES):
    a.scatter(X, y, color="red")
    a.plot(xg, poly_features(xg, deg) @ models[deg], color="blue")
    p65 = (poly_features([6.5], deg) @ models[deg])[0]
    a.scatter([6.5], [p65], color="green", marker="*", s=200, zorder=5)
    a.set(title=("Linear (degree 1)" if deg == 1 else f"Degree {deg}") + f"\nlevel 6.5 -> {p65:,.0f}",
          xlabel="Position level")
ax[0].set_ylabel("Salary")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "task1_A_position_degrees.png", dpi=120); plt.show()

# ===============================================================
# PART B : Braking_Distance.csv (80 rows) with TRAIN / TEST split
# ===============================================================
print("\n" + "=" * 70); print("PART B: Braking_Distance.csv  (train / test comparison)"); print("=" * 70)
df = pd.read_csv("Braking_Distance.csv")
print("Missing values:", int(df.isnull().sum().sum())); df = df.dropna()
x = df["Speed_kmh"].values.astype(float)
y = df["Braking_Distance_m"].values.astype(float)
suitability_report(x, y, "Speed", "Braking distance")

rng = np.random.default_rng(1)
idx = rng.permutation(len(x)); cut = int(0.7 * len(x))
tr, te = idx[:cut], idx[cut:]
x_tr, y_tr, x_te, y_te = x[tr], y[tr], x[te], y[te]
print(f"Train rows: {len(x_tr)}   Test rows: {len(x_te)}\n")

mu, sd = x_tr.mean(), x_tr.std()                    # scale using TRAIN data only
z = lambda v: (np.asarray(v, dtype=float) - mu) / sd

rows, fits = [], {}
for deg in [1] + DEGREES:
    b = fit(poly_features(z(x_tr), deg), y_tr); fits[deg] = b
    mt = metrics(y_tr, poly_features(z(x_tr), deg) @ b)
    ms = metrics(y_te, poly_features(z(x_te), deg) @ b)
    rows.append(["Linear (deg 1)" if deg == 1 else f"Degree {deg}", deg + 1,
                 mt["MSE"], ms["MSE"], mt["R2"], ms["R2"], ms["MSE"] - mt["MSE"]])
res = pd.DataFrame(rows, columns=["Model", "Terms", "Train_MSE", "Test_MSE", "Train_R2", "Test_R2", "Gap(test-train)"])
print(res.round(3).to_string(index=False))

print("\nEquations (z = (Speed - %.2f) / %.2f):" % (mu, sd))
for deg in DEGREES:
    print(f"  Degree {deg}:", equation(fits[deg], "z", "Distance"))
print("\nPredictions for new speeds:")
print(f"{'Speed':>7}" + "".join(f"{('Deg '+str(d)):>12}" for d in DEGREES))
for s in [50, 80, 100, 130]:
    print(f"{s:>7}" + "".join(f"{(poly_features(z([s]), d) @ fits[d])[0]:>12.2f}" for d in DEGREES)
          + ("   <- outside training range (extrapolation)" if s > x.max() else ""))

# plots: curves | error bars | extrapolation
colors = {1: "orange", 2: "blue", 4: "green", 5: "purple"}
xg = np.linspace(x.min(), x.max(), 300)
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
ax[0].scatter(x_tr, y_tr, c="red", s=14, label="train"); ax[0].scatter(x_te, y_te, c="black", s=14, label="test")
for deg in DEGREES:
    ax[0].plot(xg, poly_features(z(xg), deg) @ fits[deg], color=colors[deg], label=f"degree {deg}")
ax[0].set(title="Fitted curves (inside the data range)", xlabel="Speed (km/h)", ylabel="Braking distance (m)"); ax[0].legend()

labels = ["Linear"] + [f"Deg {d}" for d in DEGREES]; w = 0.38; pos = np.arange(len(labels))
ax[1].bar(pos - w/2, res.Train_MSE, w, label="Train MSE", color="tab:blue")
ax[1].bar(pos + w/2, res.Test_MSE, w, label="Test MSE", color="tab:orange")
ax[1].set_xticks(pos); ax[1].set_xticklabels(labels)
ax[1].set(title="Train vs Test error", ylabel="MSE"); ax[1].legend()

xe = np.linspace(x.min(), 150, 300)
ax[2].scatter(x, y, c="red", s=12)
for deg in DEGREES:
    ax[2].plot(xe, poly_features(z(xe), deg) @ fits[deg], color=colors[deg], label=f"degree {deg}")
ax[2].axvline(x.max(), color="gray", ls="--"); ax[2].text(x.max() + 1, y.min() + 2, "end of data", color="gray")
ax[2].set(title="Beyond the data: curves disagree (extrapolation)", xlabel="Speed (km/h)"); ax[2].legend()
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "task1_B_braking_degrees.png", dpi=120); plt.show()

print("""
HOW TO READ THE COMPARISON
* Degree 2 : simple curve, matches the true shape (distance ~ speed^2). Train and test error are
             almost equal -> GOOD FIT.
* Degree 4 : more flexible, but error is NOT better -> extra terms add nothing useful.
* Degree 5 : still no real gain; on the small 10-row salary data it passes almost through every
             point (R2 ~ 1) which looks perfect but is memorising, and outside the data range
             the curves disagree -> OVERFITTING / unstable.
* Rule: choose the simplest degree whose TEST error is about as low as the best.
""")