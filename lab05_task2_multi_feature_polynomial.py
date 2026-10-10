"""
Lab 05 - Task 2 : Polynomial Regression with MULTIPLE features, FROM SCRATCH
Dataset: Car_MPG.csv (200 rows) -> predict MPG from Horsepower, Weight_kg, EngineSize_L, Age_years
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

# ---------------------------------------------------------------
# Functions written by us
# ---------------------------------------------------------------
def poly_features(X, degree, names):
    """All terms up to `degree`, including squares and interactions.
       e.g. 2 features a,b, degree 2 -> 1, a, b, a*a, a*b, b*b"""
    n, m = X.shape
    cols, labels = [np.ones(n)], ["1"]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(m), d):
            cols.append(np.prod(X[:, combo], axis=1))
            labels.append("*".join(names[i] for i in combo))
    return np.column_stack(cols), labels

def fit(Xp, y):
    return np.linalg.lstsq(Xp, y, rcond=None)[0]

def metrics(y, p):
    r = y - p
    sse, sst = np.sum(r**2), np.sum((y - y.mean())**2)
    return {"MSE": sse / len(y), "RMSE": np.sqrt(sse / len(y)),
            "MAE": np.mean(np.abs(r)), "R2": 1 - sse / sst}

# ---------------------------------------------------------------
# 1. Load and explore
# ---------------------------------------------------------------
df = pd.read_csv("Car_MPG.csv")
print(df.head(), "\n")
print("Shape:", df.shape, "\nMissing values:\n", df.isnull().sum(), "\n")
df = df.dropna()
print(df.describe().round(2), "\n")
corr = df.corr()
print("Correlation with MPG:\n", corr["MPG"].round(3), "\n")

feat_names = ["Horsepower", "Weight_kg", "EngineSize_L", "Age_years"]
short = ["HP", "Wt", "Eng", "Age"]
X = df[feat_names].values.astype(float)
y = df["MPG"].values.astype(float)

# explore: nonlinear pattern + correlation heat-map
fig, ax = plt.subplots(1, 5, figsize=(21, 4))
for a, c in zip(ax[:4], feat_names):
    a.scatter(df[c], df["MPG"], s=12); a.set(title=f"MPG vs {c}", xlabel=c, ylabel="MPG")
im = ax[4].imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax[4].set_xticks(range(5)); ax[4].set_yticks(range(5))
ax[4].set_xticklabels(corr.columns, rotation=45, ha="right"); ax[4].set_yticklabels(corr.columns)
ax[4].set_title("Correlation matrix"); plt.colorbar(im, ax=ax[4])
plt.tight_layout(); plt.savefig("task2_explore.png", dpi=110); plt.show()

# ---------------------------------------------------------------
# 2. Split 70/30 and standardise (train mean/std)
# ---------------------------------------------------------------
rng = np.random.default_rng(3)
idx = rng.permutation(len(X)); cut = int(0.7 * len(X))
tr, te = idx[:cut], idx[cut:]
mu, sd = X[tr].mean(0), X[tr].std(0)
Z_tr, Z_te = (X[tr] - mu) / sd, (X[te] - mu) / sd
y_tr, y_te = y[tr], y[te]
print("Train:", Z_tr.shape, "Test:", Z_te.shape, "\n")

# ---------------------------------------------------------------
# 3. Feature engineering - look at the new features
# ---------------------------------------------------------------
Xp, labels = poly_features(Z_tr, 2, short)
print("Degree-2 features created from 4 variables:", len(labels), "->", labels, "\n")
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, lab, ttl in [(ax[0], "HP", "HP (original)"), (ax[1], "HP*HP", "HP squared (new)"), (ax[2], "HP*Wt", "HP x Weight (interaction)")]:
    a.scatter(Xp[:, labels.index(lab)], y_tr, s=12, color="tab:purple")
    a.set(title=f"MPG vs {ttl}", xlabel=lab, ylabel="MPG")
plt.tight_layout(); plt.savefig("task2_engineered_features.png", dpi=110); plt.show()

# ---------------------------------------------------------------
# 4. Experiment with degree
# ---------------------------------------------------------------
rows = []
for deg in [1, 2, 3, 4]:
    A_tr, _ = poly_features(Z_tr, deg, short); A_te, _ = poly_features(Z_te, deg, short)
    b = fit(A_tr, y_tr)
    mt, ms = metrics(y_tr, A_tr @ b), metrics(y_te, A_te @ b)
    rows.append([deg, A_tr.shape[1], mt["MSE"], ms["MSE"], mt["R2"], ms["R2"]])
res = pd.DataFrame(rows, columns=["Degree", "Num_terms", "Train_MSE", "Test_MSE", "Train_R2", "Test_R2"])
print("EFFECT OF DEGREE (all 4 features)\n", res.round(3).to_string(index=False))
# simplest degree whose test error is within 5% of the minimum (Occam's razor)
best = int(res[res["Test_MSE"] <= 1.05 * res["Test_MSE"].min()]["Degree"].min())
print(f"\nChosen degree = {best}  (simplest degree within 5% of the lowest test MSE)\n")

# ---------------------------------------------------------------
# 5. Final model + coefficients
# ---------------------------------------------------------------
A_tr, lab = poly_features(Z_tr, best, short); A_te, _ = poly_features(Z_te, best, short)
beta = fit(A_tr, y_tr)
p_te = A_te @ beta
m = metrics(y_te, p_te)
print(f"FINAL MODEL degree {best}:  Test MSE={m['MSE']:.3f}  RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  R2={m['R2']:.4f}")
coef = pd.DataFrame({"term": lab, "coefficient": beta.round(4)})
coef["abs"] = coef.coefficient.abs()
print("\nLargest coefficients (inputs are standardised, so sizes are comparable):")
print(coef.sort_values("abs", ascending=False).head(8).drop(columns="abs").to_string(index=False), "\n")

# ---------------------------------------------------------------
# 6. Feature selection : which variables are worth expanding?
# ---------------------------------------------------------------
print("FEATURE SELECTION (degree = %d)" % best)
subsets = {"HP only": [0], "HP + Weight": [0, 1], "HP + Weight + Age": [0, 1, 3], "All 4 features": [0, 1, 2, 3]}
sel_rows = []
for nm, cols in subsets.items():
    a_tr, _ = poly_features(Z_tr[:, cols], best, [short[i] for i in cols])
    a_te, _ = poly_features(Z_te[:, cols], best, [short[i] for i in cols])
    b = fit(a_tr, y_tr); ms = metrics(y_te, a_te @ b)
    sel_rows.append([nm, a_tr.shape[1], ms["MSE"], ms["R2"]])
print(pd.DataFrame(sel_rows, columns=["Features", "Num_terms", "Test_MSE", "Test_R2"]).round(3).to_string(index=False), "\n")

# ---------------------------------------------------------------
# 7. Visualise predictions
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].scatter(y_te, p_te); lim = [y.min(), y.max()]
ax[0].plot(lim, lim, "r--"); ax[0].set(title=f"Test: Actual vs Predicted (degree {best})", xlabel="Actual MPG", ylabel="Predicted MPG")
ax[1].plot(res.Degree, res.Train_MSE, "o-", label="Train MSE"); ax[1].plot(res.Degree, res.Test_MSE, "s-", label="Test MSE")
ax[1].set(title="Degree vs Error", xlabel="Degree", ylabel="MSE"); ax[1].legend()
ax[2].scatter(p_te, y_te - p_te, color="tab:green"); ax[2].axhline(0, color="r", ls="--")
ax[2].set(title="Residuals (test)", xlabel="Predicted", ylabel="Actual - Predicted")
plt.tight_layout(); plt.savefig("task2_results.png", dpi=110); plt.show()

# ---------------------------------------------------------------
# 8. Predict a new car
# ---------------------------------------------------------------
new = np.array([[150, 1700, 2.2, 5]], dtype=float)
a_new, _ = poly_features((new - mu) / sd, best, short)
print(f"New car (HP 150, 1700 kg, 2.2 L, 5 yrs) -> predicted MPG = {(a_new @ beta)[0]:.2f}")

print(f"""
DISCUSSION
* MPG falls in a CURVE with horsepower (steep for small engines, flatter later), so degree 1
  underfits; degree {best} captures the curve and interactions (e.g. HP x Weight).
* Number of terms grows fast: 4 features -> 5 terms (deg 1), 15 (deg 2), 35 (deg 3), 70 (deg 4).
  With only {len(y_tr)} training rows, high degrees start to memorise noise (train error falls, test error rises).
* Feature selection matters: expanding only the useful variables gives almost the same test
  accuracy with far fewer terms -> simpler, more stable model.
* Trade-off: more complexity = lower bias but higher variance. Pick the simplest model with good TEST score.
""")
