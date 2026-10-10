# Lab 05 – Polynomial Regression (from scratch)

## 1. What is in this folder

| File | Purpose |
|---|---|
| `lab05_task1_polynomial_regression.py` | **Lab Task 1** – single feature, degree experiment, under/over-fitting |
| `lab05_task2_multi_feature_polynomial.py` | **Lab Task 2** – many features, polynomial feature engineering, feature selection |
| `Position_Salaries.csv` | 10 rows – the exact table from the lab manual ("Truth or Bluff") |
| `Braking_Distance.csv` | 80 rows – Speed → Braking distance (one feature, curved) |
| `Car_MPG.csv` | 200 rows – Horsepower, Weight, Engine size, Age → MPG (four features) |
| `make_datasets.py` | Regenerates the CSVs (the data is synthetic – Kaggle could not be downloaded; you may replace the CSVs with real ones that have the same columns) |

**Run:** `pip install numpy pandas matplotlib`, then `python lab05_task1_polynomial_regression.py` and `python lab05_task2_multi_feature_polynomial.py`.

**No scikit-learn is used.** `PolynomialFeatures` and `LinearRegression` are replaced by our own functions, so the equation is built by our own code.

---

## 2. What your tasks are (in simple words)

### Task 1 – Single feature
1. **Load** a dataset with one input (X) and one output (y).
2. **Choose a degree** and fit a polynomial: `y = b0 + b1·x + b2·x² + … + bn·xⁿ`.
3. **Predict on test data** and **evaluate** with MSE / RMSE / MAE / R².
4. **Plot** the curve with the data points.
5. **Try different degrees** and explain **underfitting vs overfitting**.

What the code does:
- *Part A* (`Position_Salaries.csv`) repeats the manual: linear fit vs degree 4, and predicts level 6.5.
- *Part B* (`Braking_Distance.csv`) splits 70/30 into train/test, tests degrees 1 to 10, and picks the simplest good degree.

### Task 2 – Multiple features
1. **Load & explore** a dataset with several features (shape, missing values, correlations).
2. **Visualise** to find non-linear patterns.
3. **Pre-process** (handle missing values, split, scale).
4. **Create polynomial features** (squares + interaction terms) and visualise them.
5. **Train, predict, evaluate** (MSE, R²), compare predicted vs actual.
6. **Experiment with degrees**, discuss model complexity vs performance.
7. **Discuss** insights and feature selection.

What the code does: `Car_MPG.csv` → heat-map and scatter plots → standardise → `poly_features()` builds all terms (e.g. `HP*HP`, `HP*Wt`) → degrees 1 to 4 → feature-subset comparison → predict a new car.

---

## 3. Core idea in 5 lines

1. A straight line cannot follow a curve, so we add powers of x as **new columns** (`x, x², x³ …`).
2. The model is still **linear in the coefficients** (b0, b1, b2…), so ordinary least squares still works. That is why it is called "polynomial *linear* regression".
3. We solve `β = (XᵀX)⁻¹Xᵀy` (we use `np.linalg.lstsq`, the numerically safe way).
4. Higher degree = more flexible = lower training error, but a risk of **overfitting**.
5. We pick the degree using **test-set error**, not training error.

---

## 4. Results you can quote

**Task 1 – Part A (Position_Salaries, whole data):**

| Degree | R² | Predicted salary, level 6.5 |
|---|---|---|
| 1 (linear) | 0.669 | 330,378.79 |
| 2 | 0.916 | 189,498.11 |
| 3 | 0.981 | 133,259.47 |
| 4 | 0.997 | **158,862.45** (matches the lab manual) |

The linear model overestimates the level 6.5 salary (330k vs about 159k). The curve fits much better.

**Task 1 – Part B (Braking_Distance, test set):** degree 1 test MSE ≈ 26.8 (underfit); degree 2 ≈ 2.87 (good); degree 10 ≈ 3.92 (test error rises, so overfitting).
Final equation: `Distance = 2.8431 + 0.15852·Speed + 0.00654772·Speed²`. Test R² = 0.997.

**Task 2 (Car_MPG, test set):**

| Degree | Terms | Train MSE | Test MSE |
|---|---|---|---|
| 1 | 5 | 3.30 | 3.59 |
| 2 | 15 | 1.04 | **1.12** |
| 3 | 35 | 0.65 | 2.20 |
| 4 | 70 | 0.43 | 4.36 |

Training error keeps falling, but test error rises after degree 2. This is the classic overfitting picture. Degree 2 has test R² = 0.98.

---

## 5. Viva questions and model answers

### A. Concept questions

**Q1. What is polynomial regression?**
It is regression where we use powers of the input (x, x², x³ …) as features to fit a curve: `y = b0 + b1x + b2x² + … + bnxⁿ`.

**Q2. Why is it called a *linear* model when the curve is not a straight line?**
Linear refers to the **coefficients**, not to x. y is a linear combination of b0…bn. After creating the columns x, x², x³ it is just multiple linear regression.

**Q3. When do we use it instead of simple linear regression?**
When the scatter plot shows a curve (points above/below the straight line in a pattern), or when residuals of the linear model show a pattern.

**Q4. What is the role of `PolynomialFeatures` and what did you do instead?**
It turns `x` into `[1, x, x², …]`. I wrote `poly_features()` that builds the same columns with `x**d`. For several variables it also creates interaction terms using `combinations_with_replacement`.

**Q5. What is an interaction term?**
A product of two different features (e.g. Horsepower × Weight). It lets the effect of one feature depend on the value of another.

**Q6. How are the coefficients calculated?**
By least squares: minimise the sum of squared errors. The closed form is `β = (XᵀX)⁻¹Xᵀy` (the normal equation). I use `np.linalg.lstsq`, which solves the same problem more stably.

**Q7. What does the column of ones in X_poly do?**
It gives the intercept b0.

**Q8. How many terms for n features and degree d?**
`C(n + d, d)` including the constant. For 4 features: degree 1 → 5, degree 2 → 15, degree 3 → 35, degree 4 → 70. So the model grows very fast.

### B. Under- and over-fitting

**Q9. What is underfitting?**
The model is too simple (e.g. a line for curved data). Both train and test error are high, which means **high bias**.

**Q10. What is overfitting?**
The model is too complex and learns the noise. Train error is very low but test error is high, which means **high variance**.

**Q11. How did you see overfitting in your results?**
In Task 2, train MSE fell from 1.04 to 0.43 from degree 2 to 4, but test MSE rose from 1.12 to 4.36. In Task 1, degree 10 had a higher test MSE than degree 2.

**Q12. What is the bias–variance trade-off?**
Increasing degree lowers bias (better fit to the true shape) but raises variance (the model changes a lot with different training data). We want the middle point.

**Q13. How did you choose the degree?**
I compared test MSE for each degree and chose the **simplest** degree whose test MSE was within 5% of the best (Occam's razor).

**Q14. How can you reduce overfitting?**
Lower the degree, use more data, remove useless features (feature selection), use regularisation (Ridge/Lasso), or cross-validation.

**Q15. Why not choose the degree with the highest training R²?**
Training R² always increases with degree, so it will always pick the most complex (overfit) model. Only unseen test data tells the truth.

### C. Data and preprocessing

**Q16. Why did the manual not split the 10-row Position_Salaries data?**
Only 10 points: too few to train and test separately, and every point is needed to learn the curve. In Part B I used a bigger dataset with a train/test split.

**Q17. Why did you standardise the features?**
High powers become huge (120⁸ ≈ 4×10¹⁶), which causes numerical problems. Standardising keeps values near 0 and makes coefficients comparable. I used the **train** mean/std also for the test data (no data leakage).

**Q18. What is data leakage?**
Using information from the test set during training or preprocessing. That is why the scaler's mean and std come only from the training data.

**Q19. How did you handle missing values?**
I checked `isnull().sum()` and used `dropna()`. (The datasets have none.)

**Q20. Why random shuffling before splitting?**
So the train and test sets are representative. I used a fixed seed so the results are reproducible.

### D. Evaluation

**Q21. Explain MSE, RMSE, MAE, R².**
MSE = mean of squared errors. RMSE = √MSE, in the same units as y. MAE = mean of absolute errors (less sensitive to outliers). R² = 1 − SSE/SST = fraction of the variance in y explained by the model.

**Q22. Can R² be negative?**
Yes, on test data, if the model is worse than just predicting the mean.

**Q23. Why is R² on training data not enough?**
It always grows with more terms. Compare with test R² (or adjusted R²).

**Q24. What do residual plots tell you?**
A random scatter around 0 means a good model. A visible pattern means the model is missing structure.

### E. Task 2 specific

**Q25. How did you find the non-linear pattern?**
Scatter plots of MPG vs each feature showed a curve for Horsepower, and the correlation heat-map showed strong negative correlations (HP −0.955, Weight −0.924).

**Q26. Why do HP, Weight and Engine size correlate with each other?**
Bigger engines have more power and more weight. This is **multicollinearity**: it makes individual coefficients unstable, but predictions stay fine.

**Q27. How did you do feature selection?**
I trained the same-degree model on different feature subsets: HP only (3 terms, test MSE 1.73) vs all four (15 terms, 1.12). Adding Weight and Age gave a big improvement (1.18 with 10 terms). Engine size added little.

**Q28. How do you interpret the coefficients?**
Inputs are standardised, so larger |coefficient| means a bigger effect. Horsepower has the largest negative effect on MPG (−4.69). Interaction and squared terms change that effect depending on other values.

**Q29. Which predictors are most influential?**
Horsepower first, then Weight, then Age (Engine size is mostly explained by HP).

**Q30. What is extrapolation, and why is it risky for polynomials?**
Predicting outside the training range. High-degree polynomials can swing wildly there (e.g. predicting 12 years in a model trained on 1–10).

### F. Code questions

**Q31. Explain `np.column_stack([x**d for d in range(degree+1)])`.**
It builds a matrix whose columns are x⁰ (ones), x¹, x², …, x^degree.

**Q32. Why `np.linalg.lstsq` instead of `inv(XᵀX)`?**
`XᵀX` of polynomial features is nearly singular for high degrees, so a direct inverse is inaccurate. `lstsq` uses a stable decomposition and gives the same answer in well-conditioned cases.

**Q33. What does `combinations_with_replacement(range(m), d)` do?**
It lists all combinations of d features, allowing repeats (e.g. (0,0) = HP², (0,1) = HP×Wt). Each one becomes a polynomial term.

**Q34. How would you do this with scikit-learn?**
`PolynomialFeatures(degree=d)` + `fit_transform(X)`, then `LinearRegression().fit(X_poly, y)`. Pipelines and `StandardScaler` can chain the steps. My code does the same math manually.

**Q35. What is the main limitation of polynomial regression?**
It overfits easily at high degrees, is sensitive to outliers, behaves badly outside the data range, and the number of features explodes with many variables.

---

## 6. Discussion text you can write in your report

> A straight line underfits curved data, giving high error on both training and test data. Adding polynomial terms lets the model follow the curve, and the error drops sharply from degree 1 to degree 2. Beyond that, extra terms only fit noise: training error keeps falling while test error rises, which is overfitting. The best model is the simplest one with a low test error (degree 2 in both tasks). For multiple features, polynomial expansion creates many terms (15 for degree 2, 70 for degree 4 with 4 variables), so feature selection and the choice of degree are essential to keep the model stable and interpretable.
