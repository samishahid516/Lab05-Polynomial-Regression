# Lab 05: Polynomial Regression From Scratch

This lab implements polynomial regression with NumPy, pandas, and Matplotlib. It does not use scikit-learn: polynomial features, least-squares fitting, evaluation metrics, train/test splitting, scaling, and feature selection are implemented directly in the Python scripts.

## Contents

- `lab05_task1_polynomial_regression.py` - single-feature regression using salary and braking-distance data
- `lab05_task2_multi_feature_polynomial.py` - multiple-feature regression for car fuel economy
- `Position_Salaries.csv` - 10-level salary example
- `Braking_Distance.csv` - speed and braking-distance data
- `Car_MPG.csv` - horsepower, weight, engine size, age, and MPG data
- `make_datasets.py` - regenerates the synthetic datasets
- `LAB05_Tasks_and_Viva.md` - task details and viva questions

## Requirements

```bash
pip install numpy pandas matplotlib
```

Run both experiments from this folder:

```bash
python lab05_task1_polynomial_regression.py
python lab05_task2_multi_feature_polynomial.py
```

Each script prints evaluation results and saves PNG plots in the project folder. Matplotlib windows may also open while the scripts run.

## What the experiments show

### Task 1: Single feature

The salary example compares a straight line with degree 2, 4, and 5 polynomials. The degree-4 model fits the ten salary points closely and predicts a salary of approximately **158,862** at level 6.5. The degree-5 model has an even higher training R², but this small dataset makes that apparent improvement a warning sign for overfitting.

The braking-distance experiment uses a reproducible 70/30 train/test split:

| Model | Train MSE | Test MSE | Test R² |
|---|---:|---:|---:|
| Linear | 25.839 | 26.789 | 0.974 |
| Degree 2 | 2.929 | **2.867** | **0.997** |
| Degree 4 | 2.913 | 3.000 | 0.997 |
| Degree 5 | 2.854 | 2.895 | 0.997 |

Degree 2 is selected because it achieves the lowest test error with the simplest useful curve. Higher degrees add complexity without improving generalisation.

### Task 2: Multiple features

The car MPG model standardises the training features, creates polynomial terms and interactions, and evaluates degrees 1 through 4:

| Degree | Terms | Train MSE | Test MSE | Test R² |
|---:|---:|---:|---:|---:|
| 1 | 5 | 3.301 | 3.586 | 0.935 |
| 2 | 15 | 1.036 | **1.122** | **0.980** |
| 3 | 35 | 0.648 | 2.197 | 0.960 |
| 4 | 70 | 0.432 | 4.356 | 0.922 |

Degree 2 is chosen using the simplest-within-5%-of-best test-error rule. The final model predicts **17.64 MPG** for a car with 150 HP, 1700 kg, a 2.2 L engine, and age 5 years. Horsepower is the strongest predictor, followed by weight and age.

## Output screenshots

### Salary models: linear versus polynomial curves

![Task 1 salary polynomial comparison](task1_A_position_degrees.png)

The degree-4 curve follows the salary progression much more closely than the linear model. The green marker shows the prediction at level 6.5.

### Braking-distance model comparison

![Task 1 braking-distance results](task1_B_braking_degrees.png)

The left plot compares fitted curves, the middle plot compares training and test MSE, and the right plot shows why polynomial extrapolation outside the observed speed range is risky.

### Dataset exploration

![Task 2 dataset exploration](task2_explore.png)

The first four panels show MPG relationships with each input. The heat map highlights the strong negative relationships between MPG and horsepower, weight, and engine size.

### Engineered polynomial features

![Task 2 engineered features](task2_engineered_features.png)

The plots show an original feature, a squared feature, and an interaction term created by the custom `poly_features()` function.

### Final MPG results

![Task 2 final results](task2_results.png)

The actual-versus-predicted plot shows test performance, the middle plot shows the overfitting pattern as degree increases, and the residual plot checks the remaining prediction errors.

## Main conclusions

Polynomial terms allow a linear-in-the-coefficients model to represent curved relationships. Increasing the degree reduces training error, but it can increase test error by fitting noise. In both experiments, degree 2 provides the best balance between flexibility, accuracy, and interpretability.
