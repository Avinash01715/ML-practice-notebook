# Medical Insurance Cost Prediction

Predict annual medical insurance charges from demographic and lifestyle features using linear regression.

## Overview

This project explores the [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) (1,338 observations) and builds a baseline linear regression model to estimate insurance charges.

**Target variable:** `charges` (USD)

**Features used:**
- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region
- Engineered features (BMI category, binary encodings)


## Workflow

1. **Data Loading & Inspection**
   - Shape, dtypes, summary statistics
   - Missing-value check (none present)

2. **Exploratory Data Analysis**
   - Histograms and distribution of numeric features
   - Box plots for outlier inspection
   - Correlation heatmap (Pearson)
   - Relationship between smoker status and charges

3. **Data Cleaning & Preprocessing**
   - Duplicate removal
   - Categorical encoding (sex → `is_female`, smoker → `is_smoker`)
   - One-hot encoding for region
   - BMI categorization (underweight / normal / overweight / obese)
   - Feature scaling (StandardScaler) on continuous variables

4. **Feature Engineering & Selection**
   - Correlation analysis
   - Chi-square tests for categorical associations

5. **Modeling**
   - Train / test split (80 / 20, `random_state=42`)
   - Ordinary Least Squares Linear Regression
   - Evaluation: R² ≈ 0.80 and Adjusted R² on the held-out test set

## Key Findings (from the notebook)

- Smoking status is the strongest predictor of high charges.
- BMI and age also show clear positive relationships with charges.
- After encoding and scaling, a simple linear model explains roughly 80 % of the variance in test-set charges.

## Tech Stack

- Python 3
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (`LinearRegression`, `train_test_split`, metrics)
- SciPy (statistical tests)

## How to Run

1. Open `Project_1.ipynb` in Google Colab, Jupyter, or VS Code.
2. Ensure `insurance.csv` is in the same directory (or update the path).
3. Run all cells sequentially.

## Future Improvements

- Try non-linear models (Random Forest, Gradient Boosting, XGBoost)
- Add interaction terms (e.g., smoker × BMI)
- Hyperparameter tuning and cross-validation
- Residual analysis and assumption checks for linear regression
- Deploy a simple prediction API or Streamlit demo

---

*Part of an ML practice series.*
