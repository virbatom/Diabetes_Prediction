# Predicting Diabetes from Lifestyle and Health Indicators
 
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
 
A complete end-to-end machine-learning pipeline that predicts whether a patient
has diagnosed diabetes based on demographics, lifestyle habits, and clinical
biomarkers. Covering the full data-science
workflow: EDA, statistical testing, imputation, feature engineering, and
benchmarking three classifiers with cross-validation.
 
> **Author:** Tomáš Virba —  Econometrics & Operations Research, Prague
> University of Economics and Business (VŠE)
 
---
 
## 📊 Project at a Glance
 
| Detail | Value |
|---|---|
| **Dataset size** | 100,000 patients (aged 18–90) |
| **Features** | 31 variables across 4 categories |
| **Target** | `diagnosed_diabetes` (binary 0 / 1) |
| **Class balance** | ~60% diabetes / ~40% no diabetes |
| **Models compared** | Logistic Regression, Random Forest, Gradient Boosting |
| **🏆 Best result** | **ROC AUC = 0.947** (Gradient Boosting, 5-fold CV) |
 
---
 
## 🎯 Objective
 
Build and compare classification models to predict diabetes diagnosis, with
emphasis on:
 
1. Properly handling missing values (KNN vs. median imputation)
2. Identifying which features are actually significant predictors
3. Engineering domain-informed features (a composite obesity indicator
   following WHO sex- and ethnicity-specific thresholds)
4. Comparing a linear baseline (Logistic Regression) against two tree-based
   ensembles
5. Discussing the practical implications of the error pattern (false negatives
   vs. false positives in a medical context)
---
 
## 🗂️ Repository Structure
 
```
Diabetes_Prediction/
├── data/
│   └── diabetes_dataset.csv        # Raw dataset (100k rows × 31 cols)
├── Diabetes_Prediction.ipynb       # Main notebook (analysis + modelling)
├── Diabetes_Prediction.html        # Rendered notebook (view without running)
└── README.md
```
 
---
 
## 📋 Dataset
 
The dataset contains 31 variables grouped into four categories:
 
| Category | Examples |
|---|---|
| **Demographic** | age, gender, ethnicity, education, income, employment |
| **Lifestyle** | smoking, alcohol, physical activity, diet, sleep, screen time |
| **Clinical history** | family history of diabetes, hypertension, cardiovascular disease |
| **Biomarkers** | BMI, blood pressure, cholesterol, glucose, insulin, HbA1c |
 
---
 
## 🔬 Methodology
 
### 1. Missing values
The raw data has no missing values, so NaNs are artificially injected to
demonstrate two imputation strategies:
- **KNN imputation (k=3)** — preserves local feature relationships
- **Median imputation** — simple, fast baseline
### 2. Exploratory Data Analysis
- Distribution plots for all numeric and categorical features
- Correlation matrix (lower triangle, masked) revealing pairs like
  BMI ↔ waist-to-hip ratio (+0.77) and glucose ↔ HbA1c (+0.62)
- Kolmogorov–Smirnov normality tests (with N = 100k, every variable rejects
  normality — illustrating that p-values lose meaning at scale)
### 3. Statistical feature significance
Each feature is tested against the target:
- **χ² test of independence** for categorical variables
- **Welch t-test** (with Mann–Whitney U as non-parametric fallback) for
  numeric variables
This identifies that gender, ethnicity, education, income, employment, and
smoking are **not statistically significant** in this dataset.
 
### 4. Feature engineering
- **Gender encoding via KNN** — `"Other"` values set to NaN, imputed with
  k=5 nearest neighbours, then rounded to binary
- **`is_obese` composite indicator** — combines BMI, waist-to-hip ratio, and
  physical activity using sex- and ethnicity-specific WHO thresholds:
| Group | BMI | WHR | Activity (min/week) |
|---|---|---|---|
| Non-Asian female | > 25 | > 0.85 | < 150 |
| Non-Asian male | > 25 | > 1.00 | < 150 |
| Asian female | > 23.5 | > 0.85 | < 150 |
| Asian male | > 23.5 | > 1.00 | < 150 |
 
### 5. Modelling
Three models evaluated with **5-fold cross-validation** using ROC AUC as the
primary metric. The `diabetes_risk_score` column is excluded to prevent target
leakage. All numeric features are median-imputed and standardised inside a
`Pipeline` + `ColumnTransformer` to keep the workflow leak-free.
 
---
 
## 📈 Results
 
| Model | Mean ROC AUC (5-fold CV) |
|---|---|
| Logistic Regression | 0.934 |
| Random Forest | 0.943 |
| **Gradient Boosting** | **0.947** |
 
Tree-based models outperform the linear baseline by roughly one AUC point.
Shallow trees (depth 3) combined with a low learning rate (0.05) and
subsampling (0.8) allow Gradient Boosting to capture non-linear interactions
(e.g. HbA1c × fasting glucose) without overfitting.
 
### The false-negative problem
 
All three models share a characteristic error pattern:
 
- **Precision ≈ 1.00** — if the model says "diabetes", it's almost always right
- **Recall ≈ 0.87** — but it misses roughly 13% of actual diabetic patients
This is analogous to early COVID-19 PCR tests: high specificity, lower
sensitivity. In a clinical setting, this would warrant **threshold tuning** to
trade off some false positives for fewer missed cases.
 
---
 
## 🛠️ Tech Stack
 
- **Python 3.10+**
- **pandas**, **NumPy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scipy.stats** — statistical tests (KS, χ², t-test, Mann–Whitney U)
- **scikit-learn** — modelling, pipelines, cross-validation, KNN imputation
---
 
## 🚀 How to Run
 
### Option 1 — view the rendered notebook (no setup)
Open [`Diabetes_Prediction.html`](Diabetes_Prediction.html) in any browser.
 
### Option 2 — run the notebook locally
 
```bash
# 1. Clone the repository
git clone https://github.com/virbatom/Diabetes_Prediction.git
cd Diabetes_Prediction
 
# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
 
# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
 
# 4. Launch Jupyter
jupyter notebook Diabetes_Prediction.ipynb
```
 
The random seed is fixed (`SEED = 296`) so results are fully reproducible.
 
---
 
## 📝 Key Takeaways
 
1. **Gradient Boosting wins** — best ROC AUC (0.947) with ~92% overall
   accuracy.
2. **Feature engineering helps trees, not linear models** — the custom
   `is_obese` indicator gave marginal gains to Random Forest and Gradient
   Boosting but did not improve Logistic Regression.
3. **Dropping "insignificant" features hurts the linear model** — weak signals
   still contribute meaningfully to a linear combination, even when individual
   p-values look unimpressive.
4. **All errors are false negatives** — a clinically important asymmetry that
   would require post-hoc threshold tuning in any real deployment.
---
 
## 📬 Contact
 
Built by **Tomáš Virba** as part of a Master's-level data-science portfolio.
 
- GitHub: [@virbatom](https://github.com/virbatom)
If you find issues or have suggestions, feel free to open an Issue or PR.
 
---
 
*This project was developed for educational and portfolio purposes. It is not
intended for clinical use.*
