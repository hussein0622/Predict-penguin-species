# Penguin Species Classification

A multiclass classification project predicting penguin species (Adelie / Chinstrap / Gentoo) from physical measurements using the Palmer Penguins dataset.

## Dataset

| Property | Value |
|---|---|
| Source | `Classic Datasets/penguins.csv` |
| Rows | 344 (342 after cleaning) |
| Features | 7 (4 numeric, 3 categorical) |
| Target | `species` — 3 classes |

**Features:** `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `island`, `sex`

**Class distribution:** Adelie (152), Gentoo (124), Chinstrap (68)

## Workflow

1. **Data Loading & Inspection** — shape, dtypes, missing values
2. **EDA** — pairplot, violin plots, correlation heatmap, key scatter plots
3. **Preprocessing** — drop 2 rows with missing measurements, impute `sex` by mode per species, one-hot encode `island`, label encode target
4. **Model Comparison** — 8 models with 5-fold stratified cross-validation
5. **Hyperparameter Tuning** — GridSearchCV on XGBoost and LightGBM
6. **Final Evaluation** — hold-out test set metrics + confusion matrix + feature importance
7. **Conclusions** — biological interpretation of results

## Models Compared

| Model | Accuracy (CV) | F1 weighted | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.996 | 0.996 | 1.000 |
| Random Forest | 0.989 | 0.989 | 1.000 |
| LightGBM | 0.989 | 0.989 | 1.000 |
| KNN | 0.985 | 0.985 | 1.000 |
| Gradient Boosting | 0.978 | 0.978 | 1.000 |
| XGBoost | 0.978 | 0.978 | 0.999 |
| Decision Tree | 0.949 | 0.948 | 0.960 |
| Baseline (majority) | 0.443 | 0.272 | 0.500 |

## Best Model

**LightGBM (tuned)** — Test set: Accuracy = 1.00, F1 = 1.00, ROC-AUC = 1.00

Best params: `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`, `num_leaves=31`

## Key Findings

- `bill_length_mm` + `bill_depth_mm` separates **Adelie from Chinstrap** (similar size, different bill shape)
- `flipper_length_mm` + `body_mass_g` cleanly separates **Gentoo** (significantly larger)
- `island` is informative but redundant with measurements
- Simple decision rule: `flipper_length_mm > 210` → Gentoo, else `bill_depth_mm < 18` → Adelie, else → Chinstrap

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
```
