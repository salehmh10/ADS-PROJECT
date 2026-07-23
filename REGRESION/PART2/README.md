# Loan Amount Prediction — Regression Project

## Overview

This project develops and evaluates Machine Learning models for predicting the loan amount of approved loan applications.

The target column is:

`loan_amount_000s`

The target is measured in thousands of US dollars. For example, a target value of `250` represents approximately `$250,000`.

The final dataset contains:

- 399,788 training rows
- 99,948 test rows
- 499,736 rows in total

The primary evaluation metric is `MAE`.

## Project Workflow

The Regression workflow includes:

1. Data validation and Train/Test splitting
2. Missing-value handling and Categorical Encoding
3. Feature Engineering
4. Linear Model evaluation
5. Tree-Based Model evaluation
6. Boosting Model development
7. Deep Tabular Model evaluation
8. Ensemble analysis
9. Final Test evaluation
10. Error Analysis
11. Sensitive Feature and Fairness analysis
12. Model Explainability
13. Final visual and technical reporting

All trainable preprocessing steps were included inside the model Pipelines. The same fixed split and three Cross-Validation folds were used throughout the project.

## Models Evaluated

### Linear Models

- Linear Regression
- Ridge
- Lasso
- ElasticNet
- Gamma Regression

`Lasso` was the best representative of the Linear Model family.

### Tree-Based Models

- Decision Tree
- Random Forest
- HistGradientBoosting

`HistGradientBoosting` was the strongest Tree-Based representative.

### Boosting Models

- CatBoost
- LightGBM
- XGBoost

The official final model is a weighted Boosting Blend:

- CatBoost: 60%
- LightGBM: 20%
- XGBoost: 20%

### Deep Tabular Models

- RealMLP
- TabM
- FT-Transformer

`RealMLP` was the strongest Deep Learning model. `FT-Transformer` was the second-best Deep family, while `TabM` was evaluated but did not continue to the final comparison.

Both `raw` and `log1p` target modes were evaluated where applicable.

## Final Results

| Model | MAE | RMSE | R² | Evaluation Role |
|---|---:|---:|---:|---|
| Boosting Blend | 61.511631 | 125.788460 | 0.712206 | Official Test |
| RealMLP | 62.159745 | 125.651422 | 0.712833 | Later descriptive Test |
| RealMLP + Sensitive Features | 61.821114 | 125.020164 | 0.715711 | Accuracy-only descriptive Test |

The official `MAE` of `61.511631` target units is approximately equal to an average absolute error of `$61,512`.

The Boosting Blend remains the official model because it was selected before the Test Set was opened. The later RealMLP results are descriptive and cannot establish a new unbiased winner.

## Ensemble Analysis

A 50/50 combination of `RealMLP` and the Boosting Blend was also evaluated.

The Ensemble:

- Improved MAE by approximately 0.44%
- Improved upper-tail error
- Passed the paired Bootstrap MAE comparison
- Worsened RMSE by approximately 0.35%

The predefined maximum allowed RMSE worsening was 0.25%. Therefore, the Ensemble was rejected.

## Error Analysis

The main error-analysis findings were:

- Prediction errors increase for larger loan amounts.
- The highest loan-amount deciles have the largest MAE.
- The models tend to underpredict large loans.
- A small percentage of difficult rows creates a large share of the total absolute error.
- RealMLP performs better on some tail metrics, while the Boosting Blend has the best official overall MAE.

## Sensitive Feature Analysis

Models were evaluated both with and without the approved Sensitive Features.

The Sensitive Feature comparison was performed using the same model configuration, Target Mode, and Hyperparameters.

The results are accuracy comparisons only. They do not prove:

- Fairness
- Discrimination
- Causality
- Legal compliance
- Fairness of loan approval decisions

The dataset contains approved loan applications only, so approval fairness was not evaluated.

## Model Explainability

The project uses:

- Native Feature Importance
- SHAP
- Grouped Permutation Importance
- Local Reference Substitution

Important Feature groups include:

- Applicant income
- Lien status
- Owner occupancy
- Loan purpose
- Tract-income ratio
- Area income information
- State, County, and MSA/MD context
- Lender and respondent information

Feature Importance explains how the models use input data. It does not prove that a Feature causes the real loan amount.

## Repository Structure

```text
notebooks/   Final executed Regression notebooks
src/         Pipeline, utility, loader, and worker modules
figures/     Final statistical visualizations
reports/     Model Card and Technical Report
results/     Final summary tables
data/        Data description and access instructions
