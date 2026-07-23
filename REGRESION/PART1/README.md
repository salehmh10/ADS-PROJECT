# HMDA Mortgage Loan Amount Prediction

## Overview

This repository contains the first regression notebook for a mortgage loan prediction project using the HMDA 2017 dataset.

The full project has two main tasks:

* **Classification:** predict whether a loan application is approved or rejected
* **Regression:** predict the loan amount for approved loan records

This notebook focuses only on the **regression preparation part**.
No final regression model is trained in this notebook yet.

---

## Dataset

The original dataset was based on HMDA 2017 mortgage records.

The raw data had about **14 million records**.

Before this notebook, the dataset was cleaned and reduced:

* Columns with very high missing values were removed
* Invalid rows were removed
* Duplicate rows were checked and removed
* Leakage columns were removed
* Only approved loan records were kept for the regression task
* The cleaned full dataset had about **8 million records**
* A random approved sample of **500000 records** was created for regression

The regression sample contains **approved loan records only**.

---

## Regression Target

The regression target is:

```text
loan_amount_000s
```

This column shows the loan amount in thousands of dollars.

Example:

```text
250 means 250000 dollars
```

The column `loan_approved` exists in the regression sample but it is always 1.
So it is not useful as a regression feature.

---

## Notebook

Main notebook:

```text
REGRESION_PART1.ipynb
```

This notebook is an initial EDA and feature engineering notebook for the regression task.

---

## What This Notebook Does

### 1. Load the regression sample

The notebook loads the 500K approved loan sample.

It also defines the main target column and basic project settings.

---

### 2. Check dataset structure

The notebook checks:

* Dataset shape
* Column names
* Data types
* Number of unique values in each column
* Sample values from each column

This step helps identify constant columns and high cardinality columns.

---

### 3. Define column groups

The columns are grouped into:

* Numeric columns
* Categorical columns
* Sensitive columns
* High cardinality columns

This makes later analysis and modeling easier.

---

### 4. Validate previous cleaning

The notebook checks whether earlier cleaning worked correctly.

It checks:

* Real missing values
* Zero values that act like missing values
* Invalid numeric ranges
* Exact duplicate rows

After checking duplicates, exact duplicate rows are removed from the dataframe.

---

### 5. Review outliers

The notebook does not remove outliers in this stage.

It only reviews extreme values using percentiles.

This is important because different models may need different outlier strategies.

---

### 6. Analyze the target variable

The target `loan_amount_000s` is analyzed using:

* Summary statistics
* Histogram
* Boxplot
* Log transformed distribution

Loan amount is right skewed.
This means most loans are in normal ranges but a small number of loans are very large.

For better visualization, some plots show loan amount in dollars and only up to the 99th percentile.
This does not change the original data.

---

### 7. Analyze numeric features

The notebook studies numeric features such as:

* Applicant income
* Population
* Minority population
* Area median family income
* Tract income ratio
* Owner occupied units
* 1 to 4 family housing units

It uses:

* Summary tables
* Histograms
* Boxplots
* Scatter plots
* Median target by numeric bins
* Pairplot
* Correlation heatmap

---

### 8. Analyze categorical features

The notebook studies categorical features such as:

* Loan type
* Loan purpose
* Preapproval status
* Property type
* Owner occupancy
* Lien status
* State
* Metro area
* County

For categorical features, the notebook shows:

* Group counts
* Mean loan amount by group
* Median loan amount by group
* Bar charts
* Boxplots using log loan amount

---

### 9. Analyze geographic features

The notebook checks location based patterns using:

* State
* Metro area
* County
* Census tract

It shows which states and metro areas have the most records and how median loan amount changes across locations.

---

### 10. Analyze sensitive features for audit

The notebook also explores sensitive or fairness related columns such as:

* Applicant race
* Applicant ethnicity
* Applicant sex
* Co applicant race
* Co applicant ethnicity
* Co applicant sex
* Minority population

These features are used only for understanding and audit.

The notebook does not make causal claims from these plots.

---

## Feature Engineering

The notebook creates several general features that may be useful for regression models.

### Numeric engineered features

| Feature                                | Meaning                                    |
| -------------------------------------- | ------------------------------------------ |
| `log1p_applicant_income`               | Log transformed applicant income           |
| `log1p_population`                     | Log transformed tract population           |
| `log1p_hud_median_family_income`       | Log transformed area median family income  |
| `log1p_owner_occupied_units`           | Log transformed owner occupied unit count  |
| `log1p_1_to_4_family_units`            | Log transformed 1 to 4 family unit count   |
| `applicant_income_to_area_income`      | Applicant income relative to area income   |
| `tract_income_ratio`                   | Tract income relative to metro area income |
| `owner_occupied_unit_ratio`            | Share of owner occupied units              |
| `family_units_per_1000_people`         | 1 to 4 family units per 1000 people        |
| `owner_occupied_units_per_1000_people` | Owner occupied units per 1000 people       |

### Categorical engineered features

| Feature                       | Meaning                                         |
| ----------------------------- | ----------------------------------------------- |
| `has_co_applicant`            | Whether the loan application has a co applicant |
| `loan_program_group`          | Conventional vs government backed loan          |
| `applicant_income_area_group` | Applicant income group relative to area income  |
| `tract_income_level`          | Tract income level relative to metro area       |
| `us_region`                   | Broad US region based on state                  |

### Audit feature

| Feature                   | Meaning                                            |
| ------------------------- | -------------------------------------------------- |
| `majority_minority_tract` | Whether minority population is at least 50 percent |

This feature is created for audit and should be used with care.

---

## Visual Feature Checks

For the new features, the notebook creates visual checks such as:

* Distribution of the feature
* Median loan amount by feature bins
* Bar charts for grouped features
* Comparisons of raw and log transformed values

The goal is not only to create new columns but also to understand whether they have a meaningful relation with the target.

---

## Base Modeling Data

At the end, the notebook creates a base modeling dataframe.

This dataframe:

* Keeps the regression target
* Keeps candidate input features
* Removes EDA helper columns
* Does not scale features
* Does not encode categorical variables

Scaling and encoding are left for model specific pipelines.

---

## Train Test Split

The notebook prepares a train and test split.

Because the target is skewed, the split uses target bins.

This helps keep small and large loan amounts in both train and test sets.

The notebook creates:

* `X_train_base`
* `X_test_base`
* `y_train_raw`
* `y_test_raw`
* `y_train_log`
* `y_test_log`

No model is trained yet.

---

## Main Idea

This notebook prepares the regression dataset before modeling.

It focuses on:

* Understanding the data
* Checking cleaning quality
* Exploring target distribution
* Exploring numeric and categorical features
* Creating useful general features
* Preparing a base dataset for later model pipelines

The next step is to build model specific preprocessing and compare regression models.

---

## Planned Next Steps

Future work includes:

* Linear Regression
* Ridge Regression
* LASSO Regression
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting models
* Model comparison using MAE RMSE and R2
* Error analysis
* Feature importance analysis

---

## Requirements

Main Python libraries used:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## Notes

The raw HMDA dataset is very large and should not be uploaded directly to GitHub.

The notebook assumes that the cleaned regression sample file is available locally:

```text
hmda_regression_approved_500k.csv
```
