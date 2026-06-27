# HMDA 2017 Mortgage Approval and Loan Amount Prediction

## Project Overview

This project uses the HMDA 2017 mortgage dataset to study mortgage loan applications in the United States.

The project has two main machine learning tasks:

1. **Classification**  
   Predict whether a loan application is approved or rejected.

2. **Regression**  
   Predict the loan amount for approved loan records.

The main goal is not only to train models but also to build a clean and meaningful data science workflow.  
This includes data understanding, cleaning, EDA, visualization, feature engineering, preprocessing, and later model comparison.

---

## Dataset

The dataset is based on HMDA mortgage application records.

Each row represents a loan application or loan record.  
The data includes information about:

- Loan type
- Loan purpose
- Applicant income
- Applicant demographics
- Property location
- Census tract information
- Loan decision
- Loan amount

The original dataset was very large, with about **14 million records**.

After cleaning and filtering, the cleaned dataset had about **8 million records**.

From this cleaned dataset, we created two final samples:

| Sample | Size | Purpose |
|---|---:|---|
| Classification sample | 500,000 rows | Predict loan approval |
| Regression sample | 500,000 rows | Predict loan amount for approved loans only |

Important note:

The **regression sample contains only approved loan records**.  
So the regression task is conditional on the loan already being approved.

---

## Target Variables

### Classification Target

The original HMDA outcome column was `action_taken`.

It had 8 possible values:

| Code | Meaning |
|---:|---|
| 1 | Loan originated |
| 2 | Application approved but not accepted |
| 3 | Application denied |
| 4 | Application withdrawn |
| 5 | File closed for incompleteness |
| 6 | Loan purchased |
| 7 | Preapproval request denied |
| 8 | Preapproval request approved but not accepted |

For binary classification, we mapped it as:

| action_taken | loan_approved |
|---|---:|
| 1, 2, 8 | 1 |
| 3, 7 | 0 |
| 4, 5, 6 | removed |

The final classification target is:

```text
loan_approved
