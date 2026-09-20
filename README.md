# Machine Learning Model for Loan Approval

An end-to-end classification project that predicts whether a personal loan application will be **approved or denied**, using only information available at application time. Built for a fictional lender, *FinTech Innovations*, to triage a backlog of 20,000 manual underwriting decisions.

**Bottom line:** a tuned, class-weighted **logistic regression** reaches **ROC-AUC 0.976** and **F1 0.83** on held-out data, matching or beating a tuned Random Forest while staying fully interpretable, which matters in consumer lending, where adverse decisions must be explainable.

---

## Results

Evaluated on a stratified 20% test set (4,000 applications, 23.9% approved).

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression (tuned)** | **0.9757** | **0.8297** | 0.7573 | **0.9174** |
| Random Forest (tuned) | 0.9642 | 0.8159 | 0.7704 | 0.8672 |

**Estimated cost of misclassification** (custom business metric, on the test set):

| Policy | Estimated cost |
|---|---|
| **Logistic Regression (tuned)** | **$6.3M** |
| Random Forest (tuned) | $7.1M |
| Deny everyone | $25.6M |
| Approve everyone | $45.4M |

The logistic regression cuts estimated error cost by roughly **75% versus denying everyone**. The cost metric uses simplified, dataset-derived assumptions (about $26.8K profit per good loan and $14.9K loss per bad loan), so treat it as an order-of-magnitude estimate.

## Key findings

- **Affordability drives approval.** Income (`MonthlyIncome`, `AnnualIncome`), `LoanAmount`, and the engineered loan-to-income ratio are the strongest predictors, followed by `NetWorth`, `TotalAssets`, and `EducationLevel`. Credit-bureau features matter but rank lower.
- **The problem is close to linearly separable.** The logistic regression matches the Random Forest, and tuning the forest bought almost nothing.
- **Target leakage was found and removed.** `RiskScore` (r ≈ -0.77 with the outcome), `InterestRate`, `BaseInterestRate`, `MonthlyLoanPayment`, and `TotalDebtToIncomeRatio` appear to be set by the lender's process *after* the decision, so they are excluded from the features.
- **A fairness red flag.** All 901 applications with a missing `EducationLevel` were denied. The model learns this pattern, so it may be denying incomplete files rather than assessing creditworthiness. This needs a compliance review before any real deployment.

## Approach

1. **Business understanding:** stakeholders, cost of false positives vs. false negatives, and success criteria (ROC-AUC ≥ 0.90 and lower dollar cost than naive policies).
2. **EDA:** class balance (24% / 76%), distributions, approval rates by category, correlation analysis, and two data quality issues (leakage and non-random missingness).
3. **Preprocessing:** a single scikit-learn `Pipeline` with a `ColumnTransformer`:
   - numeric: median imputation and standard scaling
   - nominal categoricals: constant imputation (`"Missing"` kept as a signal) and one-hot encoding
   - `EducationLevel`: ordinal encoding (Missing < High School < … < Doctorate)
   - engineered feature: `LoanAmount / AnnualIncome`, added via `FeatureUnion`
4. **Modeling:** logistic regression and Random Forest, compared with 5-fold stratified cross-validation. Random Forest tuned with `RandomizedSearchCV`, logistic regression with `GridSearchCV`.
5. **Evaluation:** ROC-AUC, F1, precision, recall, the custom cost metric, confusion matrices, ROC curves, feature importance, and per-segment accuracy.

## Recommendation

Deploy the logistic regression as **decision support, not full automation**: auto-decide high-confidence approvals and denials, and route the borderline middle band to a human underwriter. Before go-live:

1. Resolve the missing-`EducationLevel` auto-deny pattern with compliance.
2. Run a fuller fairness audit across protected-class proxies such as age and marital status.
3. Replace the illustrative cost assumptions with real loss and margin figures.
4. Monitor approval rates and error costs after launch, and retrain periodically.

## Limitations

- The data looks **synthetic or rule-generated** (near-linear separability, very clean gradients), so real-world performance needs validation on fresh live data.
- Cost-metric assumptions are simplified (60% loss-given-default, full-term interest as profit).
- Fairness checks here are limited to per-segment accuracy, which is not sufficient for a regulated lending context.

## Getting started

```bash
git clone https://github.com/charlesesoro-cmyk/-Machine-Learning-Model-for-Loan-Approval.git
cd -Machine-Learning-Model-for-Loan-Approval

python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt

mkdir -p data
# place financial_loan_data.csv in data/

jupyter notebook financial_loan_risk.ipynb
```

The dataset is **not included** in this repository. The notebook expects it at `data/financial_loan_data.csv`.

## Repository structure

```
.
├── financial_loan_risk.ipynb   # full analysis: EDA, preprocessing, modeling, evaluation
├── requirements.txt
├── .gitignore                  # keeps data files out of version control
└── README.md
```

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter
