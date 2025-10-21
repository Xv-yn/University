# Rationale for Population-Based Screening

## What is Screening?

Screening identifies individuals likely to have or develop a health condition.

It’s not diagnostic—it detects early indicators or risk markers.

Purpose:

- Detect diseases in asymptomatic populations early.
- Improve population health outcomes through early intervention and treatment.

Criteria for an Effective Screening Program:

- The condition should be common and serious.
- Early detection improves outcomes.
- Accurate, simple, and acceptable test.
- Accessible and cost-effective program.
- Effective treatment or management available.
- Continuous evaluation of program benefits and harms.

## Evaluating Screening and Diagnostic Tests

Key Questions

1. Can the test be reliably performed?
   - Accuracy: Closeness to the true value.
   - Precision: Consistency on repeated measures.
2. Was it evaluated in an appropriate population?
3. Was a valid gold standard used?
4. Was an optimal cut-off chosen to balance sensitivity and specificity?

## Statistical Evaluation: Sensitivity, Specificity, PPV, NPV

|            | Disease +           | Disease –           |
| ---------- | ------------------- | ------------------- |
| **Test +** | True Positive (TP)  | False Positive (FP) |
| **Test –** | False Negative (FN) | True Negative (TN)  |

Formulas

- Sensitivity = TP / (TP + FN)
  - Proportion of diseased correctly identified
  - “If someone has the disease, how likely is the test to detect it?”
- Specificity = TN / (TN + FP)
  - Proportion of non-diseased correctly identified
  - “If someone doesn’t have the disease, how likely is the test to say so?”
- Positive Predictive Value (PPV) = TP / (TP + FP)
  - Probability that a positive result means the person actually has the disease.
- Negative Predictive Value (NPV) = TN / (TN + FN)
  - Probability that a negative result means the person does not have the disease.

Effect of Prevalence

- PPV increases with higher disease prevalence.
- NPV increases when disease is rare.

Thus, screening tests are most useful in high-risk or high-prevalence populations.

## Applying Disease Frequencies

Example

A test for prostate cancer:

- Prevalence: 40%
- Sensitivity: 90%
- Specificity: 70%

The probability a person truly has cancer after a positive result depends not only on
sensitivity/specificity but also disease prevalence.

## Trade-Offs: Sensitivity vs. Specificity

Increasing sensitivity → captures more true cases but increases false positives.

- Cut-off lowered (e.g., SnNout: Sensitive test rules out disease if negative).

Increasing specificity → fewer false positives but may miss true cases.

Choosing the cut-off depends on:

- Severity of missing a diagnosis (false negative).
- Cost and harm of unnecessary further testing (false positive).

## Receiver Operating Characteristic (ROC) Curves

Definition

A plot of True Positive Rate (Sensitivity) vs. False Positive Rate (1 – Specificity) at different
cut-offs.

- Illustrates test performance across all thresholds.

Interpretation

- Diagonal line = random chance (no discriminative power).
- Closer to top-left corner = better test performance.
- Area Under Curve (AUC):
  - 1.0 = perfect test
  - 0.5 = worthless test

Clinical Use

- Compare diagnostic tests objectively.
- Identify optimal threshold for balance between sensitivity and specificity.
