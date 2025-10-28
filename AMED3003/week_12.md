# Introduction to Medical Laboratory Regulatory Standards & Quality Assurance

Accreditation & Regulatory Frameworks

- Primary standard: AS ISO 15189:2023 – Medical laboratories: Requirements for quality and competence.

Regulatory bodies:

- NATA (National Association of Testing Authorities) – accredits labs in Australia.
- SAI Global – certification and compliance.
- TGA & NSQHS – additional oversight for clinical diagnostics.

### Laboratory Quality Management System (QMS)

Accredited laboratories must maintain a QMS that ensures reliability and traceability of all test
results. The system comprises four key structural domains:

1. Structural & Governance – defined responsibilities, documentation, and compliance.
2. Resources – personnel competence, facilities, calibrated equipment, reagents, and service
   agreements.
3. Processes – includes:
   - Pre-examination: sample collection, handling, and storage.
   - Examination: method validation / verification.
   - Post-examination: result interpretation, reporting, and storage.
4. Management systems – document control, auditing, and corrective action for non-conformances.

# Validation vs Verification of Bioanalytical Methods

What is Method Validation?

The process of defining analytical requirements and confirming that a method meets performance
expectations for its intended clinical purpose.

Essentially, validation proves a new or modified method is “fit-for-purpose.”

It involves experimental assessment of analytical parameters such as:

- Linearity
- Accuracy and Precision
- Sensitivity (LOD/LOQ)
- Selectivity (Specificity)
- Matrix Effects
- Ruggedness (Robustness)
- Stability
- Measurement Uncertainty

> [!NOTE] Case Study
> The Theranos scandal is a real-world example of insufficient method validation leading to unreliable
> clinical data and patient harm.

## What is Method Verification?

Verification determines whether a commercial or existing validated method performs as expected in the
end-user’s environment.

- Confirms reproducibility of manufacturer-supplied validation data.
- Required when implementing FDA-/TGA-approved kits or established reference methods.

When to Validate vs Verify:
| Scenario | Required Process |
| --------------------------------------------- | ---------------- |
| New or modified in-house method | **Validation** |
| Commercial kit or previously validated method | **Verification** |

# Evaluating the Performance of a Bioanalytical Method

Each method must meet defined performance characteristics, tested quantitatively:
| **Characteristic** | **Definition / Purpose** |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| **Linearity** | Ability of response to be directly proportional to analyte concentration (R² ≈ 0.98 – 1.00). |
| **Precision** | Closeness of independent test results. Expressed as %RSD or %CV (< 5 %). |
| • _Repeatability_ | Same operator, same equipment, short interval. |
| • _Reproducibility_ | Different operators, instruments, sites, times. |
| **Accuracy (Trueness/Bias)** | Closeness to the true or reference value (< 5 % bias). |
| **Sensitivity (LOD/LOQ)** | LOD = lowest detectable level; LOQ = lowest quantifiable level. |
| **Selectivity / Specificity** | Ability to measure the analyte without interference. |
| **Matrix Effects** | Influence of other sample components (e.g., haemolysis, icterus, lipaemia). |
| **Ruggedness (Robustness)** | Resistance to minor procedural changes. |
| **Stability** | Chemical stability of the analyte over time. |
| **Concordance / Comparability** | Agreement between different methods (Linear regression & Bland-Altman analysis). |
| **Measurement Uncertainty (MU)** | Numerical estimate of potential result dispersion. |

###### Example of Performance Evaluation

Aim: Compare cerebrospinal-fluid dopamine assay results between:

- Reference Lab (“gold standard”),
- Lab 1 (fully validated), and
- Lab 2 (minimal validation).

Statistical Tools (GraphPad Prism):

- Linear Regression – determines correlation, slope, and intercept.
- Bland-Altman Analysis – assesses bias and limits of agreement.

Performance summary criteria:

- R² = 0.98 – 1.00
- %CV < 5 %
- < 5 % bias
- Minimal matrix effects
- Good concordance

# Key Takeaways

Accredited labs operate under ISO 15189 with stringent QMS requirements.

Validation = prove a new method works; Verification = confirm an existing one performs as expected.

Quality assurance relies on systematic evaluation of precision, accuracy, linearity, and uncertainty.

Statistical comparison methods (e.g., Linear regression, Bland-Altman) ensure traceability to
reference standards.

Proper validation prevents clinical errors and upholds the integrity of diagnostic testing.
