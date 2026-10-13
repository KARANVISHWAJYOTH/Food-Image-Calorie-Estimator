# Dataset Audit

Fill this in after running `ml/notebooks/01_data_exploration.ipynb`. Treat outlier flags as candidates for review, not automatic errors.

## Class Balance Observations
- Food-101 classes/split counts reviewed:
- Notable imbalance or missing classes:
- Follow-up needed:

## Label Noise Observations
- Classes or images with questionable labels:
- Evidence and examples (class/image path):
- Follow-up needed:

## Nutrition Value Outliers
- Candidate dish IDs and flagged values:
- Review outcome and evidence:
- Follow-up needed:

## Label Alignment Summary
- Matching strategy and confidence threshold:
- Total Food-101 classes (denominator): ___
- Automatically matched: ___ classes (___% of all classes):
- Manually overridden: ___ classes (___% of all classes):
- Still unmatched: ___ classes (___% of all classes):
- Calculate each percentage as `class count / total Food-101 classes * 100`; use the results from `ml/data/combined_labels.csv` and `ml/data/unmatched_classes.csv`.
- Unmatched class list reviewed:
- **Modeling assumption, not ground truth:** matched Nutrition5k dish values are assigned as class-level Food-101 nutrition targets. Document this limitation in the final model documentation:

## Data Quality Decisions
| Decision | Rationale/evidence | Affected records | Date |
|---|---|---|---|
|  |  |  |  |
