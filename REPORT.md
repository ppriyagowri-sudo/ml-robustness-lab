# Robustness of Classifiers under Noise, Missing Data and Distribution Shift

## 1. Objective
To measure how much the accuracy of two common classifiers drops when test data is degraded in three realistic ways, and whether one model is more robust than the other.

## 2. Method
- **Data:** Breast Cancer Wisconsin dataset (569 samples, 30 numeric features), 70/30 stratified split, seed 42.
- **Models:** Logistic Regression (with standard scaling) and Random Forest (200 trees), both trained once on clean data.
- **Tests (applied to the test set only):**
  - Gaussian noise: noise scaled to each feature's standard deviation (0 to 2.0 std), 20 repeats.
  - Missing values: 0% to 50% of values removed and filled with training means, 20 repeats.
  - Distribution shift: all features shifted by -1.0 to +1.0 std.
- **Metrics:** accuracy for all tests; malignant recall for the shift test.

## 3. Results
| Test | Level | Logistic Regression | Random Forest |
|---|---|---|---|
| Clean | none | 0.9883 | 0.9415 |
| Noise | 0.5 std | 0.9506 | 0.9254 |
| Noise | 2.0 std | 0.8018 | 0.7640 |
| Missing | 20% | 0.9675 | 0.9404 |
| Missing | 50% | 0.9161 | 0.8596 |
| Shift | -1.0 std (recall) | 0.3438 | 0.3281 |
| Shift | +1.0 std (accuracy) | 0.4211 | 0.4327 |

## 4. Findings
1. **Random Forest loses less accuracy under mild corruption**, but Logistic Regression scores higher in absolute terms at every noise and missing-value level.
2. **Both models fail under heavy corruption.** At 2.0 std noise, accuracy falls to about 0.80 and 0.76.
3. **Random Forest collapses faster at 50% missing values**, the only case where it drops further than Logistic Regression.
4. **Distribution shift is the most dangerous test and is asymmetric.** A downward shift of 1 std makes both models miss about two thirds of malignant cases. An upward shift gives recall of 1.0, but only because almost every case is flagged malignant, and accuracy falls to about 0.42.

## 5. Discussion
Which model is "more robust" depends on the question. If the goal is the smallest drop from the clean score, Random Forest wins under mild noise and missing data. If the goal is the highest final accuracy, Logistic Regression wins throughout. For a medical setting, the shift result matters most: a model that silently misses cancers when instrument readings drift low is unsafe even if accuracy looks acceptable. Monitoring input distributions in production would be needed before deploying either model.

Possible explanations, which were not tested here: correlated features (radius, perimeter, area) may let Random Forest find alternative splits when some values are missing, and malignant tumors tend to have larger measurements, so a downward shift makes them resemble benign ones.

## 6. Limitations
- One small dataset and one train/test split (171 test rows; one sample is about 0.6%).
- Default hyperparameters, no tuning.
- The shift moves all 30 features together, which is harsher than typical real-world drift.
- Only mean imputation was tested for missing values.
- Only two models were compared.

## 7. Future work
- Shift only a subset of features.
- Test other imputation methods.
- Add more models, such as gradient boosting and SVM.
- Repeat on a second dataset to check that the findings generalise.
