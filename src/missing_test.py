import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from common import load_split, get_models, SEED

X_train, X_test, y_train, y_test = load_split()

# wrap each model with a median imputer (learned from training data only)
models = {
    name: make_pipeline(SimpleImputer(strategy="median"), model)
    for name, model in get_models().items()
}
for m in models.values():
    m.fit(X_train, y_train)

levels = [0, 0.05, 0.1, 0.2, 0.3, 0.5]   # fraction of test values deleted
n_repeats = 20

rows = []
for name, model in models.items():
    for level in levels:
        accs = []
        for r in range(n_repeats):
            rng = np.random.default_rng(SEED + r)
            mask = rng.random(X_test.shape) < level
            X_missing = X_test.mask(mask)   # sets masked cells to NaN
            accs.append(accuracy_score(y_test, model.predict(X_missing)))
        rows.append({
            "model": name,
            "missing_fraction": level,
            "mean_accuracy": round(np.mean(accs), 4),
            "std_accuracy": round(np.std(accs), 4),
        })

results = pd.DataFrame(rows)
print(results.to_string(index=False))
results.to_csv("results/missing_results.csv", index=False)
