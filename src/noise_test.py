import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from common import load_split, get_models, SEED

X_train, X_test, y_train, y_test = load_split()
models = get_models()
for m in models.values():
    m.fit(X_train, y_train)

feature_std = X_train.std().values
levels = [0, 0.1, 0.25, 0.5, 1.0, 2.0]   # noise as a multiple of each feature's std
n_repeats = 20

rows = []
for name, model in models.items():
    for level in levels:
        accs = []
        for r in range(n_repeats):
            rng = np.random.default_rng(SEED + r)
            noise = rng.normal(0, 1, X_test.shape) * feature_std * level
            X_noisy = X_test + noise
            accs.append(accuracy_score(y_test, model.predict(X_noisy)))
        rows.append({
            "model": name,
            "noise_level": level,
            "mean_accuracy": round(np.mean(accs), 4),
            "std_accuracy": round(np.std(accs), 4),
        })

results = pd.DataFrame(rows)
print(results.to_string(index=False))
results.to_csv("results/noise_results.csv", index=False)
