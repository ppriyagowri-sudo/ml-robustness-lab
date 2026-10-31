import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from common import load_split, get_models

X_train, X_test, y_train, y_test = load_split()
models = get_models()
for m in models.values():
    m.fit(X_train, y_train)

feature_std = X_train.std()
shifts = [-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0]   # multiples of each feature's std

rows = []
for name, model in models.items():
    for s in shifts:
        X_shifted = X_test + s * feature_std
        pred = model.predict(X_shifted)
        rows.append({
            "model": name,
            "shift": s,
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "malignant_recall": round(recall_score(y_test, pred, pos_label=0), 4),
        })

results = pd.DataFrame(rows)
print(results.to_string(index=False))
results.to_csv("results/shift_results.csv", index=False)
