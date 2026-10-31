import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from common import load_split, get_models

X_train, X_test, y_train, y_test = load_split()
print("Train:", X_train.shape, "Test:", X_test.shape)

rows = []
for name, model in get_models().items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    })

results = pd.DataFrame(rows)
print(results)
results.to_csv("results/baseline.csv", index=False)