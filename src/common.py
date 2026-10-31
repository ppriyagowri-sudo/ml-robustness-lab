import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

SEED = 42


def load_split():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)


def get_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
