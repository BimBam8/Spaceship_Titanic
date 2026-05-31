from src.features.features import build_feature_engineer, build_column_transformer
from src.features.pipeline import build_pipeline
from src.data_import.data import get_train_val_test_clean
import pickle
import json
import os


def train(X, y, **params):
    pipeline = build_pipeline(
        build_feature_engineer(), build_column_transformer(), **params
    )
    pipeline.fit(X, y)
    return pipeline


def train_from_run_specs(X, y, run_id: str):
    params = None
    with open(f"mlrun_setups/{run_id}.json", "rb") as f:
        params = json.load(f)
    pipeline = build_pipeline(
        build_feature_engineer(), build_column_transformer(), **params
    )
    pipeline.fit(X, y)
    return pipeline


def save(model, run: str, logs: dict = None):
    os.makedirs(os.path.dirname(f"mlruns/{run}/model.pkl"), exist_ok=True)
    with open(f"mlruns/{run}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    if logs is None:
        return
    with open(f"mlrun_setups/{run}.json", "w") as f:
        json.dump(logs, f)


def load(run: str):
    with open(f"mlruns/{run}/model.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_clean()
    model = train_from_run_specs(X_train, y_train, "1")
    model = load("mlruns/1.pkl")
    print(model.predict(X_train))
    params = {
        "n_estimators": 3,
    }
    
    # model = train(X_train, y_train, **{})
    save(model, "2")
