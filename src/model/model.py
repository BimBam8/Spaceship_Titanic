from src.features.features import build_feature_engineer, build_column_transformer
from src.features.pipeline import build_pipeline
from src.data_import.data import get_train_val_test_clean
import pickle


def train(X, y):
    pipeline = build_pipeline(build_feature_engineer(), build_column_transformer())
    pipeline.fit(X, y)
    return pipeline


def save(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_clean()
    # model = train(X_train, y_train)
    model = load("mlruns/1.pkl")
    print(model.predict(X_train))
    # save(model, "mlruns/1.pkl")
