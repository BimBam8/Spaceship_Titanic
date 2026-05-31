from src.features.features import build_feature_engineer, build_column_transformer
from src.features.pipeline import build_pipeline
from src.data_import.data import get_train_val_test_clean

def train(X, y):
    pipeline = build_pipeline(build_feature_engineer(), build_column_transformer())
    pipeline.fit(X, y)
    return pipeline


if __name__=="__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_clean()
    model = train(X_train, y_train)
    print(model.predict(X_train))