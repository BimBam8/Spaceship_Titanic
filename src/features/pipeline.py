from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier


def build_pipeline(
    feature_engineer: FunctionTransformer, col_transformer: ColumnTransformer, **params
) -> Pipeline:
    pipeline = Pipeline(
        [
            ("feature_engineering", feature_engineer),
            ("pre_proc", col_transformer),
            ("model", XGBClassifier(**params, random_state=42)),
        ]
    )
    return pipeline
