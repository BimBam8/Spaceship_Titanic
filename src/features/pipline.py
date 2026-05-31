from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from pandas import DataFrame


def build_pipline(
    feature_engineer: DataFrame, col_transformer: ColumnTransformer, **params
) -> Pipeline:
    pipeline = Pipeline(
        [
            ("feature_engineering", feature_engineer),
            ("pre_proc", col_transformer),
            ("model", XGBClassifier(params, random_state=42)),
        ]
    )
    return pipeline
