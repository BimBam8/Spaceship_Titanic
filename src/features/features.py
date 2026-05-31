from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def feature_engineer(X):
    X = X.copy()
    return X
def build_feature_engineer():
    return FunctionTransformer(feature_engineer)
def build_column_transformer() -> ColumnTransformer:
    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    cat_cols = ["HomePlanet", "Cabin", "Destination"]

    transformer = ColumnTransformer(
        transformers=[
            ("impute", SimpleImputer(), num_cols),
            ("categorical", OneHotEncoder(), cat_cols),
            ("numerical", StandardScaler(), num_cols),
        ]
    )

    return transformer
