from sklearn.model_selection import train_test_split
import pandas as pd


def drop(X, subset, how="any"):
    X = X.copy()
    Xc = X.dropna(subset=subset, how=how)

    return Xc


def get_train_val_test():
    df = pd.read_csv("data/train.csv")

    X = df.drop(["Transported"], axis=1)
    y = df["Transported"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_train_val_test_clean():
    df = pd.read_csv("data/train.csv")

    cat_cols = ["HomePlanet", "Cabin", "Destination"]

    df = drop(df, cat_cols)

    X = df.drop(["Transported"], axis=1)
    y = df["Transported"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
