import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def build_column_transformer()->ColumnTransformer:
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cat_cols = ['HomePlanet', 'Cabin', 'Destination']

    transformer = ColumnTransformer(transformers=[
        ('impute', SimpleImputer(), num_cols),
        ('categorical', OneHotEncoder(), cat_cols),
        ('numerical', StandardScaler(), num_cols)
    ])

    return transformer

if __name__ == "__main__":
    df = pd.read_csv('data/train.csv')

    X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Transported']),
                                                     df['Transported'],
                                                     test_size=0.3)
    transformer = build_column_transformer()
    transformer.fit(X_train)
    print(pd.DataFrame(transformer.transform(X_train)))

