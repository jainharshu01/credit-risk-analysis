import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer



def build_feature_pipeline(
        numeric_features, 
        categorical_features, 
        ordinal_features):

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
            ("ord", ordinal_pipeline, ordinal_features)
        ]
    )

    return preprocessor

def select_features_by_correlation(X, y, threshold=0.02):
    """
    Select features whose absolute correlation with target
    is above a given threshold.
    """
    corr = pd.DataFrame(X).corrwith(y).abs()
    selected_features = corr[corr > threshold].index.tolist()
    return selected_features
