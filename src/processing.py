from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Returns the column transformer that is to be applied to the columns
def column_transformer(
        categorical_predictors : list[str],
        numerical_predictors: list[str]):
    
    numeric = StandardScaler()
    enc = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers = [
        ("cat", enc, categorical_predictors),
        ("num", numeric, numerical_predictors)
        ],
    remainder="passthrough"
    )

    return preprocessor