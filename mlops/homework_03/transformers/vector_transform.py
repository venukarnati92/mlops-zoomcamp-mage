from typing import Tuple
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    categorical = ['PULocationID', 'DOLocationID']

    dv = DictVectorizer()
    lr = LinearRegression()

    train_dicts = df[categorical].to_dict(orient='records')

    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    lr.fit(X_train, y_train)

    print(lr.intercept_)

    return dv, lr 