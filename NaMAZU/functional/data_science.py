from sklearn.linear_model import LinearRegression
import pandas as pd


__all__ = ["train_linear_regressor"]


def train_linear_regressor(df: pd.DataFrame, x: str, y: str):
    """
    Trains a linear regression model using the sklearn library.
    """
    regressor = LinearRegression()
    regressor.fit(df[x].values, df[y].values)
    return regressor
