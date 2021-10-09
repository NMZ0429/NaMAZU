from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

__all__ = [
    "train_linear_regressor",
    "calculate_sample_stats",
    "error_bound_of_mean",
    "estimated_total",
    "error_bound_of_total",
    "calculate_sufficient_n_for_population_total",
    "calculate_sufficient_n_for_mean",
]


def train_linear_regressor(df: pd.DataFrame, x: Union[str, List[str]], y: str):
    """
    Trains a linear regression model using the sklearn library.

    Args:
        df (pd.DataFrame): The dataframe to train the model on.
        x (Union[str, List[str]]): Single or list of column names to use as features.
        y (str): The column name of the target variable.

    Returns:
        sklearn.linear_model.LinearRegression: The trained model.
    """
    regressor = LinearRegression()
    regressor.fit(df[x].values, df[y].values)
    return regressor


#################################
### Sampling Theory Utilities ###
#################################


def calculate_sample_stats(ys: Union[np.ndarray, List[float]]) -> Tuple[float, float]:
    """Calculates the sample mean and variance of a list of values.

    Args:
        ys (Union[np.ndarray, List[float]]): List or array of values.

    Returns:
        Tuple[float]: Sample mean and sample variance.
    """
    mean: float = np.mean(ys)
    var: float = np.var(ys)
    var = var * len(ys) / (len(ys) - 1)  # corrected sample variance

    return (mean, var)


def error_bound_of_mean(
    N: int, n: int, sample_var: float, alpha: float = 0.05
) -> float:
    """Return the error bound of the estimation of population mean 
    given the number of samples and the sample variance.

    Args:
        N (int): Population size.
        n (int): Sample size.
        sample_var (float): Sample variance.
        alpha (float, optional): Confidence level. Defaults to 0.5.

    Returns:
        float: Error bound of the estimation of population mean. B.
    """
    sample_sd = np.sqrt(sample_var)
    if alpha != 0.05:
        raise NotImplementedError(
            "alpha != 0.05, Other values are not permitted at this moment."
        )
    else:
        z_value = 2
    return z_value * sample_sd * np.sqrt(1 - n / N) / np.sqrt(n)


def calculate_sufficient_n_for_mean(
    N: int,
    B: float,
    population_var: float = None,
    sample_var: float = None,
    sample_range: float = None,
    alpha: float = 0.05,
) -> float:
    """Calculates the number of samples enough to estimate population mean within given error bound B.

    One of variance parameters (population_var, sample_var, sample_range) must be provided to calculate the
    required number of samples.

    Args:
        N (int): Population size.
        B (float): Error bound.
        population_var (float, optional): Population variance if known. Defaults to None.
        sample_var (float, optional): Sample variance if known. Defaults to None.
        sample_range (float, optional): Sample range given by max - min if known. Defaults to None.
        alpha (float, optional): Confidence level. Defaults to 0.05.

    Raises:
        NotImplementError: If unsupported alpha value is provided.
        Exception: None of the three variance parameters are given. One of them must be given.

    Returns:
        float: Succifient number of samples to estimate population mean within given error bound B.
    """
    if alpha != 0.05:
        raise NotImplementedError(
            "alpha != 0.05, Other values are not permitted at this moment."
        )
    else:
        z_value = 2

    if population_var:
        print("population_var", population_var)
        v = population_var
    elif sample_var:
        print("sample_var", sample_var)
        v = sample_var
    elif sample_range:
        print("sample_range", sample_range)
        v = (sample_range / 4) ** 2
    else:
        raise Exception(
            "Must specify either population_var, sample_var, or sample_range"
        )
    D = (B ** 2) / (z_value ** 2)
    return (N * v) / ((N - 1) * D + v)


def estimated_total(N: int, sample_mean: float) -> float:
    """Calculates the estimated population total given the sample mean."""
    return sample_mean * N


def error_bound_of_total(N: int, n: int, sample_v: float, alpha: float = 0.05) -> float:
    """Return the error bound of the estimation of population total 
    given the number of samples and the sample variance.

    Args:
        N (int): Population size.
        n (int): Sample size.
        sample_v (float): Sample variance.
        alpha (float, optional): Confidence level. Defaults to 0.05.

    Returns:
        float: [description]
    """
    sample_sd = np.sqrt(sample_v)
    if alpha != 0.05:
        raise NotImplementedError("alpha != 0.05, Other values are not permitted")
    else:
        z_value = 2
    return z_value * N * sample_sd * np.sqrt(1 - n / N) / np.sqrt(n)


def calculate_sufficient_n_for_population_total(
    N: int,
    B: float,
    population_var: float = None,
    sample_var: float = None,
    sample_range: float = None,
) -> float:
    """Calculates the number of samples enough to estimate population toal within given error bound B.

    One of variance parameters (population_var, sample_var, sample_range) must be provided to calculate the
    required number of samples.

    Args:
        N (int): Population size.
        B (float): Error bound.
        population_var (float, optional): Population variance if known. Defaults to None.
        sample_var (float, optional): Sample variance if known. Defaults to None.
        sample_range (float, optional): Sample range given by max - min if known. Defaults to None.

    Raises:
        Exception: None of the three variance parameters are given. One of them must be given.

    Returns:
        float: Succifient number of samples to estimate population total within given error bound B.
    """

    if population_var:
        print("population_var", population_var)
        v = population_var
    elif sample_var:
        print("sample_var", sample_var)
        v = sample_var
    elif sample_range:
        print("sample_range", sample_range)
        v = (sample_range / 4) ** 2
    else:
        raise Exception(
            "Must specify either population_var, sample_var, or sample_range"
        )
    D = (B ** 2) / (4 * (N ** 2))
    return (N * v) / ((N - 1) * D + v)
