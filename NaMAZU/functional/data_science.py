from typing import Dict, Iterable, List, Union, Tuple
from scipy.stats import t  # type: ignore

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
    "parse_tab_seperated_txt",
    "sxy_of",
    "sxx_of",
    "least_square_estimate",
    "estimate_variance_of_linear_regressor",
    "t_statistic_of_beta1",
    "calculate_CI_of_centred_model_at",
    "fit_general_least_square_regression",
    "get_prediction_interval",
    "t_stats_for_correlation",
    "get_p_value_of_tstat",
    "_search_t_table",
    "get_alt_sxx",
    "get_alt_sxy",
]

ArrayLike = Union[List[float], np.ndarray]


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


def calculate_sample_stats(ys: ArrayLike) -> Tuple[float, float]:
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


def test_population_proportion(
    n: int, num_positive: int, N: int = 0, alpha: float = 0.05, verbose: bool = False
) -> Tuple[float, float, float, Tuple[float, float]]:
    """Test the population proportion of the given number of positive samples.

    Args:
        n (int): Number of samples.
        num_positive (int): Number of positive samples.
        N (int, optional): Population size if known, 0 meaning unknown. Defaults to 0.
        alpha (float, optional): Confidence level. Defaults to 0.05.
        verbose (bool, optional): Print the results. Defaults to False.

    Returns:
        Tuple[float, float, float, Tuple[float, float]]: estimated proportion, sample_variance, error_bound, and confidence interval.
    """
    if n < num_positive:
        raise ValueError(f"n ({n}) must be greater than num_positive ({num_positive})")
    if N == 0:
        print(f"N is 0, assuming n/N is 0")
    if alpha != 0.05:
        raise NotImplementedError("alpha != 0.05")
    else:
        z_value = 2
    estimated_proportion = num_positive / n
    sample_variance = n * (n - 1) * (estimated_proportion) * (1 - estimated_proportion)
    f = 0 if N == 0 else (n / N)
    error_bound = z_value * np.sqrt(
        (estimated_proportion) * (1 - estimated_proportion) / (n - 1) * (1 - f)
    )

    ci = estimated_proportion - error_bound, estimated_proportion + error_bound

    if verbose:
        summary = f"""
        estimated_proportion: {estimated_proportion}
        sample_variance: {sample_variance}
        error_bound: {error_bound}
        {1 - alpha}% confidence interval: {ci}
        """
        print(summary)

    return estimated_proportion, sample_variance, error_bound, ci


def calculate_sufficient_n_for_proportion(
    N: int, error_bound: float, p: float = None
) -> int:
    """Calculates the number of samples enough to estimate population proportion within given error bound.

    Args:
        N (int): Population size.
        error_bound (float): Error bound.
        p (float, optional): Population proportion if known. Defaults to None.
    
    Returns:
        int: Succifient number of samples to estimate population proportion within given error bound.
    """
    if p is None:
        p = 0.5
    pq = p * (1 - p)
    D = error_bound ** 2 / 4
    return np.ceil(N * pq / ((N - 1) * D + pq))


###########################
### Regression Analysis ###
###########################


def __parse_list_or_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        x = x.reshape(-1, 1)
        print(f"reshaping x to {x.shape}, n = {x.shape[0]}")
    else:
        x = np.array(x).reshape(-1, 1)
        print(f"reshaping x to {x.shape}, n = {x.shape[0]}")

    return x


def parse_tab_seperated_txt(txt_path: str) -> Dict[str, List[float]]:
    """Parses a tab seperated text file into a dictionary of lists of floats.

    Args:
        txt_path: The path to the text file.
    
    Returns:
        Dict[str, List[float]]: A dictionary of lists of floats.
    """

    rtn = {}

    with open("q4.txt") as file:
        lines = file.readlines()
        col_names = lines[0].split("\t")
        data = lines[1:]
        for col in col_names:
            col_data = [float(line.split("\t")) for line in data]  # type: ignore
            rtn[col] = col_data

    return rtn


def sxy_of(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum((x - np.mean(x)) * (y - np.mean(y)))


def sxx_of(x: np.ndarray) -> float:
    return np.sum((x - np.mean(x)) ** 2)


def least_square_estimate(x: ArrayLike, y: ArrayLike) -> Tuple[float, float]:
    x = __parse_list_or_array(x)
    y = __parse_list_or_array(y)

    sxy = sxy_of(x, y)
    sxx = sxx_of(x)
    beta1 = sxy / sxx
    beta0 = np.mean(y) - beta1 * np.mean(x)

    return beta0, beta1


def estimate_variance_of_linear_regressor(
    x: ArrayLike, y: ArrayLike, beta1: float
) -> float:
    """calculates the variance of the linear regressor.

    FOrmula:
        sigma^2 = SSE / (n - 2)

    Args:
        x (ArrayLike): Independent variable of shape (n, 0).
        y (ArrayLike): Dependent variable of shape (n, 0).
        beta1 (float): The slope of the linear regressor.

    Returns:
        float: Estimated variance of the linear regressor.
    """
    x, y = __parse_list_or_array(x), __parse_list_or_array(y)
    syy = sxx_of(y)
    SSE = syy - beta1 * sxy_of(x, y)
    n = x.shape[0]
    return SSE / (n - 2)


def t_statistic_of_beta1(x: ArrayLike, y: ArrayLike, beta1: float) -> float:
    """Estimates the t-statistic for the beta1 parameter.

    The return value should be used to determine whether the linear regressor
    is statistically significant. 

    Args:
        x (ArrayLike): Numpy array or list of floats.
        y (ArrayLike): Numpy array or list of floats.
        beta1 (float): Slope of the linear regression.

    Returns:
        float: t-statistic for the beta1 parameter.
    """
    x, y = __parse_list_or_array(x), __parse_list_or_array(y)
    var = estimate_variance_of_linear_regressor(x=x, y=y, beta1=beta1)
    t_statistic = beta1 / (np.sqrt(var / sxx_of(x)))

    return t_statistic


def calculate_CI_of_centred_model_at(
    target_x: float,
    x: ArrayLike,
    y: ArrayLike,
    beta0: float,
    beta1: float,
    t_value: float,
    a0: float = None,
) -> Tuple[float, float]:
    """Calculates the confidence interval for centred the linear regression at
    given value x.

    At this moment, t_value must be provided by user.

    Args:
        target_x (float): The value at which to calculate the confidence interval.
        x (ArrayLike): Numpy array or list of floats.
        y (ArrayLike): Numpy array or list of floats.
        beta0 (float): Intercept of the linear regression.
        beta1 (float): Slope of the linear regression.
        t_value (float): t-statistic for the beta1 parameter.
        a0 (float): Special coefficient of beta0. Defaults to None.

    Returns:
        Tuple[float, float]: The confidence interval for the linear regression.
    """
    x, y = __parse_list_or_array(x), __parse_list_or_array(y)
    # calculate inside of root
    n = x.shape[0]
    x_mean = np.mean(x)
    target_x = 4000

    a = 1 if a0 is None else a0 ** 2
    inside_root = (a / n) + (target_x - x_mean) ** 2 / sxx_of(x)

    S = np.sqrt(estimate_variance_of_linear_regressor(x=x, y=y, beta1=beta1))

    error_bound = S * t_value * np.sqrt(inside_root)

    lower_bound = beta0 + beta1 * target_x - error_bound
    upper_bound = beta0 + beta1 * target_x + error_bound

    return (lower_bound, upper_bound)


def get_prediction_interval(
    x_val,
    x: np.ndarray,
    y: np.ndarray,
    t_value: float = None,
    alpha: float = 0.1,
    round_to_3: bool = False,
    debug: bool = False,
) -> tuple:
    if x.shape != y.shape:
        raise ValueError(
            "x and y must have the same shape, get x.shape={}, y.shape={}".format(
                x.shape, y.shape
            )
        )
    beta0, beta1 = least_square_estimate(x, y)
    predict = beta0 + beta1 * x_val

    n = len(x)
    if t_value is None:
        if alpha == 0.1:
            t_value = _search_t_table(alpha=alpha, degrees_of_freedom=n, two_side=False)
        else:
            raise NotImplementedError("alpha={} is not implemented".format(alpha))

    s = np.sqrt(estimate_variance_of_linear_regressor(x, y, beta1))

    root = np.sqrt(1 + (1 / n) + ((x_val - np.mean(x)) ** 2 / (sxx_of(x))))

    if round_to_3:
        # round to 3 digits
        predict = np.round(predict, 3)
        s = round(s, 3)
        root = round(root, 3)
        print("rounded to 3 decimal. s={}, root={}".format(s, root))

    if debug:
        print(f"t_value={t_value}", f"predict={predict}" f"s={s}", f"root={root}")
    error_bound = t_value * s * root

    return predict, error_bound


########################
# Correlation Analysis #
########################


def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """calculate correlation coefficient

    Formula:
        r = cov(x, y) / sqrt(sxx * syy)

    Args:
        x (np.ndarray): x
        y (np.ndarray): y

    Returns:
        float: correlation coefficient
    """

    sxy = sxy_of(x, y)
    sxx = sxx_of(x)
    syy = sxx_of(y)
    return sxy / np.sqrt(sxx * syy)


def t_stats_for_correlation(corr: float, n: int, text_precision: bool = False) -> float:
    """calculate t stats for correlation.

    Formulae from:
        t_stats = r * sqrt(n-2) / sqrt(1-r**2)

    Args:
        corr (float): correlation coefficient
        n (int): sample size

    Returns:
        float: t stats
    """
    if text_precision:
        corr = np.round(corr, 4)
    return corr * np.sqrt(n - 2) / np.sqrt(1 - corr ** 2)


def get_p_value_of_tstat(
    t_value: float, degrees_of_freedom: int
) -> Tuple[float, float, float]:
    """calculate p value of t stat.

    Args:
        t_value (float): t stat
        degrees_of_freedom (int): degrees of freedom

    Returns:
        Tuple[float, float, float]: negative, positive, non_zero test p value
    """
    one_sided_pval1 = t.cdf(t_value, degrees_of_freedom)  # 片側検定のp値 1
    one_sided_pval2 = t.sf(t_value, degrees_of_freedom)  # 片側検定のp値 2
    two_sided_pval = min(one_sided_pval1, one_sided_pval2) * 2

    return one_sided_pval1, one_sided_pval2, two_sided_pval


def fit_general_least_square_regression(
    x: np.ndarray, y: np.ndarray, quad_model: bool = False
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return the coefficients of the linear regression model and SSE and variance estimates.

    Formula:
        betas = (X.T @ X)^-1 @ X.T @ y
        SSE = Y.T @ Y - betas.T @ X.T @ Y
        variance = SSE / (n - 2)

    Args:
        x (np.ndarray): Independent variable. Shape: (n, p)
        y (np.ndarray): Dependent variable. Shape: (n, 1)
        quad_model (bool): If True, use quadratic model. Default is False.

    Returns:
        Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]: beta, (SSE, variance).
    """
    if quad_model:
        x = np.concatenate(
            (
                np.ones((x.shape[0], 1)),
                x[:, 1].reshape((-1, 1)),
                x[:, 1].reshape((-1, 1)) ** 2,
            ),
            axis=1,
        )
    n = x.shape[0]
    x_dash_x = np.matmul(x.T, x)
    x_dash_x_inv = np.linalg.inv(x_dash_x)
    x_dash_y = np.matmul(x.T, y)
    beta = np.matmul(x_dash_x_inv, x_dash_y)

    y_dash_y = np.matmul(y.T, y)
    sse = y_dash_y - np.matmul(beta.T, x_dash_y)

    return beta, (sse.item(), (sse / (n - 2)).item())


#####################
# Utility Functions #
#####################


def _search_t_table(
    alpha: float, degrees_of_freedom: int, two_side: bool, human_precision: int = 3
) -> float:
    """search t table.

    Args:
        alpha (float): alpha
        degrees_of_freedom (int): degrees of freedom
        two_side (bool): two side

    Returns:
        float: t value
    """
    if two_side:
        alpha = alpha / 2
    if degrees_of_freedom > 2000:
        print("Degrees of freedom is too large. Result is not precise.")
    return round(t.ppf(1 - alpha, degrees_of_freedom), human_precision)


def get_alt_sxx(x_sum: float, sqd_x_sum: float, n: int) -> float:
    """calculate sxx by given sum values"""

    return sqd_x_sum - 2 * x_sum * x_sum / n + x_sum ** 2 / n


def get_alt_sxy(x_sum: float, y_sum: float, xy_sum: float, n: int) -> float:
    """calculate sxy by given sum values"""
    return xy_sum - x_sum / n * y_sum - y_sum / n * x_sum + x_sum * y_sum / n
