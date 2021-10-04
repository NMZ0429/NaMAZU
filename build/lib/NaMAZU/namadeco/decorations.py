from functools import wraps
import time

__all__ = ["print_docstring", "measure_runtime"]


def print_docstring(func):
    """Decorator to print the docstring of the function at execution if provided arguments are not correct."""

    @wraps(func)
    def wrapper(*args, **kargs):
        try:
            return func(*args, **kargs)
        except TypeError:
            print(f"--- {func.__name__} ---\n{func.__doc__}")
            print("-" * (len(func.__name__) + 8))

    return wrapper


def measure_runtime(func):
    """Decorator to measure the runtime of the function at execution."""

    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        print(f"{func.__name__} tooks {time.time() - start} seconds.")
        return result

    return wrapper
