from functional import *
from typing import List, Any


class Compose:
    """Composes a list of functions into a single function"""

    def __init__(self, functions: List):
        self.functions = functions
        self.per_func_input_dict = {}

    def __call__(self, input, *args: Any, **kwds: Any) -> Any:
        for func in self.functions[:-1]:
            input = func(input, *args, **kwds)
        return input
