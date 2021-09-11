from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pytorch_lightning import LightningModule

from .instance import Query, Result

__all__ = ["PredictionAssistant"]


class PredictionAssistant(ABC):
    def __init__(self, models: Dict[str, Tuple[LightningModule, str]], *args, **kwargs):
        self.__models = self._load_models(models)
        self.args = args
        self.kwargs = kwargs

    def _load_models(self, models: Dict[str, Tuple[LightningModule, str]]):
        module_list = {}
        for k, v in models.items():
            model_func, ckpt = v
            module_list[k] = model_func.load_from_checkpoint(ckpt).eval()

        return module_list

    @abstractmethod
    def predict(self, query: Query) -> Result:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `infer`"
        )
