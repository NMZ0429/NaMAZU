import os
from typing import Dict, Union, Tuple
import copy
import cv2 as cv

import numpy as np

from onnxruntime import InferenceSession

from NaMAZU.onnx_api.midas import WEIGHT_PATH
from .utils import download_weight

WEIGHT_PATH = {
    "basic": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/basic.onnx",
    "mobile": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/mobile.onnx",
    "human": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/human_seg.onnx",
    "portrait": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/portrait.onnx",
}


__all__ = ["U2NetInference"]


class U2NetInference:
    def __init__(self, model: str):
        if not os.path.exists(model):
            if model in WEIGHT_PATH:
                model_path = download_weight(WEIGHT_PATH[model])
            else:
                raise ValueError(
                    f"Downloadable model not found: Available models are {list(WEIGHT_PATH.keys())}"
                )
        else:
            model_path = model

        self.IS = InferenceSession(model_path)
        self.input_name = self.IS.get_inputs()[0].name
        self.output_name = self.IS.get_outputs()[0].name
        self.input_size = 320
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def predict(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Returns: masks
        """

        x = self._read_image(image)

        x = self.__preprocess(x)

        # Predict
        onnx_result = self.IS.run(
            [self.output_name], {self.input_name: x}
        )  # return list of np.ndarray

        # Postprocess
        onnx_result = np.array(onnx_result)
        onnx_result = self.__postprocess(onnx_result)

        return onnx_result

    def render(
        self, prediction: np.ndarray, image: Union[str, np.ndarray]
    ) -> np.ndarray:
        """Render prediction on image.

        Args:
            prediction (np.ndarray): Predicted mask in cv2 format.
            image (Union[str, np.ndarray]): Image in cv2 format or path to image.
        
        Returns:
            np.ndarray: Rendered image.
        """
        img = cv.imread(image) if isinstance(image, str) else image
        mask = copy.deepcopy(prediction)
        mask = cv.resize(mask, dsize=(img.shape[1], img.shape[0])) / 255

        return (img * mask.reshape(img.shape[0], img.shape[1], 1)).astype(np.uint8)

    def _read_image(self, image=Union[str, np.ndarray]) -> np.ndarray:
        """Return cv2 image object if given path. Otherwise return numpy array.

        Args:
            Union[str, np.ndarray]: image path or numpy array in cv2 format

        Returns:
            np.ndarray: RGB image.
        """
        if isinstance(image, str):
            rtn = cv.imread(image)
            rtn = cv.cvtColor(rtn, cv.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            rtn = image
            print("Image is pre-loaded. Make sure that the image is in RGB format.")
        else:
            raise ValueError(f"image must be str or np.ndarray. got {type(image)}")

        return rtn

    def __preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image.

        1. Resize to training size.
        2. Convert to float32.
        3. Normalize.
        4. Channel swap.
        5. Add batch dimension.

        Args:
            img (np.ndarray): image in cv2 format
        """
        x = cv.resize(img, dsize=(self.input_size, self.input_size))
        x = np.array(x, dtype=np.float32)
        x = (x / 255 - self.mean) / self.std
        x = x.transpose(2, 0, 1).astype("float32")
        x = x.reshape(-1, 3, self.input_size, self.input_size)

        return x

    def __postprocess(self, onnx_result: np.ndarray) -> np.ndarray:
        """Postprocess image.

        1. Remove batch dimension.
        2. Channel swap.
        3. Denormalize.
        4. Resize to original size.
        5. Convert to cv2 format.

        Args:
            img (np.ndarray): image in cv2 format
        """
        onnx_result = onnx_result.squeeze()
        min_value = np.min(onnx_result)
        max_value = np.max(onnx_result)
        onnx_result = (onnx_result - min_value) / (max_value - min_value)
        onnx_result *= 255
        onnx_result = onnx_result.astype("uint8")

        return onnx_result

