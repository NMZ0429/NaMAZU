from typing import Union

import cv2
import numpy as np
import onnxruntime as rt

from .utils import download_weight

WEIGHT_PATH = {
    "small": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/mono_depth_small.onnx",
    "large": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/mono_depth_large.onnx",
}

__all__ = ["MiDASInference"]


class MiDASInference:
    def __init__(
        self,
        model_path: str = "",
        model_type: str = "",
        download: bool = False,
        save_path: str = "",
        show_exp: bool = False,
    ) -> None:
        """MiDAS Inference class.

        Args:
            model_path (str): path to onnx file
            model_type (str): 'large' or 'small' to download weight. Defaults to ''.
            download (bool, optional): True to download weight. Defaults to False.
            save_path (str, optional): path to save weight. Defaults to ''.
            show_exp (bool, optional): True to display expected input size. Defaults to False.
        """
        model_path = self.__maybe_download(model_path, download, model_type, save_path)
        self.model_path = model_path
        self.session = rt.InferenceSession(self.model_path)
        self.__collect_setting(show_exp)

    def predict(self, img: Union[str, np.ndarray]) -> np.ndarray:
        """Predict.

        Args:
            img (Union[str, np.ndarray]): image path or numpy array in cv2 format
        
        Returns:
            np.ndarray: predicted depthmap
        """
        if isinstance(img, str):
            img = self._read_image(img)
        img = self.__preprocess(img)
        output = self.session.run([self.output_name], {self.input_name: img})[0]
        return output

    def render(
        self, prediction: np.ndarray, query: Union[str, np.ndarray]
    ) -> np.ndarray:
        """Return the resized depth map in cv2 foramt.

        Args:
            prediction (np.ndarray): predicted depthmap
            query (Union[str, np.ndarray]): query image path or numpy array in cv2 format used for resizing.

        Returns:
            np.ndarray: Resized depthmap
        """
        if isinstance(query, np.ndarray):
            orig_img_size = (query.shape[1], query.shape[0])
        else:
            orig_img_size = self._read_image(query).shape[:2][::-1]
        prediction = prediction.transpose((1, 2, 0))
        prediction = cv2.resize(
            prediction, orig_img_size, interpolation=cv2.INTER_CUBIC
        )
        return prediction

    def _read_image(self, path: str) -> np.ndarray:
        """Read image and output RGB image (0-1).
        Args:
            path (str): path to file
        Returns:
            array: RGB image (0-1)
        """
        img = cv2.imread(path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        return img

    def __preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image.

        1. Resize to training size.
        2. Normalize.
        3. Add batch dimension.
        4. Convert to float32.
        
        Args:
            img (np.ndarray): image in cv2 format
        """
        img = cv2.resize(img, (self.w, self.h))
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def __collect_setting(self, verbose: bool = True) -> None:
        """Collect setting from onnx file.

        Args:
            verbose (bool, optional): True to display setting. Defaults to True.
        """
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.h, self.w = self.input_shape[2], self.input_shape[3]

        if verbose:
            print(f"Input shape: {self.input_shape}")
            print("Normalization expected.")

    def __maybe_download(self, model_path, download, model_type, save_path) -> str:
        # parse input and download weight if desired
        if model_path == "":
            if download:
                if model_type == "large" or model_type == "small":
                    model_path = download_weight(WEIGHT_PATH[model_type], save_path)
                else:
                    raise ValueError("model_type must be either 'large' or 'small'.")
            else:
                raise ValueError("model_path is required.")

        return model_path
