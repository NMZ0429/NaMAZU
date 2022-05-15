from typing import Union
import onnxruntime as rt
import cv2 as cv
import numpy as np

Image = Union[str, np.ndarray]


class RealESRGANInference:
    def __init__(self, model: str) -> None:
        self.model = model
        self.session = rt.InferenceSession(self.model)
        _, _, self.w, self.h = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: Image) -> np.ndarray:
        image = self._read_image(image)
        input_image = self.__preprocess(image)
        result = self.session.run([self.output_name], {self.input_name: input_image})[0]

        return self.__postprocess(result)

    def _read_image(self, image: Image) -> np.ndarray:
        """Return cv2 image object if image is str, otherwise return image.
        Args:
            image (Union[str, np.ndarray]): image path or numpy array in cv2 format
        Returns:
            np.ndarray: cv2 image object
        """
        if isinstance(image, str):
            rtn = cv.imread(image)
        elif isinstance(image, np.ndarray):
            rtn = image
        else:
            raise TypeError("Input must be a path or cv2 image.")
        return rtn

    def __preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image.

        1. Resize to training size.
        2. BGR to RGB
        3. Chnnel swap
        4. Add batch dimension.
        5. Convert to float32.
        6. Normalize.

        Args:
            img (np.ndarray): image in cv2 format

        Returns:
            np.ndarray: preprocessed image
        """
        img = cv.resize(img, dsize=(self.h, self.w))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = img.astype("float32")
        img = img / 255.0

        return img

    def __postprocess(self, img: np.ndarray) -> np.ndarray:
        """Postprocess image.

        1. Remove batch dimension.
        2. Scale to 0-255.
        3. Chnnel swap
        4. RGB to BGR

        Args:
            img (np.ndarray): image in cv2 format

        Returns:
            np.ndarray: postprocessed image
        """
        hr_image = np.squeeze(img)
        hr_image = np.clip((hr_image * 255), 0, 255).astype(np.uint8)
        hr_image = hr_image.transpose(1, 2, 0)
        hr_image = cv.cvtColor(hr_image, cv.COLOR_RGB2BGR)

        return hr_image
