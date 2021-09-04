import numpy as np
from PIL import Image


def jpg_to_npy(jpg_path: str, save: bool = True, file_name: str = None) -> np.ndarray:
    """Return a numpy array from a jpg image.
    Args:
        jpg_path (str): The path to the jpg image.
        save (bool): Whether to save the numpy array to a file. Default is True.
        file_name (str): The name of the file to save to. Defaults to the jpg_path with the extension replaced by .npy.
    Returns:
        numpy.ndarray: Numpy array of the image
    """
    img = Image.open(jpg_path)
    img = np.asanyarray(img)

    if save:
        if file_name is None:
            file_name = jpg_path.replace(".jpg", ".npy")
        np.save(file_name, img)

    return img


def npy_to_jpg(npy_path: str, save: bool = True, file_name: str = None) -> np.ndarray:
    """Return a numpy array from a jpg image.
    Args:
        npy_path (str): The path to the npy image.
        save (bool): Whether to save the numpy array to a file. Default is True.
        file_name (str): The name of the file to save to. Defaults to the npy_path with the extension replaced by .jpg.
    Returns:
        numpy.ndarray: Numpy array of the image
    """
    img = np.load(npy_path)
    img = Image.fromarray(img)

    if save:
        if file_name is None:
            file_name = npy_path.replace(".npy", ".jpg")
        img.save(file_name)

    return img
