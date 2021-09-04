import numpy as np
from PIL import Image


def img_to_npy(jpg_path: str, save: bool = False, file_name: str = None) -> np.ndarray:
    """Return a numpy array from a jpg image.
    Args:
        img_path (str): The path to the img image.
        save (bool): Whether to save the numpy array to a file. Default is False.
        file_name (str): The name of the file to save to. Defaults to the jpg_path with the extension replaced by .npy.
    Returns:
        numpy.ndarray: Numpy array of the image
    """
    img = Image.open(jpg_path)
    img = np.asanyarray(img)

    ext = "." + jpg_path.split(".")[-1]

    if save:
        if file_name is None:
            file_name = jpg_path.replace(ext, ".npy")
        np.save(file_name, img)

    return img


def npy_to_png(npy_path: str, save: bool = False, file_name: str = None) -> np.ndarray:
    """Return a numpy array from a jpg image.
    Args:
        npy_path (str): The path to the npy image.
        save (bool): Whether to save the numpy array to a file. Default is False.
        file_name (str): The name of the file without .png to save to . Defaults to the npy_path with the extension replaced by .jpg.
    Returns:
        numpy.ndarray: Numpy array of the image
    """
    img = np.load(npy_path)
    img = Image.fromarray(img)

    if save:
        if file_name is None:
            file_name = npy_path.replace(".npy", ".png")
        img.save(file_name + ".png")

    return img


def split_image(
    img_path: str, direction: int = 0, save: bool = False, file_name: str = None
) -> None:
    """Split image by vertical centre line if direction is 0 otherwise by horizontal line.
    If save is True, resultant image is saved as file_name.

    Args:
        img_path (str): path to original
        direction (int, optional): 1 to split image along horizontal centre. Defaults to 0.
        save (bool, optional): Save resultant images. Defaults to False.
        file_name (str, optional): Output name. Defaults to None.

    Returns:
        None
    """
    img = Image.open(img_path)
    width, height = img.size

    if direction == 0:
        centre = width // 2
        img1 = img.crop((0, 0, centre, height))
        img2 = img.crop((centre, 0, width, height))
    else:
        centre = height // 2
        img1 = img.crop((0, 0, width, centre))
        img2 = img.crop((0, centre, width, height))

    if save:
        file_name = file_name if file_name else img_path
        img1.save(file_name + "_1" + ".png")
        img2.save(file_name + "_2" + ".png")
