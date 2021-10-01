from os.path import join
from pathlib import Path
from typing import Callable, Tuple, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL.Image import Image
from tqdm import tqdm


__all__ = [
    "img_to_npy",
    "npy_to_png",
    "apply_mask_to",
    "split_image",
    "compose_two_png",
    "apply_to_all",
    "change_frame_rates_in",
    "save_all_frames",
    "collect_images",
]


#################
# Image Process #
#################


def img_to_npy(jpg_path: str, save: bool = False, file_name: str = None) -> np.ndarray:
    """Return a numpy array from a jpg image.
    Args:
        img_path (str): The path to the img image.
        save (bool): Whether to save the numpy array to a file. Default is False.
        file_name (str): The name of the file to save to. Defaults to the jpg_path with the extension replaced by .npy.
    Returns:
        numpy.ndarray: Numpy array of the image
    """
    img = PILImage.open(jpg_path)
    img = np.asanyarray(img)

    ext = "." + jpg_path.split(".")[-1]

    if save:
        if file_name is None:
            file_name = jpg_path.replace(ext, ".npy")
        np.save(file_name, img)

    return img


def npy_to_png(npy_path: str, save: bool = False, file_name: str = None) -> Image:
    """Return a numpy array from a jpg image.
    Args:
        npy_path (str): The path to the npy image.
        save (bool): Whether to save the numpy array to a file. Default is False.
        file_name (str): The name of the file without .png to save to . Defaults to the npy_path with the extension replaced by .jpg.
    Returns:
        Image: PIL Image object
    """
    img = np.load(npy_path)
    img = PILImage.fromarray(img)

    if save:
        if file_name is None:
            file_name = npy_path.replace(".npy", ".png")
        img.save(file_name + ".png")

    return img


def apply_mask_to(target_img: Union[Image, str], mask_img: Union[Image, str]) -> Image:
    """Apply mask to target image.

    Args:
        target_img (Union[Image, str]): PIL Image or path to image to apply mask to.
        mask_img (Union[Image, str]): PIL Image or path to mask image.

    Returns:
        Image: PIL Image object with applied mask.
    """
    if isinstance(target_img, str):
        target_img = PILImage.open(target_img)
    if isinstance(mask_img, str):
        mask_img = PILImage.open(mask_img)

    mask = mask_img.convert("L")
    target_img_cp = target_img.copy()
    target_img_cp.putalpha(mask)

    return target_img_cp


def split_image(
    img_path: str, direction: int = 0, save: bool = False, file_name: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Split image by vertical centre line if direction is 0 otherwise by horizontal line.
    If save is True, resultant image is saved as file_name.

    Args:
        img_path (str): path to original
        direction (int, optional): 1 to split image along horizontal centre. Defaults to 0.
        save (bool, optional): Save resultant images. Defaults to False.
        file_name (str, optional): Output name. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of numpy arrays.
    """
    img = PILImage.open(img_path)
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

    return np.asanyarray(img1), np.asanyarray(img2)


def compose_two_png(
    back_png_path: str,
    front_png_path: str,
    position: Tuple[int, int],
    save: bool = False,
    out_name: str = None,
) -> Image:
    """Overray second png to first png with given position. Save output png to save_path if save is True.

    Args:
        back_png_path (str): Path to background png.
        front_png_path (str): Path to foreground png.
        position (Tuple[int, int]): Position of overlay.
        save (bool, optional): Whether to save output. Defaults to False.
        out_name (str, optional): Name of output file. Defaults to None.

    Returns:
        Image: PIL Image object
    """
    back = PILImage.open(back_png_path)
    front = PILImage.open(front_png_path)

    back.paste(front, position, front)

    if save:
        if not out_name:
            out_name = front_png_path.replace(".png", "_compose.png")
        back.save(out_name)

    return back


def apply_to_all(func: Callable, imgs_dir: str, out_dir: str = None, **kwargs) -> None:
    """Apply function to all images in imgs_dir and save to out_dir.

    Args:
        function (function): Function to apply to images.
        imgs_dir (str): Path to directory containing images.
        out_dir (str, optional): Path to directory to save images. Use imgs_dir_output if None. Defaults to None.
        **kwargs: Keyword arguments to pass to function.

    Returns:
        None
    """
    if out_dir is None:
        out_dir = imgs_dir + "_output"

    inputs = list(Path(imgs_dir).glob("**/*.jpg"))
    inputs += list(Path(imgs_dir).glob("**/*.png"))
    inputs = sorted(list(map(str, inputs)))

    out = Path(out_dir)
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)

    print(
        f"Applying function\n  -{func}\nto all images in\n  -{imgs_dir}\nand saving to\n  -{out_dir}"
    )

    # TODO: Test other functions
    for x in tqdm(inputs):
        out = func(x, save=True, file_name=join(out_dir, x.split("/")[-1]), **kwargs)


#################
# Video Process #
#################


def change_frame_rates_in(mp4_dir: str, fps: int) -> None:
    """Change frame rate of all mp4 files in mp4_dir.

    Args:
        mp4_dir (str): Path to directory containing mp4 files.
        fps (int): New frame rate.
    """
    import subprocess

    subprocess.call("./fps_change.sh {} {}".format(mp4_dir, fps), shell=True)


def save_all_frames(
    video_path: str, dir_path: str, naming: str = "{}", ext: str = "png"
) -> int:
    """Save all frames from video_path to dir_path in a naming format with a given extension.
    Naming format must contains {} which will be replaced by the frame number.

    Args:
        video_path (str): Path to video file.
        dir_path (str): Path to directory to save frames.
        naming (str, optional): Naming rule of frames that must contains "{}". Defaults to "{}".
        ext (str, optional): Image format of frames. Defaults to "png".

    Returns:
        int: 1 if success, 0 otherwise.
    """
    if "{}" not in naming:
        print(f"{naming} must contain {{}} for naming format. Using Default naming.")
        naming = "{}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    nb_frames = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            file_idx: str = str(n).zfill(nb_frames)
            file_name = out_dir / (naming.format(file_idx) + "." + ext)
            cv2.imwrite(str(file_name), frame)
            n += 1
        else:
            return 1


###################
# Data Collection #
###################


def collect_images(
    keywords: List[str],
    max_num_images: int = 10,
    out_dir: str = "collected_images",
    num_threads: int = 4,
    search_engine: Literal["google", "bing", "baidu"] = "bing",
) -> None:
    """
    Collect images from Bing image search.

    Args:
        keywords (List[str]): List of keywords to search.
        max_num_images (int, optional): Maximum number of images to download.
            Defaults to 10.
        out_dir (str, optional): Output directory. Defaults to "collected_images".
        num_threads (int, optional): Number of threads to use. Defaults to 4.
        search_engine (Literal["google", "bing", "baidu"], optional): Search engine to use.

    Examples:
        >>> collect_images(["cat", "dog"], max_num_images=10, out_dir="collected_images")

        >>> collect_images(["cat", "dog"], max_num_images=10, out_dir="collected_images", search_engine="google")
    """
    try:
        from icrawler.builtin import (
            BingImageCrawler,
            GoogleImageCrawler,
            BaiduImageCrawler,
        )
    except ImportError:
        raise ImportError(
            "Please install icrawler to use the function 'collect_images()'"
        )
    engines = {
        "google": GoogleImageCrawler,
        "bing": BingImageCrawler,
        "baidu": BaiduImageCrawler,
    }
    crawler = engines[search_engine](
        downloader_threads=num_threads, storage={"root_dir": out_dir}
    )
    kwargs = " ".join(keywords)
    crawler.crawl(keyword=kwargs, max_num=max_num_images)
