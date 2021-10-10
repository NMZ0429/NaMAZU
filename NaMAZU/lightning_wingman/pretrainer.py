import os
import random
from glob import glob
from typing import Dict, List
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

try:
    from byol_pytorch import BYOL
except ImportError:
    print("Semi-Supervised Learning reuires package 'byol-pytorch'!!")
    raise ImportError(
        "BYOL not installed. Please install it with [pip install byol-pytorch]"
    )

__all__ = ["self_supervised_training"]


def __choose_model(model: str) -> torch.nn.Module:
    if model == "vgg16":
        return models.vgg16(pretrained=True)
    elif model == "vgg19":
        return models.vgg19(pretrained=True)
    elif model == "resnet50":
        return models.resnet50(pretrained=True)
    elif model == "resnet101":
        return models.resnet101(pretrained=True)
    elif model == "resnet152":
        return models.resnet152(pretrained=True)
    elif model == "densenet121":
        return models.densenet121(pretrained=True)
    elif model == "densenet161":
        return models.densenet161(pretrained=True)
    elif model == "densenet169":
        return models.densenet169(pretrained=True)
    elif model == "densenet201":
        return models.densenet201(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model}")


def __set_number_of_threads(num_threads: int) -> None:
    print("Setting number of threads to {}".format(num_threads))
    torch.set_num_threads(num_threads)


def __prepare_image_pathes(image_dirs: List[str]) -> Dict[str, List[str]]:
    """Return the dict of list of image paths for each directory in image_dirs"""
    image_paths = {}
    for img_dir in image_dirs:
        if not os.path.isdir(img_dir):
            raise ValueError("Image directory {} does not exist".format(img_dir))
        imgs = (
            glob(os.path.join(img_dir, "*.png"))
            + glob(os.path.join(img_dir, "*.jpg"))
            + glob(os.path.join(img_dir, "*.jpeg"))
        )
        image_paths[img_dir] = imgs

    return image_paths


def sample_unlabelled_images(list_imgs: List[str], n: int) -> torch.Tensor:
    chosen = random.sample(list_imgs, n)
    batch = []
    for img_path in chosen:
        img = Image.open(img_path).convert("RGB")
        # img = img.resize((256, 256))
        img = transforms.ToTensor()(img)
        batch.append(img)

    batch = torch.stack(batch)
    return batch


def __stratified_sampling(
    datasets: List[List[str]], n: int, img_size: int
) -> torch.Tensor:
    """Sample n images from each dataset and return a tensor
    
    Args:
        datasets[List[List[str]]]: List of lists of image paths
        n[int]: Number of images to sample
    
    Returns:
        torch.Tensor: Tensor of shape (n*len(datasets), 3, 224, 224)
    """
    batch = []
    for dataset in datasets:
        imgs = random.sample(dataset, n)
        for img_path in imgs:
            img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
            img = transforms.ToTensor()(img)
            batch.append(img)

    return torch.stack(batch, dim=0)


def print_training_summary(dataset_dict: Dict[str, List[str]], model) -> None:
    print("Training summary:")
    print(f"\tModel: {model}")
    print("\tNumber of images:")
    for key, value in dataset_dict.items():
        print("\t\t{}: {}".format(key, len(value)))


def self_supervised_training(
    model_choice: str,
    image_dirs: List[str],
    batch_size: int,
    num_threads: int = 6,
    num_iterations: int = 10000,
    save_dir: str = "",
    img_size: int = 256,
    device: str = "cuda",
    simsiam: bool = False,
) -> torch.nn.Module:  # type: ignore
    """Run self supervised training on given model with images given by image_dirs.

    If len(image_dirs) is 1, then the model is trained on the single dataset. 
    Otherwise, mini-batch consists of images drown n // len(dataset) times from each dataset.
    Argument simsiam is used to indicate whether to do SimSiam training or not.

    Args:
        model_choice (str): Model to train currently VGG, ResNet and DenseNet are supported.
        image_dirs (List[str]): List of pathes of datasets.
        batch_size (int): Batch size.
        num_threads (int): Number of cpu threads to use.
        num_iterations (int): Number of epochs. Default is 10000.
        save_dir (str, optional): Trained model is saved to the directory if given otherwise returned. Defaults to "".
        img_size (int, optional): Image size. Defaults to 256.
        device (str, optional): Device to use. Defaults to "cuda".
        simsiam (bool, optional): If True, use SimSiam. Defaults to False.

    Returns:
        torch.nn.Module: Trained model.
    """
    __set_number_of_threads(num_threads)

    # Set up dataset
    if batch_size % len(image_dirs) != 0:
        # raise ValueError("Batch size must be divisible by number of datasets")
        batch_size -= batch_size % len(image_dirs)
    n_per_subset = batch_size // len(image_dirs)

    image_path_dict = __prepare_image_pathes(image_dirs)
    datasets = list(image_path_dict.values())

    # Set up required things
    model = __choose_model(model_choice)
    model.to(torch.device(device))
    learner = BYOL(
        net=model,
        image_size=img_size,
        hidden_layer="avgpool",
        use_momentum=(not simsiam),
    )
    learner.to(torch.device(device))
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    # Print summary
    print_training_summary(image_path_dict, model_choice)

    # Run main loop
    for _ in tqdm(range(num_iterations)):
        images = __stratified_sampling(
            datasets=datasets, n=n_per_subset, img_size=img_size
        )
        images = images.to(torch.device(device))
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if not simsiam:
            learner.update_moving_average()  # update moving average of target encoder

    # Save model
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {save_dir}")
        torch.save(
            learner.state_dict(), os.path.join(save_dir, f"trained_{model_choice}.pth")
        )

    return model


# ---------------------------------------------------------------------------------
