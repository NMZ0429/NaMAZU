from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


__all__ = ["Result", "Query"]


@dataclass
class Query:
    __query: Dict[str, Any]

    def __str__(self) -> str:
        doc = f"Query: \n"
        for key in self.__query.keys():
            doc += f"      {key}, type {type(self.__query[key])}\n"

        return doc

    def __repr__(self) -> str:
        return f"Query(query={self.__query})"

    def __getitem__(self, key: str):
        return self.__query[key]


from dataclasses import dataclass
from typing import List

import torch
from albumentations import Compose
from torch import Tensor
from torchvision.io import read_video


@dataclass
class Video:
    """Dataclass to store input video as Tensor. Also works as a DataLoader for a video.

    Attributes:
    -----------
    path: str
        Path to the video.
    num_frames: int
        Number of frames that compose a sample. Equivalent to the length of LSTM.
    step_size: int
        Number of frames to skip between frames.
    batch_size: int
        Number of samples to include in a mini-batch.
    video_name: str
        Name of the video
    raw_frames: Tensor
        Tensor of shape (T, H, W, C) without any preprocessing.
    frames: Tensor
        Tensor of shape (T, C, H, W) after preprocessing.
    offset: int
        Number of frames not to be infered at the beginning.
    list_mini_batches: List[List[List[int]]]
        3D list of frame indices. first dimension containes mini-batch and second dimension contains each sample.
    """

    path: str
    num_frames: int
    step_size: int
    preprocess: Compose
    batch_size: int = 5

    def __post_init__(self):
        self.video_name = self.path.split("/")[-1].replace(".mp4", "")
        # convert mp4 into Tensor of shape (T, C, H, W)
        self.raw_frames = read_video(self.path, pts_unit="sec")[0].detach().numpy()
        """self.frames = torch.stack(
            [self.preprocess(image=frames[i])["image"] for i in range(len(frames))]
        )"""
        self._prepare_minibatch()

    def __len__(self) -> int:
        """Return the number of mini batches in the video."""
        return len(self.list_mini_batches)

    def _get_sample(self, indices: List[int]) -> Tensor:
        """Return a Tensor of shape (T, C, H, W)

        Args:
            idx: List of indices of length T

        Returns:
            Tensor: Tensor of frames stacked at Time channel
        """
        rtn = []
        for i in indices:
            item = self.raw_frames[i]
            item = self.preprocess(image=item)["image"]
            rtn.append(item)
        return torch.stack(rtn)

    def __getitem__(self, idx: int) -> Tensor:
        """Return a minibatch of the sequence of frames

        Args:
            idx: index of the batch

        Returns:
            Tensor: Tensor (N, T, C, H, W) of frames stacked at channel N
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        mini_batch = []
        for i in self.list_mini_batches[idx]:
            mini_batch.append(self._get_sample(i))

        return torch.stack(mini_batch)

    def _prepare_minibatch(self) -> None:
        # make a nested list that containes a list of frame indices to feed for each forward path for LSTM
        # e.g. [[0, 3, 6], [1, 4, 7], [2, 4, 8] ...] means  3 frames with step_size 3 composes single sample.
        # if num_frames = 5, step_size = 3, total = 16
        total = (self.num_frames - 1) * (self.step_size + 1) + 1
        self.offset = total
        targets = list(range(0, total, self.step_size + 1))
        frame_seq = []
        while targets[-1] < len(self.raw_frames):
            frame_seq.append(targets)
            targets = [x + 1 for x in targets]

        # Split the list of samples into the list of mini batches
        self.list_mini_batches = [
            frame_seq[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range((len(frame_seq) + self.batch_size - 1) // self.batch_size)
        ]


@dataclass
class Result:
    """Dataclass to store single result of the metric trainer."""

    embedding: Tensor
    distance: Tensor

    def __post_init__(self):
        if self.embedding.shape[0] != self.distance.shape[0]:
            raise ValueError("Embedding and distance must have same number of samples")

        self.other_data: dict = {}

    def __len__(self) -> int:
        """Return the length of inference."""
        return len(self.embedding)

    def __getitem__(self, idx: str) -> Tensor:
        """Return embedding or distannce of the inference."""
        if idx == "embedding":
            return self.embedding
        elif idx == "distance":
            return self.distance
        if idx in self.other_data:
            return self.other_data[idx]
        else:
            raise IndexError("Unknown Index.")

    def __setitem__(self, idx: str, value) -> None:
        """Add any data into the container."""
        if idx == "embedding":
            self.embedding = value
        elif idx == "distance":
            self.distance = value
        else:
            self.other_data[idx] = value
