from typing import Union
import torch
from torch import Tensor

__all__ = ["KNN"]


class KNN(torch.nn.Module):
    def __init__(
        self,
        n_neighbors: int,
        training_data: Tensor = torch.tensor([]),
        distance_measure: str = "euclidean",
        training_labels: Tensor = None,
    ):
        """KNN implementation using PyTorch.

        Args:
            k (int): N umber of nearest neighbors to return.
            training_data (Tensor, optional): N by M tensor of training data. Defaults to None.
            ditance_measure (str, optional): The choice of distance to use. Defaults to "euclidean".
            training_labels (Tensor, optional): N by 1 tensor of training labels. Defaults to None.

        Raises:
            ValueError: Raise if the chosen distance measure is not supported.
        """
        super().__init__()
        self.k = n_neighbors
        self.training_data = training_data
        self.training_labels = training_labels
        self.num_classes: int = 0
        self.distance_measure = distance_measure

        if self.training_labels is not None:
            self.num_classes = int(self.training_labels.max().item() + 1)

        self.__choose_distance_measure()

    def fit(self, training_data: Tensor) -> None:
        """Train KNN.

        Args:
            training_data (Tensor): Training data.
        """
        self.training_data = training_data

    def forward(self, x: Tensor) -> Tensor:
        """Return the indices of the k nearest neighbors of x.

        Args:
            x (Tensor): input point.

        Returns:
            Tensor: N by K tensor of k indices of the k nearest neighbors of x.
        Raises:
            ValueError: Raise if the model has not been fit.
        """
        if self.training_data is None:
            raise ValueError(
                "Model hasn't been fitted yet. Use self.fit(training_data)."
            )
        x = self._validate_input(x)
        distances = self.calc_distasnce(x)
        _, indices = distances.topk(self.k, dim=1, largest=False, sorted=True)
        return indices

    def get_k_nearest_neighbors(self, x: Tensor) -> Tensor:
        """Return the k nearest neighbors of x.

        Args:
            x (Tensor): input point.

        Returns:
            Tensor: 1d tensor of k nearest neighbors of x.
        """
        indices = self.forward(x)
        return self.training_data[indices]

    def get_k_nearest_labels(self, x: Tensor) -> Tensor:
        """Return the k nearest labels of x.

        Args:
            x (Tensor): input point.

        Returns:
            Tensor: 1d tensor of k nearest labels of x.
        """
        if self.training_labels is None:
            raise ValueError("No training labels provided")
        indices = self.forward(x)
        return self.training_labels[indices]

    ####################
    # Helper functions #
    ####################

    def _calculate_euclidean(self, x: Tensor) -> Tensor:
        return torch.cdist(x, self.training_data, 2)

    def _calculate_manhattan(self, x: Tensor) -> Tensor:
        return torch.cdist(x, self.training_data, 1)

    def _calculate_cosine(self, x: Tensor) -> Tensor:
        return self._sim_matrix(x, self.training_data)

    def _validate_input(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.training_data.shape[1]:
            raise ValueError(
                f"Input dimension must match training data. Got {x.shape[1]} but expected {self.training_data.shape[1]}"
            )

        return x

    def _cov(self, x, rowvar=False, bias=False, ddof=None, aweights=None):
        """Estimates covariance matrix like numpy.cov"""
        # ensure at least 2D
        if x.dim() == 1:
            x = x.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and x.shape[0] != 1:
            x = x.t()

        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0

        w = aweights
        if w is not None:
            if not torch.is_tensor(w):
                w = torch.tensor(w, dtype=torch.float)
            w_sum = torch.sum(w)
            avg = torch.sum(x * (w / w_sum)[:, None], 0)
        else:
            avg = torch.mean(x, 0)

        # Determine the normalization
        if w is None:
            fact = x.shape[0] - ddof
        elif ddof == 0:
            fact = w_sum  # type: ignore
        elif aweights is None:
            fact = w_sum - ddof  # type: ignore
        else:
            fact = w_sum - ddof * torch.sum(w * w) / w_sum  # type: ignore

        xm = x.sub(avg.expand_as(x))

        if w is None:
            X_T = xm.t()
        else:
            X_T = torch.mm(torch.diag(w), xm).t()

        c = torch.mm(X_T, xm)
        c = c / fact

        return c.squeeze()

    def _setup_mahalanobis(self):
        """Calculate class wise covariance matrix"""
        if not self.training_labels:
            raise ValueError("Mahalanovis distance requires labeled training data.")

        cov_mat_list = []
        for i in range(self.num_classes):
            cluster = self.training_data[self.training_labels == i]
            cov_mat_list.append(self._cov(cluster).unsqueeze(0))
        self.cluster_cov = torch.cat(cov_mat_list, dim=0)

    def _calculate_mahalanobis(
        self, x: Tensor, v: Tensor = None, cov: Tensor = None
    ) -> Tensor:
        """Calulate the mahalanobis distance between u and v.

        Args:
            x (Tensor): First input vector.
            v (Tensor): Second input vector. Defaults to use training data.
            cov (Tensor): Covariane matrix. Defaults to use training data.

        Returns:
            Tensor: Distance between u and v.
        """
        v = self.training_data if not v else v
        cov = self.cluster_cov if not cov else cov

        delta = x - v
        m = torch.dot(delta, torch.matmul(torch.inverse(cov), delta))
        return torch.sqrt(m)

    def _sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        b = self.training_data if b is None else b
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def __choose_distance_measure(self) -> None:
        if self.distance_measure == "euclidean":
            self.calc_distasnce = self._calculate_euclidean
        elif self.distance_measure == "manhattan":
            self.calc_distasnce = self._calculate_manhattan
        elif self.distance_measure == "cosine":
            self.calc_distasnce = self._calculate_cosine
        elif self.distance_measure == "mahalanobis":
            self._setup_mahalanobis()
            self.calc_distasnce = self._calculate_mahalanobis
        else:
            raise ValueError(f"{self.distance_measure} is not a valid distance measure")
