from typing import Tuple, Union
import torch
import numpy as np

from math import pi

from torch import Tensor

__all__ = ["GMM"]


class GMM(torch.nn.Module):
    """PyTorch implementation of Gauusian Mixture Model with pytorch lightning support.

    Attributes:
        var (Tensor): Variance of the Gaussian distribution.
        mu (Tensor): Mean of the Gaussian distribution.
        pi (Tensor): Weight of the Gaussian distribution.
        covariance_type (str): Type of covariance, one of ["diag", "full"].
        eps (float): Precision.
        init_params (str): Method to init params, one of ["random","kmeans"].
        log_likelihood (float): Log-likelihood of the data.
        n_components (int): Number of mixture components.
        n_features (int): Number of features per sample.
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        covariance_type: str = "full",
        eps: float = 1.0e-6,
        init_params: str = "kmeans",
        mu_init: Tensor = None,
        var_init: Tensor = None,
    ) -> None:
        """Initialize GMM  

        Args:
            n_components (int): Number of mixture components.
            n_features (int): Number of features per sample.
            covariance_type (str, optional): Covariance type, one of ["diag", "full"]. Defaults to "full".
            eps (float, optional): Precision. Defaults to 1.0e-6.
            init_params (str, optional): Method to init params, one of ["random","kmeans"]. Defaults to "kmeans".
            mu_init (Tensor, optional): Initial mean. Defaults to None.
            var_init (Tensor, optional): Initial variance. Defaults to None.
        """
        super().__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self.__init_params()

    def __init_params(self):
        """Initializes the model parameters using the given initialization method.
        """
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), (
                "Input mu_init does not have required tensor dimensions (1, %i, %i)"
                % (self.n_components, self.n_features)
            )
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)  # type: ignore
        else:
            self.mu = torch.nn.Parameter(  # type: ignore
                torch.randn(1, self.n_components, self.n_features), requires_grad=False
            )

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (
                    1,
                    self.n_components,
                    self.n_features,
                ), (
                    "Input var_init does not have required tensor dimensions (1, %i, %i)"
                    % (self.n_components, self.n_features)
                )
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)  # type: ignore
            else:
                self.var = torch.nn.Parameter(  # type: ignore
                    torch.ones(1, self.n_components, self.n_features),
                    requires_grad=False,
                )
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (
                    1,
                    self.n_components,
                    self.n_features,
                    self.n_features,
                ), (
                    "Input var_init does not have required tensor dimensions (1, %i, %i, %i)"
                    % (self.n_components, self.n_features, self.n_features)
                )
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False,)  # type: ignore
            else:
                self.var = torch.nn.Parameter(  # type: ignore
                    torch.eye(self.n_features, dtype=torch.float64)
                    .reshape(1, 1, self.n_features, self.n_features)
                    .repeat(1, self.n_components, 1, 1),
                    requires_grad=False,
                )

        # (1, k, 1)
        self.pi = torch.nn.Parameter(  # type: ignore
            torch.Tensor(1, self.n_components, 1), requires_grad=False
        ).fill_(1.0 / self.n_components)

        self.params_fitted = False

    def __check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def bic(self, x: Tensor) -> float:
        """Bayesian information criterion for a batch of samples.

        Args:
            x (Tensor): Samples of shape (n, d) or (n, 1, d).
        
        Returns:
            float: BIC score.
        """
        x = self.__check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = (
            self.n_features * self.n_components
            + self.n_features
            + self.n_components
            - 1
        )

        bic = -2.0 * self.__score(x, sum_data=False).mean() * n + free_params * np.log(
            n
        )

        return bic

    def fit(
        self, x: Tensor, delta: float = 1e-3, n_iter: int = 100, warm_start=False
    ) -> None:
        """Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).

        Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
        The model then extends them to (n, 1, d).
        The model parametrization (mu, sigma) is stored as (1, k, d),
        probabilities are shaped (n, k, 1) if they relate to an individual sample,
        or (1, k, 1) if they assign membership probabilities to one of the mixture components.

        Args:
            x (Tensor): A tensor of shape (n, d) or (n, 1, d).
            delta (float, optional): Delta param for EM algorithm. Defaults to 1e-3.
            n_iter (int, optional): Number of iteration to fit. Defaults to 100.
            warm_start (bool, optional): True to prevent initializing parameters. Defaults to False.
        """
        if not warm_start and self.params_fitted:
            self.__init_params()

        x = self.__check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(
                self.log_likelihood
            ):
                device = self.mu.device
                # When the log-likelihood assumes inane values, reinitialize model
                self.__init__(
                    self.n_components,
                    self.n_features,
                    covariance_type=self.covariance_type,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps,
                )
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    (self.mu.data,) = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)  # type: ignore
                self.__update_var(var_old)  # type: ignore

        self.params_fitted = True

    def forward(self, x, probs=False) -> Tensor:
        """Predict assignment of x to mixture components by calculating the
        likelihood that each component is responsible for each point in x.
        If prob is True, returns normalized probability of each class for each point.

        Args:
            x (Tensor): A tensor of shape (n, d) or (n, 1, d).
            probs (bool, optional): True to get normalized probabilities for each class. 
                                    Defaults to False.

        Returns:
            Tensor: A tensor of shape (n) if probs is False, else (n, k) 
        """
        x = self.__check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(
                torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor)  # type: ignore
            )

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Returns normalized probabilities of class membership.
        Args:
            x (Tensor): A tensor of shape (n, d) or (n, 1, d).
        Returns:
            Tensor: A tensor of shape (n)
        """
        return self(x, probs=True)

    def score_samples(self, x: Tensor) -> Tensor:
        """Return log-likelihood of data under the model with current params.
        Args:
            x (Tensor): A tensor of shape (n, d) or (n, 1, d).
        Returns:
            Tensor: A tensor of shape (n).
        """
        x = self.__check_size(x)
        score = self.__score(x, sum_data=False)

        return score

    def _estimate_log_prob(self, x: Tensor) -> Tensor:
        """
        Return (n, k, 1) tensor of the log-likelihood for each k-th component.
        Args:
            x (Tensor): A tensor of shape (n, d) or (n, 1, d).
        Returns:
            Tensor: A tensor of shape (n, k, 1)
        """
        x = self.__check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var
            precision = torch.inverse(var)
            d = x.shape[-1]

            log_2pi = d * np.log(2.0 * pi)

            log_det = self._calculate_log_det(precision)

            x = x.double()
            mu = mu.double()
            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)

            x_mu_T_precision = self.__calculate_matmul_n_times(
                self.n_components, x_mu_T, precision
            )
            x_mu_T_precision_x_mu = self.__calculate_matmul(x_mu_T_precision, x_mu)

            return -0.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum(
                (mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True
            )
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -0.5 * (self.n_features * np.log(2.0 * pi) + log_p) + log_det

        else:
            raise ValueError("Invalid covariance type.")

    def _calculate_log_det(self, var: Tensor) -> Tensor:
        """Calculate the log determinant of a matrix to prevent overflow.

        Args:
            var (Tensor): A tensor of shape (1, k, d, d).

        Returns:
            Tensor: A tensor of shape (1, k, 1).
        """
        log_det = torch.empty(size=(self.n_components,)).to(var.device)

        for k in range(self.n_components):
            evals, evecs = torch.linalg.eig(var[0, k])  # type: ignore
            log_det[k] = torch.log(evals.real).sum()

        return log_det.unsqueeze(-1)

    def _e_step(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """E-step of the EM algorithm.

        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.

        Args:
            x (Tensor): A tensor of shape (n, d) or (n, 1, d).

        Returns:
            log_prob_norm (Tensor):  torch.Tensor (1)
            log_resp (Tensor):       torch.Tensor (n, k, 1)
        """
        x = self.__check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x: Tensor, log_resp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """M-step of the EM algorithm.

        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        
        Args:
            x (Tensor): A tensor of shape (n, d) or (n, 1, d).
            log_resp (Tensor): A tensor of shape (n, k, 1).
        
        Returns:
            pi (Tensor): A tensor of shape (1, k, 1).
            mu (Tensor): A tensor of shape (1, k, d).
            var (Tensor): A tensor of shape (1, k, d, d).
        """
        x = self.__check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = (
                torch.sum(
                    (x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2))
                    * resp.unsqueeze(-1),
                    dim=0,
                    keepdim=True,
                )
                / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1)
                + eps
            )
        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps
        else:
            raise ValueError(f"Unknown covariance type: {self.covariance_type}")

        pi = pi / x.shape[0]

        return pi, mu, var

    def __em(self, x: Tensor) -> None:
        """Single iteration of the EM algorithm.

        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        
        Args:
            x (Tensor): A tensor of shape (n, 1, d).
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x: Tensor, sum_data: bool = True) -> Tensor:
        """Computes the log-likelihood of the data under the model.

        Return summed log-likelihood if sum_data is True, otherwise 
        returns the log-likelihood for each data point.

        Args:
            x (Tensor): A tensor of shape (n, 1, d).
            sum_data (bool): If True, sum the log-likelihood over the data points.

        Returns:
            Tensor: Summed log_likelihood if sum_data is True, otherwise log_likelihood for each data point.
        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if sum_data:
            return per_sample_score.sum()
        else:
            return torch.squeeze(per_sample_score)

    def __update_mu(self, mu: Union[Tensor, torch.FloatTensor]) -> None:
        """Updates mean to the provided value.

        Args:
            mu (torch.FloatTensor): A tensor of shape (1, k, d).
        """
        assert mu.size() in [
            (self.n_components, self.n_features),
            (1, self.n_components, self.n_features),
        ], (
            "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
            % (self.n_components, self.n_features, self.n_components, self.n_features)
        )

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var: Union[Tensor, torch.FloatTensor]) -> None:
        """Updates variance to the provided value.
        args:
            var (torch.FloatTensor): A tensor of shape (1, k, d, d).
        """
        if self.covariance_type == "full":
            assert var.size() in [
                (self.n_components, self.n_features, self.n_features),
                (1, self.n_components, self.n_features, self.n_features),
            ], (
                "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)"
                % (
                    self.n_components,
                    self.n_features,
                    self.n_features,
                    self.n_components,
                    self.n_features,
                    self.n_features,
                )
            )

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [
                (self.n_components, self.n_features),
                (1, self.n_components, self.n_features),
            ], (
                "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
                % (
                    self.n_components,
                    self.n_features,
                    self.n_components,
                    self.n_features,
                )
            )

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi: Union[Tensor, torch.FloatTensor]) -> None:
        """Updates pi to the provided value.
        
        Args:
            pi (torch.FloatTensor): A tensor of shape (1, k).
        """
        assert pi.size() in [(1, self.n_components, 1)], (
            "Input pi does not have required tensor dimensions (%i, %i, %i)"
            % (1, self.n_components, 1)
        )

        self.pi.data = pi

    def get_kmeans_mu(
        self, x, n_centers, init_times: int = 50, min_delta: float = 1e-3
    ) -> Tensor:
        """Find an initial value for the mean. 
        
        Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.

        Args:
            x (Tensor): A tensor of shape of (n, d) or (n, 1, d).
            init_times (int): Number of times to run the k-means algorithm.
            min_delta (float): Minimum change in the loss function to stop the k-means algorithm.
        
        Returns:
            Tensor: A tensor of shape (1, k, d).
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf

        center = x[
            np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...,
        ]

        for i in range(init_times):
            tmp_center = x[
                np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False),
                ...,
            ]
            l2_dis = torch.norm(
                (x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2
            )
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm(
                (x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2
            )
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return center.unsqueeze(0) * (x_max - x_min) + x_min

    def __calculate_matmul_n_times(
        self, n_components: int, mat_a: Tensor, mat_b: Tensor
    ) -> Tensor:
        """Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].

        Bypasses torch.matmul to reduce memory footprint.

        Args:
            mat_a (Tensor): A tensor of shape (n, k, 1, d).
            mat_b (Tensor): A tensor of shape (1, k, d, d).

        Returns:
            Tensor: A tensor of shape (n, k, 1, d).
        """
        res = torch.zeros(mat_a.shape).double().to(mat_a.device)

        for i in range(n_components):
            mat_a_i = mat_a[:, i, :, :].squeeze(-2)
            mat_b_i = mat_b[0, i, :, :].squeeze()
            res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

        return res

    def __calculate_matmul(self, mat_a, mat_b):
        """Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].

        Bypasses torch.matmul to reduce memory footprint.

        Args:
            mat_a (Tensor): A tensor of shape (n, k, 1, d).
            mat_b (Tensor): A tensor of shape (n, k, d, 1).

        Returns:
            Tensor: A tensor of shape (n, k, 1).
        """
        assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
        return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)
