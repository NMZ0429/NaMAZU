from torch import Tensor
import torch
from typing import List, Union
from pytorch_lightning import LightningModule


class MultiModalNet(LightningModule):
    def __init__(
        self,
        modality_dimensions,
        latent_dim,
        direct_docking,
        num_classes,
        is_regression: bool = False,
        modality_preprocessors=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_modalities = len(modality_dimensions)

        self.__prepare_model()

        self.clf = torch.nn.Sequential(
            torch.nn.ReLU, torch.nn.Linear(latent_dim, num_classes)
        )

        if is_regression:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        x, t = batch

    def _varify_batch(self, batch):
        if len(batch) != self.num_modalities:
            raise ValueError(
                f"The number of modalities in the batch must be {self.num_modalities} but {len(batch)} was given."
            )

    def _by_modal_forward(self, modalities: List[Tensor]) -> Tensor:
        output_list = []
        if self.bypass_docking:
            output_list = modalities
        else:
            for i, modality in enumerate(modalities):
                x = getattr(self, f"stream_{i}")(modality)
                x = torch.nn.functional.relu(x)  # type: ignore
                output_list.append(x)

        return torch.stack(
            output_list, dim=-1
        )  # [batch_size, latent_dim, num_modalities]

    def _check_lacking_modality(
        self, modalities: Tensor, modal_available: Union[Tensor, None]
    ) -> Tensor:
        if modal_available is None:
            return torch.ones(
                modalities.shape[0],
                self.num_modalities,
                dtype=torch.float,
                device=self.device,
            )
        else:
            return modal_available

    def _check_prior(self, prior_modality_distribution: Union[Tensor, None]) -> Tensor:
        if prior_modality_distribution is None:
            return torch.ones(
                self.num_modalities, dtype=torch.float, device=self.device
            )
        else:
            return prior_modality_distribution

    def _mulnom_sampling(
        self,
        laten_modalities: Tensor,
        prior_modality_distribution: Tensor,
        modal_available: Tensor,
    ):
        weighted_prior = torch.mul(prior_modality_distribution, modal_available)
        weighted_prior = torch.div(prior_modality_distribution, weighted_prior)

        modality_idx = torch.multinomial(
            weighted_prior,
            num_samples=self.hparams.latend_dim,  # type: ignore
            replacement=True,
        )
        chosen_modalities = torch.nn.functional.one_hot(  # type: ignore
            modality_idx, num_classes=self.num_modalities
        ).float()

        return torch.sum(torch.mul(laten_modalities, chosen_modalities), dim=1)

    def __prepare_model(self):
        if not self.hparams.direct_docking:  # type: ignore
            for i, modal_dim in enumerate(self.hparams.modality_dimensions):  # type: ignore
                setattr(
                    self,
                    f"stream_{i}",
                    torch.nn.Linear(modal_dim, self.hparams.latent_dim),  # type: ignore
                )
        else:
            for i in self.hparams.modality_dimensions:  # type: ignore
                if i != self.hparams.latend_dim:  # type: ignore
                    raise ValueError(
                        f"The input data must be the shape of (batch_size, {self.hparams.latent_dim}) but input_size_list containes {i}."  # type: ignore
                    )
