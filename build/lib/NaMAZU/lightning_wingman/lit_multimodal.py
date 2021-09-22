from torch import Tensor
import torch
from typing import List, Union, Dict
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule


class MultiModalNet(LightningModule):
    def __init__(
        self,
        modality_dimensions: List[int],
        modality_encoders: List[torch.nn.Module],
        latent_dim: int = 512,
        direct_fusion: bool = False,
        num_classes: int = 1,
        is_regression: bool = False,
        train_encoders: bool = False,
        use_default_modality_dropout: bool = True,
        md_prob: float = 0.5,
        **kwargs,
    ) -> None:
        """Multipurpose multimodal network.

        This model operates stochastic modality sampling to increase robustness of fusion vector.
        Is consists of three parts:
            1. Encoder:
                List of encoder modules for each modality. The choice of encoder is determined by the user depending on modalities.
            2. Fusion layers:
                Main part of the model.They are the list of linear layersa and one sampling function.
                They takes the list of vectors from encoders and fuse them stochastically.
                There are two adjustable parameters for the fusion layers:
                    1. modal_available:
                        This is a tensor of shape (batch_size, num_modalities) which indicates whether the modality is available or not. 
                        Use built-in modality droput function to utilize this and make the model more robust.
                    2. prior_modality_distribution:
                        This is a tensor of of shape [batch_size, num_modalities], which indicates the prior distribution of feature importance of each modality.
                        Changing this dynamically can improve the performance of the model and there is a built-in adjustment algorithm for this. But you can 
                        implment your own (書いたらPRしてね。)
                        
            3. Output layer:
                Output layer. It is a linear layer to convert the fusion vector to the desired output.

        The written network is only for fusion operation and output layers so by passing modality encoders, 
        this model can process any type of modality and output strandarized vector for eithre classification or regression.
        Make you that you pass correct modality dimension and encoder.

        For example, pass two CNNs if the input modalities are RGB image and audio spectrogram.
        If you are dealing with audio and video, you can pass CNNLSTM and LSTM respectively.

        Args:
            modality_dimensions (List[int]): List of output dimensions for each modality encoder.
            modality_encoders (List[torch.nn.Module], optional): List of modality encoders.
            latent_dim (int, optional): Dimension to which adjust output from each encoder. Output of fusion layer is also this. Defaults to 512.
            direct_fusion (bool, optional): True to skip the adjustment FC layer but doing so requires all 
                                             encoders output the same shape of vector. Defaults to False.
            num_classes (int, optional): Number of classes for classification task. Defaults to 1.
            is_regression (bool, optional): True to train it with MSE. Don't forget to set num_classes to 1. Defaults to False.
            train_encoders (bool, optional): True to train encoders too. Defaults to False.
            use_default_modality_dropout (bool, optional): True to use default modality dropout. Defaults to True.
            md_prob (float, optional): Probability of modality dropout. Defaults to 0.5.
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_modalities = len(modality_dimensions)

        self.__prepare_model()

        self.output_layer = torch.nn.Sequential(
            torch.nn.ReLU, torch.nn.Linear(latent_dim, num_classes)
        )

        if is_regression:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        if not train_encoders:
            self.encoders.eval()

    def forward(self, x_list: List[Tensor], is_train: bool = False) -> Tensor:
        self._varify_batch(x_list)

        if is_train:
            ma = self.modality_dropout()
            prior = self.stochstic_fusion()
        else:
            ma, prior = None, None
        y_list = [encoder(x) for encoder, x in zip(self.encoders, x_list)]
        y_stack = self._by_modal_forward(y_list)
        availability = self._check_lacking_modality(y_stack, modal_available=ma)
        prior_modality_distribution = self._check_prior(
            prior_modality_distribution=prior
        )
        fusion_vector = self._mulnom_sampling(
            y_stack, prior_modality_distribution, availability
        )

        return self.output_layer(fusion_vector)

    def configure_optimizers(self,) -> Dict[str, Union[Adam, StepLR]]:
        params = []
        for i in range(self.num_modalities):
            params += list(getattr(self, f"fc_{i}").parameters())
        if self.hparams.train_encoders:  # type: ignore
            params += list(self.encoders.parameters())
        optimizer = Adam(params, lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": StepLR(optimizer, step_size=2, gamma=0.8),
        }

    def _varify_batch(self, batch: List[Tensor]):
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
                x = getattr(self, f"fc_{i}")(modality)
                x = torch.nn.functional.relu(x)  # type: ignore
                output_list.append(x)

        return torch.stack(
            output_list, dim=-1
        )  # [batch_size, latent_dim, num_modalities]

    def _check_lacking_modality(
        self, modalities: Tensor, modal_available: Union[Tensor, None] = None
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

    def _check_prior(
        self, prior_modality_distribution: Union[Tensor, None] = None
    ) -> Tensor:
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

    # Setup methods

    def __prepare_model(self):
        if not self.hparams.direct_fusion:  # type: ignore
            for i, modal_dim in enumerate(self.hparams.modality_dimensions):  # type: ignore
                setattr(
                    self,
                    f"fc_{i}",
                    torch.nn.Linear(modal_dim, self.hparams.latent_dim),  # type: ignore
                )
        else:
            for i in self.hparams.modality_dimensions:  # type: ignore
                if i != self.hparams.latend_dim:  # type: ignore
                    raise ValueError(
                        f"The input data must be the shape of (batch_size, {self.hparams.latent_dim}) but input_size_list containes {i}."  # type: ignore
                    )

        self.encoders = torch.nn.ModuleList(self.hparams.modality_encoders)  # type: ignore

    # TODO: Add stochastic dropout
    # TODO: update forward method to support above changes

    def modality_dropout(self):
        drop = torch.rand(1).item()
        if drop > self.hparams.md_prob:  # type: ignore
            return torch.zeros(
                self.num_modalities, dtype=torch.float, device=self.device
            )

    def stochstic_fusion(self):
        pass
