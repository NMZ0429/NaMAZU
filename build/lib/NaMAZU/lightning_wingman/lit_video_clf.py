from typing import Dict, List, Tuple, Union

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .video_clf import CNNLSTM, CNNClassifier

__all__ = ["LitVideoClf"]


class LitVideoClf(LightningModule):
    def __init__(self, use_lstm: bool, model_config: dict, *args, **kwargs) -> None:
        """Lighting module for video classification.

        Args:
            use_lstm (bool): whether to use CNNLSTM or single frame CNN
            model_config (dict): Dictionary with model configuration for CNNLSTM or CNNClassifier
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.__check_model_config(model_config)

        if use_lstm:
            self.model = CNNLSTM(**model_config)
        else:
            self.model = CNNClassifier(**model_config)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy()

    def configure_optimizers(self,) -> Dict[str, Union[Adam, StepLR]]:
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": StepLR(optimizer, step_size=2, gamma=0.8),
        }

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.acc(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb) -> torch.Tensor:
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb) -> torch.Tensor:
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)
        return loss

    def test_step(self, batch, batch_nb) -> torch.Tensor:
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

        return loss

    def __check_model_config(self, config_dict: dict) -> bool:
        """Return true if config_dict contains keys which are
        num_classes, latent_dim, cnn for CNNClassifier. If
        self.hparams.use_lstm is true, dict also needs to have
        lstm_layers, hidden_dim, bidirectional and attention.
        Return false if keys are missing.
        
        Args:
            config_dict (dict): Dictionary with model configuration
        """
        if not config_dict:
            raise ValueError("config_dict is empty")

        if not config_dict.get("num_classes"):
            raise ValueError("config_dict must contain num_classes for CNNClassifier")
        if not config_dict.get("latent_dim"):
            raise ValueError("config_dict must contain latent_dim for CNNClassifier")
        if not config_dict.get("cnn"):
            raise ValueError("config_dict must contain cnn for CNNClassifier")
        if not self.hparams.use_lstm:  # type: ignore
            if not config_dict.get("lstm_layers"):
                raise ValueError("config_dict must contain lstm_layers for CNNLSTM")
            if not config_dict.get("hidden_dim"):
                raise ValueError("config_dict must contain hidden_dim for CNNLSTM")
            if not config_dict.get("bidirectional"):
                raise ValueError("config_dict must contain bidirectional for CNNLSTM")
            if not config_dict.get("attention"):
                raise ValueError("config_dict must contain attention for CNNLSTM")

        return True
