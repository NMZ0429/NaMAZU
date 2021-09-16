from typing import Dict, List, Tuple, Union

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from video_clf import CNNLSTM, CNNClassifier

__all__ = ["LitVideoClf"]


class LitVideoClf(LightningModule):
    def __init__(self, use_lstm: bool, model_config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

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
