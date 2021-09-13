import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timm.models import create_model
from torchvision.models import resnet152


class FeatureExtractor(nn.Module):
    """CNN module for feature extraction. The last layer is a linear layer with latent_dim output channels
    with batch normalization. The input is a 3-channel RGB image and the output is a latent vector of size latent_dim.

    Attributes:
        backbone (nn.Module): The backbone CNN model.
        final (nn.Module): The linear layer with batch normalization.
    """

    def __init__(self, latent_dim: int, cnn: str = "resnet152d") -> None:
        """
        Args:
            latent_dim (int): Dimension of the latent space. Default: 512.
            cnn (str): CNN model name in timm.models. Default: "resnet152d".
        """
        super().__init__()
        self.backbone = create_model(
            cnn, pretrained=True, num_classes=0
        )  # delete clf_layer
        self.final = nn.Sequential(
            nn.Linear(self.backbone.num_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
        )

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x


class LSTM(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_layers: int = 1,
        hidden_dim: int = 1024,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.hidden_state = None

    def reset_hidden_state(self) -> None:
        self.hidden_state = None

    def forward(self, x: Tensor) -> Tensor:
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


# TODO: here


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim=512,
        backbone="resnet152d",
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    ):
        super(CNNLSTM, self).__init__()
        self.encoder = FeatureExtractor(latent_dim, cnn=backbone)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.attention_layer = nn.Linear(
            2 * hidden_dim if bidirectional else hidden_dim, 1
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers(x)


class ConvClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(ConvClassifier, self).__init__()
        resnet = resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
            nn.Linear(latent_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.feature_extractor(x)
        x = x.view(batch_size * seq_length, -1)
        x = self.final(x)
        x = x.view(batch_size, seq_length, -1)
        return x
