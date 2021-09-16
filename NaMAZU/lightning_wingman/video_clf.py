import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

__all__ = ["CNNClassifier", "CNNLSTM"]


class FeatureExtractor(nn.Module):
    """CNN module for feature extraction. The last layer is a linear layer with latent_dim output channels
    with batch normalization. The input is a 3-channel RGB image and the output is a latent vector of size latent_dim.

    Attributes:
        backbone (nn.Module): The backbone CNN model.
        final (nn.Module): The linear layer with batch normalization.
    """

    def __init__(
        self, latent_dim: int, cnn: str = "resnet152d", freeze_cnn: bool = True
    ) -> None:
        """
        Args:
            latent_dim (int): Dimension of the latent space. Default: 512.
            cnn (str): CNN model name in timm.models. Default: "resnet152d".
            freeze_cnn (bool): Freeze the backbone CNN model. Default: True.
        """
        super().__init__()
        self.backbone = create_model(
            cnn, pretrained=True, num_classes=0
        )  # delete clf_layer
        self.final = nn.Sequential(
            nn.Linear(self.backbone.num_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
        )
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_cnn

    def forward(self, x: Tensor) -> Tensor:
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
        """LSTM module for classification. The input is a latent vector of size latent_dim and the output is a vector of size hidden_dim.

        Args:
            latent_dim (int): Input dimension of the LSTM.
            num_layers (int, optional): Number of layers in the LSTM. Default: 1.
            hidden_dim (int, optional): Size of the hidden state of the LSTM. Default: 1024.
            bidirectional (bool, optional): Whether to use a bidirectional LSTM. Default: True.
        """
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
        """Reset the hidden state of the LSTM."""
        self.hidden_state = None

    def forward(self, x: Tensor) -> Tensor:
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 512,
        cnn: str = "resnet152d",
        lstm_layers: int = 1,
        hidden_dim: int = 1024,
        bidirectional: bool = True,
        attention: bool = True,
    ) -> None:
        """Video Classifier using LSTM as a frame encoder.

        Args:
            num_classes (int): Number of classes.
            latent_dim (int, optional): Latent space dim from encoder. Defaults to 512.
            backbone (str, optional): Choice of backbone in timm. Defaults to "resnet152d".
            lstm_layers (int, optional): Number of stacking layers in LSTM. Defaults to 1.
            hidden_dim (int, optional): Hidden dim for LSTM. Defaults to 1024.
            bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
            attention (bool, optional): Whether to self attention before classification layer. Defaults to True.
        """
        super().__init__()
        self.encoder = FeatureExtractor(latent_dim, cnn=cnn)
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


class CNNClassifier(nn.Module):
    """Simple CNN classifier"""

    def __init__(self, num_classes: int, latent_dim: int, cnn: str = "resnet152d"):
        """Simple CNN classifier processing all sequential frames as a distinct sample.

        Args:
            num_classes (int): Number of classes.
            latent_dim (int): Dimension of the latent space.
            cnn (str, optional): Feature extractor. Defaults to "resnet152d".
        """
        super().__init__()
        self.feature_extractor = FeatureExtractor(latent_dim=latent_dim, cnn=cnn)
        self.final = nn.Sequential(
            nn.Linear(latent_dim, num_classes), nn.Softmax(dim=-1),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.feature_extractor(x)
        x = x.view(batch_size * seq_length, -1)
        x = self.final(x)
        x = x.view(batch_size, seq_length, -1)
        return x
