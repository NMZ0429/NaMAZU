import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from .u2net import U2NET
import torch

# TODO: implement U2NETP
class LitU2Net(LightningModule):
    def __init__(
        self, ckpt_path: str, in_chans: int = 3, out_chans: int = 1, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = U2NET(in_ch=in_chans, out_ch=out_chans)
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        return self.forward(x)
