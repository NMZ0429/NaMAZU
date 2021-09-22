from os.path import sep
from pathlib import Path
from typing import List, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


import cv2
import numpy as np
import torch
from NaMAZU.functional.image_control import apply_mask_to
from PIL import Image
from PIL.Image import Image as PILImage
from pytorch_lightning import LightningModule
from torch import Tensor
from torchvision import transforms

from .u2net import U2NET, U2NETP, RescaleT, ToTensorLab

__all__ = ["LitU2Net"]


class LitU2Net(LightningModule):
    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 1,
        model_type: Literal["basic", "mobile", "human", "portrait"] = "basic",
        train_model: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.__load_model()
        self.preprocess = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])

        if train_model:
            self.model.train()
            self.bce_loss = torch.nn.BCELoss(size_average=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        leading_loss, loss = self.__multi_bce_loss_fusion(*y_hat, labels_v=y)

        return {"loss": loss, "log": {"train_loss": loss, "train_tar": leading_loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        leading_loss, loss = self.__multi_bce_loss_fusion(*y_hat, labels_v=y)

        return {"val_loss": loss, "log": {"val_loss": loss, "val_tar": leading_loss}}

    def __multi_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):

        loss0 = self.bce_loss(d0, labels_v)
        loss1 = self.bce_loss(d1, labels_v)
        loss2 = self.bce_loss(d2, labels_v)
        loss3 = self.bce_loss(d3, labels_v)
        loss4 = self.bce_loss(d4, labels_v)
        loss5 = self.bce_loss(d5, labels_v)
        loss6 = self.bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print(
            "l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"
            % (
                loss0.data.item(),
                loss1.data.item(),
                loss2.data.item(),
                loss3.data.item(),
                loss4.data.item(),
                loss5.data.item(),
                loss6.data.item(),
            )
        )

        return loss0, loss

    def predict(self, x_path: str, save: bool = False, save_path: str = "",) -> Tensor:
        x = cv2.imread(x_path)
        x = self.__input_preprocess(x)
        d1 = self.forward(x)[0]
        pred = d1[:, 0, :, :]
        pred = self.__normPRED(pred)
        if save:
            self.__save_output(x_path, pred, save_path)

        return pred

    def __normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

    def __save_output(self, original_image: str, pred: Tensor, d_dir: str):

        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        im = Image.fromarray(predict_np * 255).convert("RGB")
        img_name = original_image.split(sep)[-1]
        # image = io.imread(original_image)
        image = Image.open(original_image)
        imo = im.resize((image.size[0], image.size[1]), resample=Image.BILINEAR)

        pb_np = np.array(imo)

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        out_dir = Path(d_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        imo.save(out_dir / (imidx + ".png"))

    def apply_mask(self, prediction: Tensor, original_image: str) -> PILImage:
        predict_np = prediction.squeeze().cpu().data.numpy()

        im = Image.fromarray(predict_np * 255).convert("RGB")
        image = Image.open(original_image)
        mask = im.resize((image.size[0], image.size[1]), resample=Image.BILINEAR)

        return apply_mask_to(image, mask)

    def __input_preprocess(self, x: PILImage) -> Tensor:
        processed = self.preprocess(x).type(torch.FloatTensor)  # type: ignore
        return processed

    def __load_model(self) -> None:
        """Download checkpoint file and load the model.
        """
        url_dict = {
            "basic": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/basic.pth",
            "mobile": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/mobile.pth",
            "human": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/human_seg.pth",
            "portrait": "https://github.com/NMZ0429/NaMAZU/releases/download/Checkpoint/portrait.pth",
        }

        if not self.hparams.model_type in url_dict:  # type: ignore
            raise ValueError(f"model_type {self.hparams.model_type} is not supported")  # type: ignore

        url = url_dict[self.hparams.model_type]  # type: ignore
        if self.hparams.model_type == "mobile":  # type: ignore
            self.model = U2NETP(
                in_chans=self.hparams.in_chans, out_chans=self.hparams.out_chans  # type: ignore
            )
        else:
            self.model = U2NET(
                in_chans=self.hparams.in_chans, out_chans=self.hparams.out_chans  # type: ignore
            )
        st_dict = torch.hub.load_state_dict_from_url(url, map_location=self.device,)
        self.model.load_state_dict(state_dict=st_dict)
        self.model.eval()
