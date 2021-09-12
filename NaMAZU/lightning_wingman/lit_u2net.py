from os.path import sep

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningModule
from torch import Tensor
from torchvision import transforms

from .u2net import U2NET, RescaleT, ToTensorLab


__all__ = ["LitU2Net"]

# TODO: implement U2NETP
class LitU2Net(LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        in_chans: int = 3,
        out_chans: int = 1,
        pretrained_weight: str = None,
        train_model: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = U2NET(in_ch=in_chans, out_ch=out_chans)
        if self.hparams.pretrained_weight:  # type: ignore
            self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()

        if not train_model:
            self.model.train()
            self.bce_loss = torch.nn.BCELoss(size_average=True)

    def forward(self, x):
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
        leading_loss, loss = self.__muti_bce_loss_fusion(*y_hat, labels_v=y)

        return {"loss": loss, "log": {"train_loss": loss, "train_tar": leading_loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        leading_loss, loss = self.__muti_bce_loss_fusion(*y_hat, labels_v=y)

        return {"val_loss": loss, "log": {"val_loss": loss, "val_tar": leading_loss}}

    def __muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):

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

    def predict(
        self,
        x_path: str,
        save: bool = False,
        image_path: str = "",
        save_path: str = "",
    ):
        x = Image.open(x_path)
        x = transforms.ToTensor()(x).unsqueeze_(0)
        x = torch.tensor(x)
        x = self.__input_preprocess(x)
        d1 = self.forward(x)[0]
        pred = d1[:, 0, :, :]
        pred = self.__normPRED(pred)
        if save:
            self.__save_output(image_path, pred, save_path)

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
        imo = im.resize((image.size[1], image.size[0]), resample=Image.BILINEAR)

        pb_np = np.array(imo)

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        imo.save(d_dir + imidx + ".png")

    def __input_preprocess(self, x: Tensor):
        trans = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])

        return trans(x)
