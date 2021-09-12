from glob import glob
from typing import List, Union

import torch
from PIL import Image
from PIL.Image import Image as PILImage
from pytorch_lightning import LightningModule
from torch.functional import Tensor
from torchvision import transforms
from NaMAZU.lightning_wingman.aninet import aninet18, aninet34, aninet50
from NaMAZU.lightning_wingman.ani_label import label_6000


__all__ = ["AniNet"]


class AniNet(LightningModule):
    def __init__(self, choice: str = "50") -> None:
        super().__init__()
        models = {"18": aninet18, "34": aninet34, "50": aninet50}
        self.model = models[choice]().eval()
        self.__set_preprocess()
        self.__load_label()

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        return self.model(x)

    def predict(self, image_path: Union[str, List[str]]) -> torch.Tensor:
        batch = self._load_img(image_path)
        # img = Image.open(image_path)
        with torch.no_grad():
            if type(batch) != list:
                batch = self.preprocess(batch).unsqueeze(0)  # type: ignore
            else:
                batch = [self.preprocess(img) for img in batch]  # type: ignore
                batch = torch.stack(batch, dim=0)
            return self.forward(batch)

    def predict_probs(self, outputs):
        probs = []
        for row in outputs:
            probs.append(torch.sigmoid(row))

        return probs

    def calc_result(self, probs: List[Tensor], thresh=0.3) -> List[str]:
        results: List[str] = []
        for prob in probs:
            tmp = prob[prob > thresh]
            inds = prob.argsort(descending=True)
            # txt = "## Predictions with probabilities above " + str(thresh) + ":\n"
            txt = ""
            for i in inds[0 : len(tmp)]:
                txt += (
                    "* "
                    + self.label[i]
                    + ": {:.4f} \n".format(prob[i].cpu().numpy())
                    + "\n"
                )

            results.append(txt)

        return results

    def run_batch_prediction(self, image_dir: str):
        img_files = sorted(glob(image_dir + "/*"))
        prediction = self.predict(img_files)
        probs = self.predict_probs(prediction)
        results = self.calc_result(probs)

        return results

    def __set_preprocess(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(360),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]
                ),
            ]
        )

    def __load_label(self):
        # TODO: update label for newer model
        self.label = label_6000

    def _load_img(
        self, image_path: Union[str, List[str]]
    ) -> Union[PILImage, List[PILImage]]:
        if type(image_path) == str:
            img = Image.open(image_path).convert("RGB")  # type: ignore
            return img
        elif type(image_path) == list:
            img_batch = []
            for i_path in image_path:
                img = Image.open(i_path).convert("RGB")
                img_batch.append(img)
            return img_batch
        else:
            raise TypeError(
                "image_path must be str or list of str, but got {}".format(
                    type(image_path)
                )
            )
