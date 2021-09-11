<div align="center">

<img src="utils/namazu_fixed.png" width="450">

**Libray including many(not yet) utilities**

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![PyPI version](https://badge.fury.io/py/NaMAZU.svg)](https://badge.fury.io/py/NaMAZU)
![pl](https://img.shields.io/badge/PyTorch%20Lightning-1.3-792EE5.svg?logo=PyTorch%20Lightning&style=popout)
![st](https://img.shields.io/badge/Streamlit-0.88-FF4B4B.svg?logo=Streamlit&style=popout)
![numpy](https://img.shields.io/badge/NumPy-1.21-013243.svg?logo=NumPy&style=popout)
![sklearn](https://img.shields.io/badge/Scikit_learn-0.23-F7931E.svg?logo=scikit-learn&style=popout)

* * *

</div>

# NaMAZU

## Lightning API

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PL](https://img.shields.io/badge/-PyTorch%20Lightning-792EE5.svg?logo=PyTorch%20Lightning&style=for-the-badge)

They are all written in PyTorch and following best practice to be used with pytorch lightning. They are all GPU enabled controlled by Lightning API.

```python
import pytorch_lightning as pl
from NaMAZU import KNN

class YourLitModule(pl.LightningModule):
    def __init__(self,*args, **kwargs):
        ...
        self.encoder = SomeEncoder()
        self.head_classifier = KNN(
            n_neighbors=5, 
            distance_measure="cosine", 
            training_data=some_known_data
        )
        self.estimator = GMM(5, 10)

    def training_step(self, batch):
        x, t = batch
        y = self.encoder(x)
        neighbors = self.head_classifier(y)
        probability = self.estimator(y)
```

### Statistical Model

* KNN: Available with euqlidean, manhattan, cosine and mahalanobis distance.
* NBC: GPU enabled naive bayes classifier.
* GMM: Gaussian Mixture probabability estimator. Of course GPU enabled.

### Deep Learning

* LitU2Net: LightningModule U2Net. Trainable and ready for prediction.
* PredictionAssistant: Coming soon.

## Functional API

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-007808.svg?style=for-the-badge&logo=FFmpeg&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white)

You can use below functions via

```python
import NaMAZU.functional as F

F.change_frame_rates_in("./test_data.mp4",fps=5)
```

### image_control

* npy_to_img
* img_to_npy
* split_image
* compose_two_png
* apply_to_all
* change_frame_rates_in
* save_all_frames

### file_control

* rename_file
* collect_file_pathes_by_ext
* zip_files

## Coming

1. st_integration. Usuful snipets and fast deoployment of LitModule to streamlit.

## **TODO**

1. Video Recognition Model
2. Feature Learning
3. Few-shot Learning
4. Audio-Visual Multimodal
