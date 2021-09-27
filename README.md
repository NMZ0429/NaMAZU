<div align="center">

<img src="utils/namazu_fixed.png" width="450">

**Many utilities for ML**

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NaMAZU)](https://pypi.org/project/NaMAZU/)
[![PyPI version](https://badge.fury.io/py/NaMAZU.svg)](https://badge.fury.io/py/NaMAZU)
[![PyPI Status](https://pepy.tech/badge/NaMAZU)](https://pepy.tech/project/NaMAZU)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NMZ0429/NaMAZU/blob/main/LICENSE)
![pl](https://img.shields.io/badge/PyTorch%20Lightning-1.3-792EE5.svg?logo=PyTorch%20Lightning&style=popout)


* * *

</div>

# NaMAZU

## Installation

Version in pip server might be older than this repo.

```zsh
pip install NaMAZU
```

## Lightning API

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PL](https://img.shields.io/badge/-PyTorch%20Lightning-792EE5.svg?logo=PyTorch%20Lightning&style=for-the-badge)

They are all written in PyTorch following best practice to be used with pytorch lightning. They are all GPU enabled controlled by Lightning API. You will never need to call `to("cuda")` to use the model on any device even with multi-GPU training!

```python
import pytorch_lightning as pl
from NaMAZU import KNN, GMM

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
        y_hat = self.head_classifier(y)
        probability = self.estimator.predict_proba(y)
```

### Statistical Model

* **KNN**: Available with euqlidean, manhattan, cosine and mahalanobis distance.
* **NBC**: GPU enabled naive bayes classifier.
* **GMM**: Gaussian Mixture probabability estimator. Of course GPU enabled.

### Deep Learning

They are all ready-to-train models with MNIST, ImageNet, UCF101 etc... using [LightingDataModule](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html).

Some models come with their pretrained-weight available by auto-downloading.

* **LitU2Net**: LightningModule U2Net. Trainable and ready for prediction.
* **AniNet**: LightningModule image classifier pretrained for japanese animations.
* **LitVideoClf**: LightningModule video classfier using either single frame CNN or CNNLSTM.
* **MultiModalNet**: LightningModule for multi-modal learning which can learn any modality with high robustness. Can be combined with any backbone.

* * *

## Functional API

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SKlearn](https://img.shields.io/badge/Scikit_learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
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
* apply_mask_to
* apply_to_all
* change_frame_rates_in
* save_all_frames
* collect_images (requires [icrawler](https://icrawler.readthedocs.io/en/latest/))

### file_control

* rename_file
* collect_file_pathes_by_ext
* zip_files
* export_list_str_as

### text_control

* search_word_from

### data_science

* train_linear_regressor

* * *

## Visual Integration

![Plotly](https://img.shields.io/badge/Plotly-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)

### st_utils

* hide_default_header_and_footer
* plot_plotly_supervised

* * *

## :rocket: Coming

* [ ] 2. PredictionAssistant
* [x] 2. Video Recognition Model
* [ ] 3. Feature Learning
* [ ] 4. Few-shot Learning
* [ ] 5. Audio-Visual Multimodal fusion (finish docstrings)
* [ ] 6. BBox template finding
* [ ] 7. CACNet
