<div align="center">

<img src="utils/namazu_fixed.png" width="450">

### Many utilities for ML

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NaMAZU)](https://pypi.org/project/NaMAZU/)
[![PyPI version](https://badge.fury.io/py/NaMAZU.svg)](https://badge.fury.io/py/NaMAZU)
[![PyPI Status](https://pepy.tech/badge/NaMAZU)](https://pepy.tech/project/NaMAZU)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NMZ0429/NaMAZU/blob/main/LICENSE)
![pl](https://img.shields.io/badge/PyTorch%20Lightning-1.3-792EE5.svg?logo=PyTorch%20Lightning&style=popout)
![onnx](https://img.shields.io/badge/ONNX-1.10-005CED.svg?logo=ONNX&style=popout)

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

### Deep Learning Models

Collection of SOTA or robust baseline models for multiple tasks fully written in pytorch lightning! They are all ready-to-train models with MNIST, ImageNet, UCF101 etc... using [LightingDataModule](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html).

Some models come with their pretrained-weight available by auto-downloading.

```python
import pytorch_lightning as pl
from NaMAZU.lightningwingman import LitVideoClf

config = {"num_classes": 10, "cnn": "resnet152d", "latent_dim":512}
model = LitVideoClf(use_lstm=False, model_config=config)

... 
# use bolts to get datamodule and pass model and datamodule to pl.trainer!
```

* **LitU2Net**: LightningModule U2Net. Trainable and ready for prediction.
* **AniNet**: LightningModule image classifier pretrained for japanese animations.
* **LitVideoClf**: LightningModule video classfier using either single frame CNN or CNNLSTM.
* **MultiModalNet**: LightningModule for multi-modal learning which can learn any modality with high robustness. Can be combined with any backbone.

### Feature Learning Interface

Before starting your fine-tuning training, try this trianign API that produces better initial weight by running a self-supervised learning to your training dataset. Only images are used and no annotation nor data cleaning is required.

Other training schemes are coming soon!

```python
from NaMAZU.lightingwingman import self_supervised_learning

# images may be stored in single or multiple directories. Stratified sampling is supported!
dir_images = "dataset/something"
dir_images2 = "dataset/something2"

self_supervised_training(
    "resnet50", 
    [dir_images, dir_images2],
    batch_size=64,
    save_dir="pretrained_models/"
    )
```

* self_supervised_training: Simple interface that you can obtain self-supervised CNN with just one line of code!

### Statistical Models

They are all written in PyTorch following best practice to be used with pytorch lightning. They are all GPU enabled controlled by Lightning API. You will never need to call `to("cuda")` to use the model on any device even with multi-GPU training!

```python
import pytorch_lightning as pl
from NaMAZU.lightningwingman import KNN, GMM

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

* **KNN**: Available with euqlidean, manhattan, cosine and mahalanobis distance.
* **NBC**: GPU enabled naive bayes classifier.
* **GMM**: Gaussian Mixture probabability estimator. Of course GPU enabled.

* * *

## ONNX API

![ONNX](https://img.shields.io/badge/ONNX-005CED.svg?style=for-the-badge&logo=ONNX&logoColor=white)

We provide many readly to use ONNX models comes with preprocess and postprocess methods. They are packed as an class object and you can use it without any coding!

1. MiDAS: Mono Depth Prediction (Light and Large models are available)

* * *

## Functional API

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SKlearn](https://img.shields.io/badge/Scikit_learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-007808.svg?style=for-the-badge&logo=FFmpeg&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white)

You can use below functions via

```python
import NaMAZU.functional as F

F.change_frame_rates("./test_data.mp4",fps=5)
```

### image_control

<details><summary>List of functions</summary><div>

* npy_to_img
* img_to_npy
* split_image
* compose_two_png
* apply_mask_to
* apply_to_all
* change_frame_rates
* save_all_frames
* make_video_from_frames
* collect_images (requires [icrawler](https://icrawler.readthedocs.io/en/latest/))

</div></details>

### file_control

<details><summary>List of functions</summary><div>

* rename_file
* collect_file_pathes_by_ext
* zip_files
* randomly_choose_files
* export_list_str_as

</div></details>

### text_control

<details><summary>List of functions</summary><div>

* search_word_from

</div></details>

### data_science

<details><summary>List of functions</summary><div>

* train_linear_regressor
* parse_tab_seperated_txt

Sampling Theory

* calculate_sample_stats
* error_bound_of_mean
* calculate_sufficient_n_for_mean
* estimated_total
* error_bound_of_total
* calculate_sufficient_n_for_population_total
* calculate_sufficient_n_for_proportion
* calculate_sufficient_n_for_proportion

Regression Analysis

* sxy_of
* sxx_of
* least_square_estimate
* estimate_variance_of_linear_regressor
* t_statistic_of_beta1
* calculate_CI_of_centred_model_at
* get_prediction_interval
* t_stats_for_correlation
* get_p_value_of_tstat
* fit_general_least_square_regression

Correlation Analysis

* "get_prediction_interval"
* "t_stats_for_correlation"
* "get_p_value_of_tstat"
* "_search_t_table"
* "get_alt_sxx"
* "get_alt_sxy"

</div></details>

### coreml

<details><summary>List of functions</summary><div>

* drop_negative

</div></details>

* * *

## Visual Integration

![Plotly](https://img.shields.io/badge/Plotly-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)

### st_utils

* hide_default_header_and_footer
* plot_plotly_supervised

## Decorator

Some utility decorators to speed up your development.

* print_docstring
* measure_runtime

* * *

## :rocket: Coming

* [ ] 2. PredictionAssistant
* [x] 2. Video Recognition Model
* [x] 3. Feature Learning
* [ ] 4. Few-shot Learning
* [ ] 5. Audio-Visual Multimodal fusion (finish docstrings)
* [ ] 6. BBox template finding
* [ ] 7. CACNet
