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

## Functional API

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![FFmpeg](https://img.shields.io/badge/FFmpeg-007808.svg?style=for-the-badge&logo=FFmpeg&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white)

### image_control

* npy_to_img
* img_to_npy
* split_image
* compose_two_png
* apply_to_all
* change_frame_rates_in
* save_all_frames

### file_control

* collect_file_pathes_by_ext
* zip_files

## Lightning API

![PL](https://img.shields.io/badge/-PyTorch%20Lightning-792EE5.svg?logo=PyTorch%20Lightning&style=for-the-badge) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

* PredictionAssistant

## Coming

1. composer. auto compose functions into one sequential operation.
2. st_integration. Usuful snipets and fast deoployment of LitModule to streamlit.

### Internal

```shell
sphinx-apidoc -f -o ./docs_src ./NaMAZU
sphinx-build ./docs_src ./docs
```
