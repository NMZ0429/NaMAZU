from pandas import read_csv
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from resnet_anime import resnet50

st.set_page_config(page_title="Made by NMZ", layout="wide")


@st.cache
def load_resnet():
    # use resnet50
    model = resnet50()
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    return model


@st.cache
def preprocess(img):
    preprocess = transforms.Compose(
        [
            transforms.Resize(360),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]
            ),
        ]
    )
    batch = preprocess(img).unsqueeze(0)

    if torch.cuda.is_available():
        batch = batch.to("cuda")

    return batch


def predict_probs(batch, model):
    with torch.no_grad():
        output = model(batch)
        probs = torch.sigmoid(output[0])

    return probs


@st.cache
def get_label():
    column_names = ["_", "en", "ja"]
    df = read_csv("label_ja.csv", names=column_names)
    # class_names = json.load(open("label_en.json", "r"))
    class_names = df.ja.to_list()

    return class_names


class_names = get_label()


@st.cache
def calc_result(probs, thresh=0.3):
    tmp = probs[probs > thresh]
    inds = probs.argsort(descending=True)
    # txt = "## Predictions with probabilities above " + str(thresh) + ":\n"
    txt = ""
    for i in inds[0 : len(tmp)]:
        txt += (
            "* " + class_names[i] + ": {:.4f} \n".format(probs[i].cpu().numpy()) + "\n"
        )

    return txt


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

model = load_resnet()

st.markdown(
    "# 教師無し対照学習で汎用オタクCNNを構築する\n\n Image understanding with respect to a specific field (Anime) using Inception trained by SimCLR"
)

st.markdown("対照学習でアニメ関連の画像をひたすら学習させたらCNNがオタクっぽいことを学習できるか実験")

c1, c2 = st.beta_columns(2)

imgfile = c1.file_uploader(
    "Upload Image: (must be at least 360^2)",
    type=["png", "jpg"],
    accept_multiple_files=False,
)

if imgfile:
    image = Image.open(imgfile)
    c1.image(image, caption="upload images", use_column_width=True)
    x = preprocess(image)
    out = predict_probs(batch=x, model=model)
    result = calc_result(out)

    c2.markdown("## 抽出した特徴...(信頼区間>0.3)")
    c2.markdown(result)
