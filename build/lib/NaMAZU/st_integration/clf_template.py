from NaMAZU.lightning_wingman import KNN
import pandas as pd
import streamlit as st
from st_utils import *

st.set_page_config(page_title="Made by NMZ", layout="wide")

hide_default_header_and_footer()

file = st.file_uploader("Upload your own file", type=["csv", "txt"])

if file:
    df = pd.read_csv(file, sep="\t", index_col=0)  # type: ignore

    place_holder = st.empty()
    place_holder.text("Loading...")
    place_holder.write(df)

    add_y = st.button("Add target label")

    if add_y:
        df["y"] = [0 if x < 6 else 1 for x in df["quality"]]
        with place_holder.container():
            st.write(df)

        st.markdown("---")
        st.markdown("Train KNN")

        x = df.drop(columns=["y"]).reset_index(drop=True).to_numpy()
        y = df["y"].reset_index(drop=True).to_numpy()

        model = KNN(
            n_neighbors=5,
            distance_measure="euclidean",
            training_data=x,
            training_labels=y,
        )

        st.write(model)

        nb = model(x[:10])
