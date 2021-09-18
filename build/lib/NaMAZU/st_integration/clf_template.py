from re import I, sub
from NaMAZU.lightning_wingman import KNN
import torch
import pandas as pd
import streamlit as st
from st_utils import *

st.set_page_config(page_title="Made by NMZ", layout="wide")

hide_default_header_and_footer()


def get_data(file):
    if "1st_run" not in st.session_state:
        st.session_state["1st_run"] = True

    if st.session_state["1st_run"]:
        if file:
            df = pd.read_csv(file, sep="\t", index_col=0)  # type: ignore

            place_holder = st.container()
            data_pos = place_holder.empty()
            data_pos.write(df)

            add_y = st.button("Add target label")

            if add_y:
                df["y"] = [0 if x < 6 else 1 for x in df["quality"]]
                data_pos.write(df)

                del place_holder

                st.markdown("---")
                st.markdown("Train KNN")

                x = df.drop(columns=["y"]).reset_index(drop=True).to_numpy()
                y = df["y"].reset_index(drop=True).to_numpy()

                x, y = (
                    torch.tensor(x, dtype=torch.float),
                    torch.tensor(y, dtype=torch.int),
                )

                st.session_state.x = x
                st.session_state.y = y

                st.session_state["1st_run"] = False

                return x, y
    else:
        return st.session_state.x, st.session_state.y


file = st.file_uploader("Upload your own file", type=["csv", "txt"])
out = get_data(file)

if out:
    x, y = out
    model = KNN(
        n_neighbors=5, distance_measure="euclidean", training_data=x, training_labels=y,
    )

    nb = model(x[:10])

    # num_f = st.radio("Number of features", options=["2", "3",],)
    # num_f = list(range(int(num_f)))
    num_f = st.radio("Number of features", options=["2", "3",],)
    num_f = list(range(int(num_f)))

    p = plot_plotly_supervised(
        x=x,
        feature_indices=num_f,
        x_label=y,
        # class_names=["bad", "good"],
        marker_size=1,
    )
    st.plotly_chart(p)
