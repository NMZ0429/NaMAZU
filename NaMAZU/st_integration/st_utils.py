import pandas as pd
import torch
from typing import Any, List, Tuple, Dict
import streamlit as st
import plotly.express as px


def hide_default_header_and_footer():
    hide_default = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_default, unsafe_allow_html=True)


def plot_plotly_supervised(
    x: torch.Tensor,
    feature_indices: List[int],
    x_label: torch.Tensor = None,
    class_names: List[str] = None,
    marker_size: int = None,
    distinc_markder: bool = False,
):
    """Plot the data in a plotly graph.

    Args:
        x: Tensor of shape (n_samples, n_features).
        x_label: Tensor of shape (n_samples, 1).
        feature_indices: List of indices of features to be plotted. 
    """

    if len(feature_indices) == 2:
        first_feature, second_feature = feature_indices
        data = {"x": x.numpy()[:, first_feature], "y": x.numpy()[:, second_feature]}
        kargs: Dict[str, Any] = {"x": "x", "y": "y"}
    elif len(feature_indices) == 3:
        first_feature, second_feature, third_feature = feature_indices
        data = {
            "x": x.numpy()[:, first_feature],
            "y": x.numpy()[:, second_feature],
            "z": x.numpy()[:, third_feature],
        }
        kargs: Dict[str, Any] = {"x": "x", "y": "y", "z": "z"}
    else:
        raise ValueError("feature_indices must be of length 2 or 3.")

    df = pd.DataFrame.from_dict(data)

    if x_label is not None:
        # prepare as continuous label
        df["target_value"] = x_label.numpy().tolist()  # px.colors.sequential.Plasma
        kargs["color"] = "target_value"
        kargs["color_continuous_scale"] = px.colors.sequential.Viridis

        # make it discrete
        if class_names is not None:
            df["class_name"] = [class_names[i] for i in df["target_value"]]
            kargs["symbol"] = "class_name"

            df = df.drop(columns=["target_value"])
            kargs["color"] = "class_name"
            del kargs["color_continuous_scale"]
            kargs["color_discrete_sequence"] = px.colors.qualitative.Plotly

    fig = (
        px.scatter(df, **kargs)
        if len(feature_indices) == 2
        else px.scatter_3d(df, **kargs)
    )

    # update the marker style
    m_size = 10 if marker_size is None else marker_size
    line_tick = 0 if not distinc_markder else m_size
    fig.update_traces(
        marker=dict(size=m_size, line=dict(width=line_tick, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    return fig


# TODO: implement this function
def accept_data(x: Any) -> Any:
    """Accept any types of data and return it as convenient type.


    Args:
        x: Any type of data.
    
    Returns:
        Any: Accepted data.
    """

    if isinstance(x, str):
        return x
    elif isinstance(x, list):
        return x
    elif isinstance(x, dict):
        return x
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, set):
        return x
    elif isinstance(x, float):
        return x
    elif isinstance(x, int):
        return x
    elif isinstance(x, bool):
        return x
    elif isinstance(x, type(None)):
        return x
    else:
        return x
