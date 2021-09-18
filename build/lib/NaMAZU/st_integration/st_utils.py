from typing import Any
import streamlit as st


def hide_default_header_and_footer():
    hide_default = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_default, unsafe_allow_html=True)


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
