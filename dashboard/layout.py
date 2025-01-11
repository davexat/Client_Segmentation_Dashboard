import streamlit as st
from dashboard.config import DEFAULT_BACKGROUND_COLOR as defcolor

def set_page_config():
    st.set_page_config(layout="wide")

def create_container(key, color = defcolor):
    st.markdown(
        f"""
        <style>
        .st-key-{key}{{
            background-color: {color};
            padding: 0px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    return st.container(border=False, key=key)