import streamlit as st

def set_page_config():
    st.set_page_config(layout="wide")

def create_container(key, color = "#1B1D22"):
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