import streamlit as st
from dashboard.config import DEFAULT_BACKGROUND_COLOR as defcolor

def set_page_config():
    st.set_page_config(layout="wide")

def create_container(key, color = defcolor, padding = 0):
    st.markdown(
        f"""
        <style>
        .st-key-{key}{{
            background-color: {color};
            padding: {padding}px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    return st.container(border=False, key=key)

def format_large_numbers(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:.2f}"

def create_header(title, value, txt_color):
    style = '''
    <div style="text-align: center; color: {txt_color}; padding: 0;">
        <h6 style="margin-bottom: 0; padding: 15px 0 0 0;">{title}</h6>
        <h1 style="margin-top: 0; padding: 0 0 10px 0;">{value}</h1>
    </div>
    '''
    return style.format(title=title, value=value, txt_color=txt_color)