import streamlit as st
from dashboard.layout import set_page_config
from dashboard.sections.overview import show_overview
from dashboard.sections.cluster_analysis import show_cluster_analysis
from dashboard.sections.comparissons import show_comparissons
from dashboard.data import load_data

set_page_config()

def set_custom_style():
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"]{
            padding: 30px;
        }
        [data-testid="stVerticalBlockBorderWrapper"]{
            padding: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    set_custom_style()
    df = load_data()
    show_overview(df)

if __name__ == "__main__":
    main()
