import streamlit as st
from dashboard.layout import set_page_config
from dashboard.sections.overview import show_overview
from dashboard.sections.cluster_analysis import show_cluster_analysis
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
        [data-testid="stVerticalBlock"]{
            gap: 10px 0px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    set_custom_style()
    df = load_data()
    option = st.sidebar.selectbox(
        "Select a section:",
        ["Overview", "Cluster Analysis"]
    )
    if option == "Overview":
        show_overview(df)
    elif option == "Cluster Analysis":
        show_cluster_analysis(df)

if __name__ == "__main__":
    main()
