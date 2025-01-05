import streamlit as st
from customer_segmentation.dashboard.layout import set_page_config
from customer_segmentation.dashboard.sections.overview import show_overview
from customer_segmentation.dashboard.sections.cluster_analysis import show_cluster_analysis
from customer_segmentation.dashboard.sections.comparissons import show_comparissons
from customer_segmentation.dashboard.data import load_data

set_page_config()

def set_custom_style():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    set_custom_style()
    #st.title("Customer Segmentation Dashboard")
    # Sidebar navigation
    st.sidebar.header("Navigation")
    option = st.sidebar.selectbox(
        "Select a section:",
        ["Overview", "Cluster Analysis", "Comparissons"]
    )
    df = load_data()
    # Render sections dynamically
    if option == "Overview":
        show_overview(df)
    elif option == "Cluster Analysis":
        show_cluster_analysis(df)
    elif option == "Comparissons":
        show_comparissons(df)

if __name__ == "__main__":
    main()
