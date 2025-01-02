import streamlit as st
from dashboard.layout import set_page_config
from dashboard.sections.overview import show_overview
from dashboard.sections.cluster_analysis import show_cluster_analysis
from dashboard.sections.comparissons import show_comparissons

set_page_config()

def main():
    st.title("Customer Segmentation Dashboard")

    # Sidebar navigation
    st.sidebar.header("Navigation")
    option = st.sidebar.selectbox(
        "Select a section:",
        ["Overview", "Cluster Analysis", "Comparissons"]
    )

    # Render sections dynamically
    if option == "Overview":
        show_overview()
    elif option == "Cluster Analysis":
        show_cluster_analysis()
    elif option == "Comparissons":
        show_comparissons()

if __name__ == "__main__":
    main()