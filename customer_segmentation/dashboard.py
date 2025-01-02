import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import plotly.express as px


@st.cache_data
def load_data():
    dataset_path = "F:/Xathorus/Cosas/programming-projects/projects-jupyter/customer_segmentation/data/processed/cleaned_dataset.csv"
    df = pd.read_csv(dataset_path)
    df = df.drop(columns=["ID", "dias_primera_compra", "n_clicks", "info_perfil"])
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    dbscan = DBSCAN(eps=0.040, min_samples=50)
    df['cluster'] = dbscan.fit_predict(df_scaled)
    df_scaled = df_scaled[df["cluster"] != -1]
    df = df[df["cluster"] != -1]
    return df

def main():
    st.title("Customer Segmentation Dashboard")

    # Sidebar navigation
    st.sidebar.header("Navigation")
    option = st.sidebar.selectbox(
        "Select a section:",
        ["Overview", "Cluster Analysis", "Insights"]
    )

    # Render sections dynamically
    if option == "Overview":
        show_overview()
    elif option == "Cluster Analysis":
        show_cluster_analysis()
    elif option == "Comparissons":
        show_comparissons()

# Section: Overview
def show_overview():
    df = load_data()
    # Definir los estilos comunes en una variable
    style = '''
    <div style="text-align: center; margin: 0;">
        <h6 style="margin-bottom: 0;">{title}</h6>
        <h1 style="margin-top: 0;">{value}</h1>
    </div>
    '''
    gb = df.groupby('cluster').size()
    total, total_0, total_1, total_2 = gb.sum(), gb.loc[0], gb.loc[1], gb.loc[2]
    # Crear las 4 columnas
    col1, col2, col3, col4 = st.columns(4)
    # Contenido en cada columna utilizando f-string para insertar los valores dinámicos
    with col1:
        st.markdown(f"{style}".format(title="Total customers", value=total), unsafe_allow_html=True)
    with col2:
        st.markdown(f"{style}".format(title="High Spenders", value=total_0), unsafe_allow_html=True)
    with col3:
        st.markdown(f"{style}".format(title="Moderate Engagers", value=total_1), unsafe_allow_html=True)
    with col4:
        st.markdown(f"{style}".format(title="Active Savers", value=total_2), unsafe_allow_html=True)

# Section: Cluster Analysis
def show_cluster_analysis():
    st.header("Cluster Analysis")
    st.write("Detailed analysis for each cluster.")
    # Placeholder for cluster-specific visualizations

# Section: Insights
def show_comparissons():
    st.header("Insights")
    st.write("Actionable insights derived from the analysis.")
    # Placeholder for conclusions and recommendations

if __name__ == "__main__":
    main()


