import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import streamlit as st

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