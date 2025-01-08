import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import streamlit as st
from customer_segmentation.utils.paths_internal import data_processed_dir

@st.cache_data
def load_data():
    dataset_path = data_processed_dir("cleaned_dataset.csv")
    df = pd.read_csv(dataset_path)
    df = df.drop(columns=["ID", "dias_primera_compra", "n_clicks", "info_perfil"])
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    dbscan = DBSCAN(eps=0.040, min_samples=50)
    df['cluster'] = dbscan.fit_predict(df_scaled)
    df_scaled = df_scaled[df["cluster"] != -1]
    df = df[df["cluster"] != -1]
    return df