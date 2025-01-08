import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dashboard.config import CLUSTER_COLORS as colors

def show_cluster_analysis(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        plot_pie_chart(df, "cluster")
    with col2:
        plot_pie_chart(df, "n_visitas")
    with col3:
        plot_pie_chart(df, "monto_compras")
    with col4:
        plot_pie_chart(df, "monto_descuentos")
        
    # plot_pie_chart(df, "cluster")
    # plot_pie_chart(df, "n_visitas")
    # plot_pie_chart(df, "monto_compras")
    # plot_pie_chart(df, "monto_descuentos")

    # Subsección para gráficos de dispersión
    plot_scatter_plots(df)

def plot_pie_chart(df, column):
    data = df.groupby('cluster')[column].sum() if column != "cluster" else df['cluster'].value_counts()
    labels = ["High Spenders", "Moderate Engagers", "Active Savers"]
    fig = px.pie(
        names=labels,
        values=data.values.tolist(),
        color=labels,
        color_discrete_map=colors,
        title=f"Distribution of {column}",
        template="plotly_dark"
    )
    fig.update_layout(
        margin=dict(l=30, r=30, t=0, b=0),
        title=dict(
            font=dict(size=16),
            y=0.9  # Ajusta la posición vertical del título (más cerca del gráfico)
        ),
        legend=dict(
            font=dict(size=15),              # Leyenda más compacta
            x=0.16,                           # Posición horizontal (centrada)
            y=0                         # Posición vertical (debajo del gráfico)
        ),
        height=400
    )
    st.plotly_chart(fig)  # Mostrar directamente

def plot_scatter_plots(df):
    scatter_vars = [("n_visitas", "monto_compras"), ("n_visitas", "monto_descuentos"), ("monto_compras", "monto_descuentos")]
    df['cluster_label'] = df['cluster'].map({0: "High Spenders", 1: "Moderate Engagers", 2: "Active Savers"})
    for var_x, var_y in scatter_vars:
        scatter_fig = px.scatter(
            df, x=var_x, y=var_y, color='cluster_label',
            title=f"Scatter: {var_x} vs {var_y}",
            template="plotly_dark",
            color_discrete_map=colors
        )
        scatter_fig.update_traces(marker=dict(size=8, opacity=0.8))
        st.plotly_chart(scatter_fig)  # Mostrar directamente
    df.drop(columns=['cluster_label'], inplace=True)
"""
def show_cluster_correlation_heatmap(df, cluster_id):
    cluster_df = df[df['cluster'] == cluster_id][["n_visitas", "monto_compras", "monto_descuentos"]]
    corr_matrix = cluster_df.corr()
    # Create the figure explicitly
    fig, ax = plt.subplots(figsize=(5, 4))
    # Plot the heatmap on the created figure
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5, ax=ax)
    # Use st.pyplot() and pass the figure
    st.pyplot(fig)
"""


    
