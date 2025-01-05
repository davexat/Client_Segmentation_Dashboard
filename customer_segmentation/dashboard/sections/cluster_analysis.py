import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from customer_segmentation.dashboard.config import CLUSTER_COLORS as colors

def show_cluster_analysis(df):
    st.header("Cluster Analysis")

    # Subsecciones para visualización de distribuciones
    st.subheader("Distributions by Cluster")
    st.plotly_chart(plot_pie_chart(df, "cluster"))
    st.plotly_chart(plot_pie_chart(df, "n_visitas"))
    st.plotly_chart(plot_pie_chart(df, "monto_compras"))
    st.plotly_chart(plot_pie_chart(df, "monto_descuentos"))

    # Subsección para gráficos de dispersión
    st.subheader("Scatter Plots Between Variables")
    scatter_figs = plot_scatter_plots(df)
    for scatter_fig in scatter_figs:
        st.plotly_chart(scatter_fig)

    #show_cluster_correlation_heatmap(df, 0)

def plot_pie_chart(df, column):
    data = df.groupby('cluster')[column].sum() if column != "cluster" else df['cluster'].value_counts()
    labels = ["High Spenders", "Moderate Engagers", "Active Savers"]
    fig = px.pie(
        names=labels,
        values=data.values.tolist(),
        color=labels,
        color_discrete_map=colors,
        title=f"Distribution of {column.capitalize()} by Cluster",
        template="plotly_dark"
    )
    return fig

def plot_scatter_plots(df):
    scatter_vars = [("n_visitas", "monto_compras"), ("n_visitas", "monto_descuentos"), ("monto_compras", "monto_descuentos")]
    df['cluster_label'] = df['cluster'].map({0: "High Spenders", 1: "Moderate Engagers", 2: "Active Savers"})
    scatter_figs = []
    for var_x, var_y in scatter_vars:
        scatter_fig = px.scatter(
            df, x=var_x, y=var_y, color='cluster_label',
            title=f"Scatter: {var_x} vs {var_y}",
            template="plotly_dark",
            color_discrete_map=colors
        )
        scatter_fig.update_traces(marker=dict(size=8, opacity=0.8))
        scatter_figs.append(scatter_fig)
    df.drop(columns=['cluster_label'], inplace=True)
    return scatter_figs
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


    
