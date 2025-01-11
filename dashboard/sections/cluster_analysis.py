import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dashboard.config import CLUSTER_COLORS as colors
from dashboard.config import CUSTOMER_TYPES as customers
from dashboard.config import DEFAULT_BACKGROUND_COLOR as defcolor
from dashboard.layout import create_container

def show_cluster_analysis(df):
    show_pie_charts(df)
    plot_scatter_plots(df)

def show_pie_charts(df, color=defcolor):
    with create_container("piecharts", defcolor):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.plotly_chart(plot_pie_chart(df, "cluster", color))
        with col2:
            st.plotly_chart(plot_pie_chart(df, "n_visitas", color))
        with col3:
            st.plotly_chart(plot_pie_chart(df, "monto_compras", color))
        with col4:
            st.plotly_chart(plot_pie_chart(df, "monto_descuentos", color))

def plot_pie_chart(df, column, color = defcolor):
    data = df.groupby('cluster')[column].sum() if column != "cluster" else df['cluster'].value_counts()
    fig = px.pie(
        names=customers,
        values=data.values.tolist(),
        color=customers,
        color_discrete_map=colors,
        title=f"Distribution of {column}",
        template="plotly_dark"
    )
    fig.update_layout(
        paper_bgcolor=color,
        margin=dict(l=30, r=30, t=0, b=0),
        title=dict(font=dict(size=15, color="white"), xanchor="center", x=0.5, y=0.9),
        legend=dict(font=dict(size=15, color="white"), xanchor="center", x=0.5, yanchor="bottom", y=0.025),
        height=400
    )
    return fig

def plot_scatter_plots(df):
    scatter_vars = [("n_visitas", "monto_compras"), ("n_visitas", "monto_descuentos"), ("monto_compras", "monto_descuentos")]
    df['cluster_label'] = df['cluster'].map({i: customers[i] for i in range(len(customers))})
    for var_x, var_y in scatter_vars:
        scatter_fig = px.scatter(
            df, x=var_x, y=var_y, color='cluster_label',
            title=f"Scatter: {var_x} vs {var_y}",
            template="plotly_dark",
            color_discrete_map=colors
        )
        scatter_fig.update_traces(marker=dict(size=8, opacity=0.8))
        st.plotly_chart(scatter_fig)
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


    
