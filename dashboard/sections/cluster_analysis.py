import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dashboard.config import CLUSTER_COLORS as colors
from dashboard.config import CUSTOMER_TYPES as customers
from dashboard.config import DEFAULT_BACKGROUND_COLOR as defcolor
from dashboard.layout import create_container, create_header, format_large_numbers

def show_cluster_analysis(df):
    st.header("Cluster Analysis")
    set_slider()
    with create_container("slider", defcolor, 10):
        cluster_number = st.slider(
            label="Select a Cluster:", 
            min_value=0, 
            max_value= len(df.groupby("cluster")) - 1,
            step=1, 
            value=0,
            format="Cluster %d"
        )
    st.subheader(f"Cluster {cluster_number}: {customers[cluster_number]}")
    # show_cluster_info(df, cluster_number)
    show_metric_cluster(df, "metrics_cluster", cluster_number)
    #show_scatter(df)

def set_slider():
    st.markdown("""<style> .stSlider { max-width: 95%; margin: auto; } </style> """, unsafe_allow_html=True)

### FUNCIONES DE MUESTRA

def show_metric_cluster(df, key, cluster_number, bg_color=defcolor, format=True):
    gb = df[df["cluster"] == cluster_number]
    columns =list(df.columns)
    totals = gb["cluster"].count(), gb["n_visitas"].sum(), gb["monto_compras"].sum(), gb["monto_descuentos"].sum()
    totals = tuple(format_large_numbers(totals[i]) if i != 0 else totals[i] for i in range(len(totals)))
        
    metrics_html = f"""
    <div style="padding: 0px; margin: 0 0 15px 0;">
        <div style="display: flex; justify-content: space-around;">
            <div style="flex: 1;">{create_header(f"Total clients", totals[0], "white")}</div>
            {"".join(f'<div style="flex: 1;">{create_header(f"{columns[i]}", totals[i+1], colors[customers[i]])}</div>' for i in range(len(customers)))}
        </div>
    </div>
    """
    with create_container(key, color = bg_color):
        st.markdown(metrics_html, unsafe_allow_html=True)

def show_scatter(df, cluster_labels = customers, cluster_colors = colors, color = defcolor):
    scatter_vars = [(df.columns[0], df.columns[1]), (df.columns[0], df.columns[2]), (df.columns[1], df.columns[2])]
    for var_x, var_y in scatter_vars:
        scatter_fig = create_scatter_figure(df, var_x, var_y, cluster_labels, cluster_colors)
        configured_scatter = configure_scatter_figure(scatter_fig, color)
        st.plotly_chart(configured_scatter)

### FUNCIONES DE CONFIGURACION

def configure_scatter_figure(fig, color, marker_size=8, marker_opacity=0.8):
    fig.update_layout(paper_bgcolor = color)
    fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))
    return fig

### ESTO NO DEBERÍA IR AQUÍ

def create_scatter_figure(df, var_x, var_y, cluster_labels, cluster_colors, template="plotly_dark"):
    df['cluster_label'] = df['cluster'].map({i: cluster_labels[i] for i in range(len(cluster_labels))})
    fig = px.scatter(
        df, x=var_x, y=var_y, color='cluster_label',
        title=f"Scatter: {var_x} vs {var_y}",
        template=template,
        color_discrete_map=cluster_colors
    )
    df.drop(columns=['cluster_label'], inplace=True)  # Clean up temporary column
    return fig

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


    
