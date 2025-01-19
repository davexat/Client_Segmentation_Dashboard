import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

from dashboard.config import CLUSTER_COLORS as colors
from dashboard.config import CUSTOMER_TYPES as customers
from dashboard.config import DEFAULT_BACKGROUND_COLOR as defcolor
from dashboard.config import configure_pie_chart_figure, configure_scatter_figure, configure_figure

from dashboard.layout import create_container, create_header, format_large_numbers

from customer_segmentation.plot import create_boxplot_figure
from customer_segmentation.plot import create_density_figure
from customer_segmentation.plot import create_scatter_figure

def show_cluster_analysis(df):
    st.header("Cluster Analysis")
    col1, col2 = st.columns(2)
    with col1:
        with create_container("slider", color = defcolor, padding = 10):
            header_placeholder = st.empty()
            cluster_number = show_slider(df)
            with header_placeholder:
                show_cluster_header(cluster_number)
        gb = df[df["cluster"] == cluster_number]
        show_metrics_for_cluster(df, gb, cluster_number)
        with create_container("scatter_cont", color = defcolor, padding = 10):
            show_scatter_container(gb)
    with col2:
        with create_container("column2", color = defcolor, padding = 10):
            show_density_and_boxplot(df, gb, colors[customers[cluster_number]])

def show_density_and_boxplot(df, gb, cluster_color, color = defcolor):
     with create_container("density_and_boxplot", color = color):
            selected_var = show_variable_selector(df)
            show_density_diagram(gb, selected_var, cluster_color)
            show_boxplot_diagram(gb, selected_var, cluster_color)

def show_scatter_container(gb):
    with create_container("scatters", color = defcolor):
                selected_vars = show_variables_selector(gb)
                show_scatter(gb, selected_vars)

def show_slider(df):
    st.markdown("""<style> .stSlider { max-width: 90%; margin: auto; } </style> """, unsafe_allow_html=True)
    cluster_number = st.slider(
        label="Select a Cluster:", 
        min_value=0, 
        max_value= len(df.groupby("cluster")) - 1,
        step=1, 
        value=0,
        format="Cluster %d"
    )
    return cluster_number

def show_cluster_header(cluster_number):
    st.subheader(f"Cluster {cluster_number}: {customers[cluster_number]}")

def show_variables_selector(df):
    variables = list(combinations(df.columns[:-1], 2))
    return st.selectbox("Select variables to display:", variables)

def show_variable_selector(df):
    variables = list(df.columns)[:-1]
    return st.selectbox("Select a variable to display:", variables)

def show_metrics_for_cluster(df, gb, cluster_number, key = "metrics", bg_color=defcolor, format=True):
    columns=list(df.columns)
    totals = gb["cluster"].count(), gb["n_visitas"].sum(), gb["monto_compras"].sum(), gb["monto_descuentos"].sum()
    totals = tuple(format_large_numbers(totals[i]) if i != 0 else totals[i] for i in range(len(totals)))
        
    metrics_html = f"""
    <div style="padding: 0px; margin: 0 0 15px 0;">
        <div style="display: flex; justify-content: space-around;">
            <div style="flex: 1;">{create_header(f"Total clients", totals[0], "white")}</div>
            {"".join(f'<div style="flex: 1;">{create_header(f"{columns[i]}", totals[i+1], colors[customers[cluster_number]])}</div>' for i in range(len(customers)))}
        </div>
    </div>
    """
    with create_container(key, color = bg_color):
        st.markdown(metrics_html, unsafe_allow_html=True)

def show_scatter(df, selected_var, cluster_labels = customers, cluster_colors = colors, color = defcolor):
    var_x, var_y = selected_var
    scatter_fig = create_scatter_figure(df, var_x, var_y, cluster_labels, cluster_colors)
    configured_scatter = configure_scatter_figure(scatter_fig, color)
    st.plotly_chart(configured_scatter)

def show_correlation_heatmap(df, color = defcolor):
    correlation_heatmap = create_correlation_heatmap(df, color)
    configured_heatmap = configure_heatmap_appearance(correlation_heatmap, color)
    st.pyplot(configured_heatmap, use_container_width=True)

def show_density_diagram(df, selected_var, cluster_color, color = defcolor):
    hist_fig = create_density_figure(df, selected_var, cluster_color)
    configured_density = configure_figure(hist_fig, color)
    st.plotly_chart(configured_density)

def show_boxplot_diagram(df, selected_var, cluster_color, color = defcolor):
    boxp_fig = create_boxplot_figure(df,selected_var, cluster_color)
    condigured_boxp = configure_figure(boxp_fig, color)
    st.plotly_chart(condigured_boxp)



    
