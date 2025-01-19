import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dashboard.config import CLUSTER_COLORS as colors
from dashboard.config import CUSTOMER_TYPES as customers
from dashboard.config import DEFAULT_BACKGROUND_COLOR as defcolor
from dashboard.config import configure_boxplot_figure, configure_histogram_figure, configure_bar_chart_figure, configure_pie_chart_figure

from dashboard.layout import create_container, create_header, format_large_numbers

from customer_segmentation.plot import create_all_boxplot_figure
from customer_segmentation.plot import create_all_histogram_figure
from customer_segmentation.plot import create_all_bar_chart_figure
from customer_segmentation.plot import create_pie_chart_figure

height = 532

def show_overview(df):
    st.header("Overview")
    col1, col2 = st.columns(2)
    with col1:
        show_clients_cluster(df)
    with col2:
        show_scores_cluster(df)

    col3, col4 = st.columns([5,5])
    with col3:
        with create_container("piecharts_cont", color = defcolor, padding = 10):
            show_pie_charts(df)
    with col4:
        with create_container ("plots_cont", color = defcolor, padding = 10):
            show_selector_graph(df)

def show_selector_graph(df, bg_color = defcolor):
    with create_container("graphs", color = bg_color):
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            selected_var = show_variable_selector()
        with col2_2:
            selected_graph = show_graph_selector()
        if selected_graph == "Bars":
            st.plotly_chart(show_bar_chart(df, selected_var, color = bg_color, height = height))
        elif selected_graph == "Histogram":
            st.plotly_chart(show_histogram(df, selected_var, color = bg_color, height = height))
        elif selected_graph == "Boxplot":
            st.plotly_chart(show_boxplot(df, selected_var, color = bg_color, height = height))

def show_clients_cluster(df, key="clients"):
    gb = df.groupby('cluster').size()
    show_metric_cluster(gb, " clients", key, bg_color=defcolor, format=False)

def show_visits_cluster(df, key="visits"):
    gb = df.groupby('cluster')["n_visitas"].sum()
    show_metric_cluster(gb, " visits", key, bg_color=defcolor)

def show_sales_cluster(df, key="sales"):
    gb = df.groupby('cluster')["monto_compras"].sum()
    show_metric_cluster(gb, " sales", key, bg_color=defcolor)

def show_scores_cluster(df, key="scores"):
    total_sales = df.groupby('cluster')["monto_compras"].sum()
    total_visits = df.groupby('cluster')["n_visitas"].sum()
    gb = (total_sales / total_visits).fillna(0)
    show_metric_cluster(gb, " S/V", key, bg_color=defcolor)

def show_discounts_cluster(df, key="discounts"):
    gb = df.groupby('cluster')["monto_descuentos"].sum()
    show_metric_cluster(gb, " discounts", key, bg_color=defcolor)

def show_metric_cluster(gb, text, key, bg_color=defcolor, format=True, percent=False):
    totals = gb.sum(), gb.loc[0], gb.loc[1], gb.loc[2]
    if format:
        totals = tuple(format_large_numbers(total) for total in totals)
    metrics_html = f"""
    <div style="padding: 0px; margin: 0 0 15px 0;">
        <div style="display: flex; justify-content: space-around;">
            <div style="flex: 1;">{create_header(f"Total{text}", totals[0], "white")}</div>
            {"".join(f'<div style="flex: 1;">{create_header(f"{customers[i]}{text}", totals[i+1], colors[customers[i]])}</div>' for i in range(len(customers)))}
        </div>
    </div>
    """
    with create_container(key, color = bg_color):
        st.markdown(metrics_html, unsafe_allow_html=True)

def show_variable_selector():
    variables = ["n_visitas", "monto_compras", "monto_descuentos"]
    return st.selectbox("Select a variable to display:", variables)

def show_graph_selector():
    variables = ["Bars", "Histogram", "Boxplot"]
    return st.selectbox("Select a graph to display:", variables)

def show_bar_chart(df, selected_var, cluster_labels = customers, cluster_colors = colors, color=defcolor, height=400):
    bar_chart_fig = create_all_bar_chart_figure(df, selected_var, cluster_labels, cluster_colors)
    return configure_bar_chart_figure(bar_chart_fig, color, height)

def show_histogram(df, selected_var, cluster_labels = customers, cluster_colors = colors, color=defcolor, height=400):
    hist_fig = create_all_histogram_figure(df, selected_var, cluster_labels, cluster_colors)
    return configure_histogram_figure(hist_fig, cluster_labels, color, height)

def show_boxplot(df, selected_var, cluster_labels = customers, cluster_colors = colors, color=defcolor, height=400):
    boxplot_fig = create_all_boxplot_figure(df, selected_var, cluster_labels, cluster_colors)
    return configure_boxplot_figure(boxplot_fig, cluster_labels, color, height)

def show_pie_charts(df, cluster_labels = customers, cluster_colors = colors, color = defcolor):
    with create_container("piecharts", color):
        col1, col2 = st.columns(2)
        variables = ["cluster", "n_visitas", "monto_compras", "monto_descuentos"]
        columns = [col1, col1, col2, col2]
        for var, col in zip(variables, columns):
            with col:
                pie_chart = create_pie_chart_figure(df, var, cluster_labels, cluster_colors)
                configured_pie_chart = configure_pie_chart_figure(pie_chart, color)
                st.plotly_chart(configured_pie_chart)