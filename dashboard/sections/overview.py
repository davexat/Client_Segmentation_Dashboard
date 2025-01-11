import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dashboard.config import CLUSTER_COLORS as colors
from dashboard.config import CUSTOMER_TYPES as customers
from dashboard.sections.cluster_analysis import show_cluster_analysis
from dashboard.layout import create_container

height = 500
def show_overview(df):
    st.header("Overview")
    col1, col2 = st.columns(2)
    with col1:
        show_clients_cluster(df)
    with col2:
        show_sales_cluster(df)

    col3, col4 = st.columns([6,4])
    with col3:
        show_cluster_analysis(df)
    with col4:
        show_selector_graph(df, "#1B1D22")

def show_selector_graph(df, bg_color):
    with create_container("graphs"):
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            selected_var = show_variable_selector()
        with col2_2:
            selected_graph = show_graph_selector()
        if selected_graph == "Bars":
            show_bar_chart(df, selected_var, bg_color)
        elif selected_graph == "Histogram":
            show_histogram(df, selected_var, bg_color)
        elif selected_graph == "Boxplot":
            show_boxplot(df, selected_var, bg_color)

def show_clients_cluster(df, key="clients"):
    gb = df.groupby('cluster').size()
    show_metric_cluster(gb, " clients", key, bg_color="#1B1D22", format=False)

def show_visits_cluster(df, key="visits"):
    gb = df.groupby('cluster')["n_visitas"].sum()
    show_metric_cluster(gb, " visits", key, bg_color="#1B1D22")

def show_sales_cluster(df, key="sales"):
    gb = df.groupby('cluster')["monto_compras"].sum()
    show_metric_cluster(gb, " sales", key, bg_color="#1B1D22")

def show_discounts_cluster(df, key="discounts"):
    gb = df.groupby('cluster')["monto_descuentos"].sum()
    show_metric_cluster(gb, " discounts", key, bg_color="#1B1D22")

def show_metric_cluster(gb, text, key, bg_color="black", format=True):
    total, total_0, total_1, total_2 = gb.sum(), gb.loc[0], gb.loc[1], gb.loc[2]
    if format:
        total = format_large_numbers(total)
        total_0 = format_large_numbers(total_0)
        total_1 = format_large_numbers(total_1)
        total_2 = format_large_numbers(total_2)
    metrics_html = f"""
    <div style="padding: 0px; margin: 0 0 15px 0;">
        <div style="display: flex; justify-content: space-around;">
            <div style="flex: 1;">{create_header(f"Total{text}", total, "white")}</div>
            <div style="flex: 1;">{create_header(f"{customers[0]}{text}", total_0, colors[customers[0]])}</div>
            <div style="flex: 1;">{create_header(f"{customers[1]}{text}", total_1, colors[customers[1]])}</div>
            <div style="flex: 1;">{create_header(f"{customers[2]}{text}", total_2, colors[customers[2]])}</div>
        </div>
    </div>
    """
    with create_container(key, bg_color):
        st.markdown(metrics_html, unsafe_allow_html=True)

def create_header(title, value, txt_color):
    style = '''
    <div style="text-align: center; color: {txt_color}; padding: 0;">
        <h6 style="margin-bottom: 0; padding: 15px 0 0 0;">{title}</h6>
        <h1 style="margin-top: 0; padding: 0 0 10px 0;">{value}</h1>
    </div>
    '''
    return style.format(title=title, value=value, txt_color=txt_color)

def show_variable_selector():
    variables = ["n_visitas", "monto_compras", "monto_descuentos"]
    return st.selectbox("Select a variable to display:", variables)

def show_graph_selector():
    variables = ["Bars", "Histogram", "Boxplot"]
    return st.selectbox("Select a graph to display:", variables)

def show_bar_chart(df, selected_var, color):
    data = {
        "Customer Type": customers,
        "Mean": [
            df[df['cluster'] == 0][selected_var].mean().astype(int),
            df[df['cluster'] == 1][selected_var].mean().astype(int),
            df[df['cluster'] == 2][selected_var].mean().astype(int)
        ]
    }
    df_plot = pd.DataFrame(data)
    fig = px.bar(
        df_plot, x="Customer Type", y="Mean", text="Mean",
        title=f"Mean {selected_var} by Customer Type",
        labels={"Customer Type": "Customer Type", "Mean": f"Mean {selected_var}"},
        template="plotly_dark"
    )
    fig.update_traces(
        texttemplate='%{text}', 
        textposition='outside', 
        marker=dict(color=[colors[customer] for customer in customers])
    )
    fig.update_layout(height=height, paper_bgcolor=color)
    st.plotly_chart(fig)

def show_histogram(df, selected_var):
    hist_fig = px.histogram(
        df, x=selected_var, color='cluster', barmode='overlay',
        title=f"Distribution of {selected_var} by Customer Type",
        labels={selected_var: selected_var, "cluster": "Customer Type"},
        template="plotly_dark",
        category_orders={"cluster": [0, 1, 2]},
        color_discrete_map={i: colors[customers[i]] for i in range(len(customers))}
    )
    hist_fig.update_layout(
        paper_bgcolor=color,
        height=height,
        legend=dict(font=dict(size=15), x=0.75, y=1.2),
        title=dict(xanchor="center", x=0.5)
    )
    hist_fig.update_traces(opacity=1)
    hist_fig.for_each_trace(lambda t: t.update(name=customers[int(t.name)]))
    st.plotly_chart(hist_fig)

def show_boxplot(df, selected_var):
    boxplot_fig = px.box(
        df, x=selected_var, y='cluster', color='cluster', orientation='h',
        title=f"Boxplot of {selected_var} by Customer Type",
        labels={selected_var: selected_var, "cluster": "Customer Type"},
        category_orders={"cluster": list(range(len(customers)))},
        color_discrete_map={i: colors[customers[i]] for i in range(len(customers))},
        hover_data={'cluster': True, 'n_visitas': True, 'monto_compras': True, 'monto_descuentos': True}
    )
    boxplot_fig.update_yaxes(
        tickvals=list(range(len(customers))),
        ticktext=customers
    )
    boxplot_fig.update_layout(
        paper_bgcolor=color,
        height=height,
        legend=dict(font=dict(size=15), x=0.75, y=1.2)
    )
    boxplot_fig.for_each_trace(lambda t: t.update(name=customers[int(t.name)]))
    st.plotly_chart(boxplot_fig)

def format_large_numbers(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:.1f}"