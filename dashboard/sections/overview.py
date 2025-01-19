import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dashboard.config import CLUSTER_COLORS as colors
from dashboard.config import CUSTOMER_TYPES as customers
from dashboard.config import DEFAULT_BACKGROUND_COLOR as defcolor
from dashboard.layout import create_container, create_header, format_large_numbers

#from customer_segmentation.plot import create_boxplot_figure
#from customer_segmentation.plot import create_histogram_figure
#from customer_segmentation.plot import create_bar_chart_figure

height = 532
def show_overview(df):
    st.header("Overview")
    col1, col2 = st.columns(2)
    with col1:
        show_clients_cluster(df)
    with col2:
        show_scores_cluster(df)

    col3, col4 = st.columns([6,4])
    with col3:
        show_pie_charts(df)
    with col4:
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
    show_metric_cluster(gb, " score", key, bg_color=defcolor)

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
    bar_chart_fig = create_bar_chart_figure(df, selected_var, cluster_labels, cluster_colors)
    return configure_bar_chart_figure(bar_chart_fig, color, height)

def show_histogram(df, selected_var, cluster_labels = customers, cluster_colors = colors, color=defcolor, height=400):
    hist_fig = create_histogram_figure(df, selected_var, cluster_labels, cluster_colors)
    return configure_histogram_figure(hist_fig, cluster_labels, color, height)

def show_boxplot(df, selected_var, cluster_labels = customers, cluster_colors = colors, color=defcolor, height=400):
    boxplot_fig = create_boxplot_figure(df, selected_var, cluster_labels, cluster_colors)
    return configure_boxplot_figure(boxplot_fig, cluster_labels, color, height)

# def show_pie_charts(df, cluster_labels = customers, cluster_colors = colors, color = defcolor):
#     with create_container("piecharts", color):
#         col1, col2, col3, col4 = st.columns(4)
#         variables = ["cluster", "n_visitas", "monto_compras", "monto_descuentos"]
#         columns = [col1, col2, col3, col4]
#         for var, col in zip(variables, columns):
#             with col:
#                 pie_chart = create_pie_chart_figure(df, var, cluster_labels, cluster_colors)
#                 configured_pie_chart = configure_pie_chart_figure(pie_chart, color)
#                 st.plotly_chart(configured_pie_chart)

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

#### METODOS DE CONFIGURACION

def configure_boxplot_figure(fig, cluster_labels, color, height=400, legend_position=(0.75, 1.2)):
    fig.update_yaxes(
        tickvals=list(range(len(cluster_labels))),
        ticktext=cluster_labels
    )
    fig.update_layout(
        plot_bgcolor=color,
        paper_bgcolor=color,
        height=height,
        legend=dict(font=dict(size=15), x=legend_position[0], y=legend_position[1]),
        title=dict(xanchor="center", x=0.5)
    )
    fig.for_each_trace(lambda t: t.update(name=cluster_labels[int(t.name)]))
    return fig

def configure_histogram_figure(fig, cluster_labels, color, height=400, legend_position=(0.75, 1.2)):
    fig.update_layout(
        plot_bgcolor=color,
        paper_bgcolor=color,
        height=height,
        legend=dict(font=dict(size=15), x=legend_position[0], y=legend_position[1]),
        title=dict(xanchor="center", x=0.5)
    )
    fig.update_traces(opacity=1)
    fig.for_each_trace(lambda t: t.update(name=cluster_labels[int(t.name)]))
    return fig

def configure_bar_chart_figure(fig, color, height=400): 
    fig.update_layout(
        plot_bgcolor=color,
        paper_bgcolor=color,
        height=height,
        title=dict(xanchor="center", x=0.5)
    )
    return fig

# def configure_pie_chart_figure(fig, color, height=400):
#     fig.update_layout(
#         paper_bgcolor=color,
#         margin=dict(l=30, r=30, t=0, b=0),
#         title=dict(font=dict(size=15, color="white"), xanchor="center", x=0.5, y=0.9),
#         legend=dict(font=dict(size=15, color="white"), xanchor="center", x=0.5, yanchor="bottom", y=0.025),
#         height=height
#     )
#     return fig

def configure_pie_chart_figure(fig, color, height=300):
    fig.update_layout(
        paper_bgcolor=color,
        margin=dict(l=0, r=50, t=50, b=10),
        title=dict(font=dict(size=15, color="white"), xanchor="center", x=0.5, y=0.92),
        legend=dict(font=dict(size=15, color="white"), xanchor="center", x=0.92, yanchor="middle", y=0.5),
        height=height
    )
    fig.update_traces(textfont=dict(size=20))
    return fig

#### ESTO NO DEBERÍA IR AQUÍ

def create_bar_chart_figure(df, selected_var, cluster_labels, cluster_colors):
    data = {
        "Customer Type": cluster_labels,
        "Mean": [
            df[df['cluster'] == i][selected_var].mean().astype(int)
            for i in range(len(cluster_labels))
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
        marker=dict(color=[cluster_colors[label] for label in cluster_labels])
    )
    return fig

def create_boxplot_figure(df, selected_var, cluster_labels, cluster_colors):
    return px.box(
        df,
        x=selected_var,
        y='cluster',
        color='cluster',
        orientation='h',
        title=f"Boxplot of {selected_var} by Customer Type",
        labels={selected_var: selected_var, "cluster": "Customer Type"},
        category_orders={"cluster": list(range(len(cluster_labels)))},
        color_discrete_map={i: cluster_colors[cluster_labels[i]] for i in range(len(cluster_labels))},
        hover_data={'cluster': True, 'n_visitas': True, 'monto_compras': True, 'monto_descuentos': True}
    )

def create_histogram_figure(df, selected_var, cluster_labels, cluster_colors):
    return px.histogram(
        df,
        x=selected_var,
        color='cluster',
        barmode='overlay',
        title=f"Distribution of {selected_var} by Customer Type",
        labels={selected_var: selected_var, "cluster": "Customer Type"},
        template="plotly_dark",
        category_orders={"cluster": list(range(len(cluster_labels)))},
        color_discrete_map={i: cluster_colors[cluster_labels[i]] for i in range(len(cluster_labels))}
    )

def create_pie_chart_figure(df, selected_var, cluster_labels, cluster_colors, template="plotly_dark"):
    if selected_var == "cluster":
        data = df['cluster'].value_counts()
    else:
        data = df.groupby('cluster')[selected_var].sum()
    
    fig = px.pie(
        names=cluster_labels,
        values=data.values.tolist(),
        color=cluster_labels,
        color_discrete_map=cluster_colors,
        title=f"Distribution of {selected_var}",
        template=template
    )
    return fig