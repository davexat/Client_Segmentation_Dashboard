import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

from dashboard.config import CLUSTER_COLORS as colors
from dashboard.config import CUSTOMER_TYPES as customers
from dashboard.config import DEFAULT_BACKGROUND_COLOR as defcolor
from dashboard.layout import create_container, create_header, format_large_numbers


def show_cluster_analysis(df):
    st.header("Cluster Analysis")
    col1, col2 = st.columns([5,5])
    with col1:
        header_placeholder = st.empty()
        cluster_number = show_slider(df)
        with header_placeholder:
            show_cluster_header(cluster_number)
        gb = df[df["cluster"] == cluster_number]
        show_metrics_for_cluster(df, gb)
    with col2:
        with create_container("scatters", color = defcolor):
            selected_vars = show_variables_selector(gb)
            show_scatter(gb, selected_vars)
        with create_container("graphs_2", color = defcolor):
            selected_var = show_variable_selector(df)
            show_density_diagram(gb, selected_var, colors[customers[cluster_number]])

def show_slider(df):
    st.markdown("""<style> .stSlider { max-width: 95%; margin: auto; } </style> """, unsafe_allow_html=True)
    with create_container("slider", defcolor, 10):
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

### FUNCIONES DE MUESTRA

def show_metrics_for_cluster(df, gb, key = "metrics", bg_color=defcolor, format=True):
    columns=list(df.columns)
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
    configured_density = configure_density_figure(hist_fig, color)
    st.plotly_chart(configured_density)

### FUNCIONES DE CONFIGURACION

def configure_scatter_figure(fig, color, height=300, marker_size=8, marker_opacity=0.8):
    fig.update_layout(
        plot_bgcolor=color,
        paper_bgcolor=color,
        height=height,
        margin=dict(l=50, r=50, t=70, b=70),
        title=dict(font=dict(size=18, color="white"), xanchor="center", x=0.5, y=0.94)
    )
    fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity), showlegend=False)
    return fig

# def configure_heatmap_appearance(plt, color):
#     plt.xticks(fontsize=8, color='white')
#     plt.yticks(fontsize=8, color='white')
#     plt.title("Correlation Heatmap", fontsize=8, weight="bold", color="white")
#     plt.gca().set_facecolor(color)
#     return plt

def configure_heatmap_appearance(fig, color):
    ax = fig.gca()
    ax.tick_params(axis='both', labelsize=8, colors='white')
    ax.set_title("Correlation Heatmap", fontsize=8, weight="bold", color="white")
    ax.set_facecolor(color)
    return fig

def configure_density_figure(fig, color, height=300):
    fig.update_layout(
        plot_bgcolor=color,
        paper_bgcolor=color,
        height=height,
        title=dict(xanchor="center", x=0.5)
    )
    fig.update_traces(opacity=1)
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

def create_correlation_heatmap(df, color):
    df_numeric = df.drop(columns=['cluster'], errors='ignore')
    corr_matrix = df_numeric.corr()
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=color)
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="coolwarm", 
        linewidths=0.5, 
        linecolor='black', 
        square=True,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 10, "weight": "bold", "color": "black"},
        vmin=-1, vmax=1,
        xticklabels=corr_matrix.columns, 
        yticklabels=corr_matrix.columns,
        ax=ax
    )
    return fig

def create_density_figure(df, selected_var, cluster_color):
    fig = px.histogram(
        df,
        x=selected_var,
        histnorm="density",
        title=f"Density Plot of {selected_var}",
        labels={selected_var: selected_var},
        template="plotly_dark"
    ).update_layout(showlegend=False).update_traces(marker_color=cluster_color)
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


    
