import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dashboard.config import CLUSTER_COLORS as colors
from dashboard.sections.cluster_analysis import show_cluster_analysis

# Función para mostrar la visión general
def show_overview(df):
    st.header("Overview")

    customer_types = ["High Spenders", "Moderate Engagers", "Active Savers"]

    col1, col2 = st.columns([7,3])
    with col1:
        show_clients_cluster(df)
        show_cluster_analysis(df)
    with col2:
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            selected_var = show_variable_selector()
        with col2_2:
            selected_graph = show_graph_selector()
        if selected_graph == "Bars":
            show_bar_chart(df, selected_var, customer_types)
        elif selected_graph == "Histogram":
            show_histogram(df, selected_var, customer_types)
        elif selected_graph == "Boxplot":
            show_boxplot(df, selected_var, customer_types)

def show_clients_cluster(df):
    gb = df.groupby('cluster').size()
    total, total_0, total_1, total_2 = gb.sum(), gb.loc[0], gb.loc[1], gb.loc[2]
    col1_1, col1_2, col1_3, col1_4 = st.columns(4)
    with col1_1:
        st.markdown(create_header("Total customers", total, "gray"), unsafe_allow_html=True)
    with col1_2:
        st.markdown(create_header("High Spenders", total_0, colors["High Spenders"]), unsafe_allow_html=True)
    with col1_3:
        st.markdown(create_header("Moderate Engagers", total_1, colors["Moderate Engagers"]), unsafe_allow_html=True)
    with col1_4:
        st.markdown(create_header("Active Savers", total_2, colors["Active Savers"]), unsafe_allow_html=True)

# Estilo para mostrar los títulos
def create_header(title, value, color):
    style = '''
    <div style="text-align: center; margin: 0; color: {color};">
        <h6 style="margin-bottom: 0;">{title}</h6>
        <h1 style="margin-top: 0;">{value}</h1>
    </div>
    '''
    return style.format(title=title, value=value, color=color)

# Función para seleccionar la variable para el gráfico
def show_variable_selector():
    variables = ["n_visitas", "monto_compras", "monto_descuentos"]
    return st.selectbox("Select a variable to display:", variables)

def show_graph_selector():
    variables = ["Bars", "Histogram", "Boxplot"]
    return st.selectbox("Select a graph to display:", variables)

# Función para mostrar el gráfico de barras
def show_bar_chart(df, selected_var, customer_types):
    data = {
        "Customer Type": customer_types,
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
        marker=dict(color=[colors["High Spenders"], colors["Moderate Engagers"], colors["Active Savers"]])
    )
    fig.update_layout(
        height=550
    )
    st.plotly_chart(fig)

# Función para mostrar el gráfico de histograma
def show_histogram(df, selected_var, customer_types):
    hist_fig = px.histogram(
        df, x=selected_var, color='cluster', barmode='overlay',
        title=f"Distribution of {selected_var} by Customer Type",
        labels={selected_var: selected_var, "cluster": "Customer Type"},
        template="plotly_dark",
        category_orders={"cluster": [0, 1, 2]},
        color_discrete_map={0: colors["High Spenders"], 1: colors["Moderate Engagers"], 2: colors["Active Savers"]}
    )
    hist_fig.update_layout(
        height=550,
        legend=dict(
            font=dict(size=15),              # Leyenda más compacta
            x=0.75,                           # Posición horizontal (centrada)
            y=1
        )
    )
    hist_fig.update_traces(opacity=1)
    hist_fig.for_each_trace(lambda t: t.update(name=customer_types[int(t.name)]))
    st.plotly_chart(hist_fig)

# Función para mostrar el gráfico de boxplot
def show_boxplot(df, selected_var, customer_types):
    boxplot_fig = px.box(
        df, x=selected_var, y='cluster', color='cluster', orientation='h',
        title=f"Boxplot of {selected_var} by Customer Type",
        labels={selected_var: selected_var, "cluster": "Customer Type"},
        category_orders={"cluster": [0, 1, 2]},
        color_discrete_map={0: colors["High Spenders"], 1: colors["Moderate Engagers"], 2: colors["Active Savers"]},
        hover_data={'cluster': True, 'n_visitas': True, 'monto_compras': True, 'monto_descuentos': True}
    )
    boxplot_fig.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=["0", "1", "2"]
    )
    boxplot_fig.update_layout(
        height=550,
        legend=dict(
            font=dict(size=15),              # Leyenda más compacta
            x=0.3,                           # Posición horizontal (centrada)
            y=-0.5                         # Posición vertical (debajo del gráfico)
                                # Ancla vertical (tope)
        )
    )
    boxplot_fig.for_each_trace(lambda t: t.update(name=customer_types[int(t.name)]))
    st.plotly_chart(boxplot_fig)