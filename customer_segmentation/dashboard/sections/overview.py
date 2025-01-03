import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from customer_segmentation.dashboard.data import load_data

# Estilo para mostrar los títulos
def create_header(title, value, color):
    style = '''
    <div style="text-align: center; margin: 0; color: {color};">
        <h6 style="margin-bottom: 0;">{title}</h6>
        <h1 style="margin-top: 0;">{value}</h1>
    </div>
    '''
    return style.format(title=title, value=value, color=color)

# Función para mostrar la visión general
def show_overview():
    st.header("Overview")
    df = load_data()

    # Colores para los clusters
    colors = {"Total": "white", "High Spenders": "purple", "Moderate Engagers": "lightblue", "Active Savers": "yellow"}
    customer_types = ["High Spenders", "Moderate Engagers", "Active Savers"]
    
    # Agrupar y contar clientes
    gb = df.groupby('cluster').size()
    total, total_0, total_1, total_2 = gb.sum(), gb.loc[0], gb.loc[1], gb.loc[2]

    # Disposición de las columnas
    col1, col2 = st.columns([3, 1])  # Columna 2/3 para Overview y 1/3 para los Scatter plots
    with col1:
        # Mostrar los datos de clientes
        col1_1, col1_2, col1_3, col1_4 = st.columns(4)
        with col1_1:
            st.markdown(create_header("Total customers", total, colors["Total"]), unsafe_allow_html=True)
        with col1_2:
            st.markdown(create_header("High Spenders", total_0, colors["High Spenders"]), unsafe_allow_html=True)
        with col1_3:
            st.markdown(create_header("Moderate Engagers", total_1, colors["Moderate Engagers"]), unsafe_allow_html=True)
        with col1_4:
            st.markdown(create_header("Active Savers", total_2, colors["Active Savers"]), unsafe_allow_html=True)
        
        # Llamar a las funciones de gráficos
        selected_var = show_variable_selector()

        # Gráficos
        col1_1, col1_2, col1_3 = st.columns(3)
        with col1_1:
            show_bar_chart(df, selected_var, colors, customer_types)
        with col1_2:
            show_histogram(df, selected_var, colors, customer_types)
        with col1_3:
            show_boxplot(df, selected_var, colors, customer_types)
    with col2:
        show_scatter_plots(df, colors, customer_types)  # Los scatter plots aparecerán aquí


# Función para seleccionar la variable para el gráfico
def show_variable_selector():
    variables = ["n_visitas", "monto_compras", "monto_descuentos"]
    return st.selectbox("Select a variable to display:", variables)

# Función para mostrar el gráfico de barras
def show_bar_chart(df, selected_var, colors, customer_types):
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
    st.plotly_chart(fig)

# Función para mostrar el gráfico de histograma
def show_histogram(df, selected_var, colors, customer_types):
    hist_fig = px.histogram(
        df, x=selected_var, color='cluster', barmode='overlay',
        title=f"Distribution of {selected_var} by Customer Type",
        labels={selected_var: selected_var, "cluster": "Customer Type"},
        template="plotly_dark",
        category_orders={"cluster": [0, 1, 2]},
        color_discrete_map={0: colors["High Spenders"], 1: colors["Moderate Engagers"], 2: colors["Active Savers"]}
    )
    hist_fig.update_traces(opacity=0.8)
    hist_fig.for_each_trace(lambda t: t.update(name=customer_types[int(t.name)]))
    st.plotly_chart(hist_fig)

# Función para mostrar el gráfico de boxplot
def show_boxplot(df, selected_var, colors, customer_types):
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
    boxplot_fig.for_each_trace(lambda t: t.update(name=customer_types[int(t.name)]))
    st.plotly_chart(boxplot_fig)

def show_scatter_plots(df, colors, customer_types):
    scatter_vars = [("n_visitas", "monto_compras"), ("n_visitas", "monto_descuentos"), ("monto_compras", "monto_descuentos")]
    
    # Crear una columna 'cluster_label' solo para la visualización
    df['cluster_label'] = df['cluster'].map({0: "High Spenders", 1: "Moderate Engagers", 2: "Active Savers"})
    
    # Usar solo una columna para apilar los gráficos
    for i, (var_x, var_y) in enumerate(scatter_vars):
        with st.container():  # Usar container para apilar los gráficos
            scatter_fig = px.scatter(
                df, x=var_x, y=var_y, color='cluster_label',
                title=f"Scatter: {var_x} vs {var_y}",
                labels={var_x: var_x, var_y: var_y},
                color_discrete_map={  # Usar exactamente el diccionario de colores proporcionado
                    "High Spenders": "purple", 
                    "Moderate Engagers": "lightblue", 
                    "Active Savers": "yellow"
                },
                template="plotly_dark"
            )
            scatter_fig.update_traces(marker=dict(size=8, opacity=0.8))
            st.plotly_chart(scatter_fig)

    # Después de la visualización, borrar la columna 'cluster_label' si ya no se necesita
    df = df.drop(columns=['cluster_label'])


    