import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from customer_segmentation.dashboard.data import load_data

def show_overview():
    st.header("Overview")
    df = load_data()

    # Estilo para los títulos
    style = '''
    <div style="text-align: center; margin: 0; color: {color};">
        <h6 style="margin-bottom: 0;">{title}</h6>
        <h1 style="margin-top: 0;">{value}</h1>
    </div>
    '''
    gb = df.groupby('cluster').size()
    total, total_0, total_1, total_2 = gb.sum(), gb.loc[0], gb.loc[1], gb.loc[2]

    colors = {"Total": "white", "High Spenders": "purple", "Moderate Engagers": "lightblue", "Active Savers": "yellow"}
    customer_types = ["High Spenders", "Moderate Engagers", "Active Savers"]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(style.format(title="Total customers", value=total, color=colors["Total"]), unsafe_allow_html=True)
    with col2:
        st.markdown(style.format(title="High Spenders", value=total_0, color=colors["High Spenders"]), unsafe_allow_html=True)
    with col3:
        st.markdown(style.format(title="Moderate Engagers", value=total_1, color=colors["Moderate Engagers"]), unsafe_allow_html=True)
    with col4:
        st.markdown(style.format(title="Active Savers", value=total_2, color=colors["Active Savers"]), unsafe_allow_html=True)

    # Variables de análisis
    variables = ["n_visitas", "monto_compras", "monto_descuentos"]
    selected_var = st.selectbox("Select a variable to display:", variables)

    # Gráfico de barras
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

    # Gráfico de histograma
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

    # Agregar boxplots horizontales (usando la misma variable seleccionada)
    boxplot_fig = px.box(
        df, x=selected_var, y='cluster', color='cluster', orientation='h',
        title=f"Boxplot of {selected_var} by Customer Type",
        labels={selected_var: selected_var, "cluster": "Customer Type"},
        category_orders={"cluster": [0, 1, 2]},
        color_discrete_map={0: colors["High Spenders"], 1: colors["Moderate Engagers"], 2: colors["Active Savers"]},
        hover_data={'cluster': True, 'n_visitas': True, 'monto_compras': True, 'monto_descuentos': True}
    )

    # Personalizar el eje Y con los números de los clusters (0, 1, 2)
    boxplot_fig.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=["0", "1", "2"]  # Mostrar números en el eje Y
    )

    # Cambiar los nombres en la leyenda para que sea más amigable
    boxplot_fig.for_each_trace(lambda t: t.update(name=customer_types[int(t.name)]))

    # Mostrar el gráfico de boxplot
    st.plotly_chart(boxplot_fig)
    
    # Gráficos de dispersión
    scatter_vars = [("n_visitas", "monto_compras"), ("n_visitas", "monto_descuentos"), ("monto_compras", "monto_descuentos")]
    col1, col2, col3 = st.columns(3)
    for i, (var_x, var_y) in enumerate(scatter_vars):
        with [col1, col2, col3][i]:
            scatter_fig = px.scatter(
                df, x=var_x, y=var_y, color='cluster',
                title=f"Scatter: {var_x} vs {var_y}",
                labels={var_x: var_x, var_y: var_y},
                color_discrete_map={0: colors["High Spenders"], 1: colors["Moderate Engagers"], 2: colors["Active Savers"]},
                template="plotly_dark"
            )
            scatter_fig.update_traces(marker=dict(size=8, opacity=0.8))
            st.plotly_chart(scatter_fig)


    