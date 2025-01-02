import streamlit as st
import plotly.express as px
from dashboard.data import load_data

def show_overview():
    st.header("Overview")
    df = load_data()

    # Styling and columns
    style = '''
    <div style="text-align: center; margin: 0; color: {color};">
        <h6 style="margin-bottom: 0;">{title}</h6>
        <h1 style="margin-top: 0;">{value}</h1>
    </div>
    '''
    gb = df.groupby('cluster').size()
    total, total_0, total_1, total_2 = gb.sum(), gb.loc[0], gb.loc[1], gb.loc[2]

    colors = {"Total": "white", "High Spenders": "purple", "Moderate Engagers": "lightblue", "Active Savers": "yellow"}
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(style.format(title="Total customers", value=total, color=colors["Total"]), unsafe_allow_html=True)
    with col2:
        st.markdown(style.format(title="High Spenders", value=total_0, color=colors["High Spenders"]), unsafe_allow_html=True)
    with col3:
        st.markdown(style.format(title="Moderate Engagers", value=total_1, color=colors["Moderate Engagers"]), unsafe_allow_html=True)
    with col4:
        st.markdown(style.format(title="Active Savers", value=total_2, color=colors["Active Savers"]), unsafe_allow_html=True)

    # Bar chart
    variables = ["n_visitas", "monto_compras", "monto_descuentos"]
    selected_var = st.selectbox("Select a variable to display:", variables)
    data = {
        "Customer Type": ["High Spenders", "Moderate Engagers", "Active Savers"],
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
    fig.update_traces(texttemplate='%{text}', textposition='outside', marker=dict(color=['purple', 'lightblue', 'yellow']))
    st.plotly_chart(fig)
