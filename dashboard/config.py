CUSTOMER_TYPES = ["High Spenders", "Balanced Users", "Active Savers"]

CLUSTER_COLORS = {
    CUSTOMER_TYPES[0]: "#78809D",
    CUSTOMER_TYPES[1]: "#AAA2B1",
    CUSTOMER_TYPES[2]: "#DBC6C0"
}

DEFAULT_BACKGROUND_COLOR = "#1B1D22"

def configure_boxplot_figure(fig, cluster_labels, color, height=400, legend_position=(0.75, 1.2)):
    fig.update_yaxes(tickvals=list(range(len(cluster_labels))), ticktext=cluster_labels)
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

def configure_pie_chart_figure(fig, color, height=300):
    fig.update_layout(
        paper_bgcolor=color,
        margin=dict(l=0, r=50, t=50, b=10),
        title=dict(font=dict(size=15, color="white"), xanchor="center", x=0.5, y=0.92),
        legend=dict(font=dict(size=15, color="white"), x=1, yanchor="middle", y=0.5),
        height=height
    )
    fig.update_traces(textfont=dict(size=20))
    return fig

def configure_scatter_figure(fig, color, height=309, marker_size=8, marker_opacity=0.8):
    fig.update_layout(
        plot_bgcolor=color,
        paper_bgcolor=color,
        height=height,
        margin=dict(l=50, r=50, t=70, b=70),
        title=dict(font=dict(size=15, color="white"), xanchor="center", x=0.5, y=0.94)
    )
    fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity), showlegend=False)
    return fig

def configure_figure(fig, color, height=300):
    fig.update_layout(
        plot_bgcolor=color,
        paper_bgcolor=color,
        height=height,
        title=dict(xanchor="center", x=0.5)
    )
    fig.update_traces(opacity=1)
    return fig
