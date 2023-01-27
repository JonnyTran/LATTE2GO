import warnings

import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from pandas import DataFrame


def configure_layout(fig, showlegend=True, showticklabels=False, showgrid=False, **kwargs) -> go.Figure:
    # Figure
    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=showgrid,
                showticklabels=showticklabels,
                title='',
                )

    fig.update_layout(
        **kwargs,
        showlegend=showlegend,
        # legend=dict(autosize=True, width=100),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        legend_orientation="v",
        autosize=False,
        margin=dict(
            l=5,
            r=5,
            b=5,
            t=30 if 'title' in kwargs else 5,
            pad=5
        ),
        xaxis=axis,
        yaxis=axis
    )
    return fig

def plot_sankey_flow(nodes: DataFrame, links: DataFrame, opacity=0.6, font_size=8, orientation="h",
                     **kwargs):
    # change hex to rgba color to add opacity
    rgba_colors = [f"rgba{tuple(int(val * 255) for val in to_rgb(color)) + (opacity if src != dst else 0,)}" \
                   for i, (src, dst, color) in links[['source', 'target', 'color']].iterrows()]

    if (nodes.index != nodes.reset_index().index).all():
        warnings.warn("`nodes.index` are not contiguous integers.")

    fig = go.Figure(data=[
        go.Sankey(
            valueformat=".2f",
            orientation=orientation,
            arrangement="snap",
            # Define nodes
            node=dict(
                pad=5,
                thickness=15,
                # line=dict(color="black", width=np.where(nodes['level'] % 2, 0, 0)),
                label=nodes['label'],
                color=nodes['color'],
                customdata=nodes['count'],
                hovertemplate='num_nodes: %{customdata}',
            ),
            # Add links between nodes
            link=dict(
                label=links['label'],
                source=links['source'],
                target=links['target'],
                value=links['mean'],
                color=rgba_colors,
                # hoverlabel=dict(align='left'),
                customdata=links['std'],
                hovertemplate='%{label}: %{value} ± %{customdata:.3f}',
            ))],
        layout_xaxis_range=[0, 1],
        layout_yaxis_range=[0, 1],
    )

    if 'layer' in nodes.columns:
        max_level = nodes['level'].max()
        for layer in nodes['layer'].unique():
            level = (max_level - nodes.query(f'layer == {layer}')['level']).max() + 1
            level = min(level, max_level - 1)
            # print(layer, level, max_level-1, level / (max_level-1))
            fig.add_vline(x=level / (max_level - 1), layer='below',
                          annotation_text=f'Layer {layer + 1}', annotation_position="top left",
                          line_dash="dot", line_color="gray", opacity=0.50, )

    fig = configure_layout(fig, paper_bgcolor='rgba(255,255,255,255)',
                           plot_bgcolor='rgba(0,0,0,0)', **kwargs)
    fig.update_layout(font_size=font_size)
    return fig
