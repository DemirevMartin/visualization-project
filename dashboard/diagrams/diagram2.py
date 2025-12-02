import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import Counter

def create_seasonal_heatmap(df, metric, aggregation='month'):
    """Generates the Seasonal Heatmap."""
    
    # 1. Aggregate Data
    if aggregation == 'quarter':
        df['timebin'] = ((df['week'] - 1) // 13 + 1).astype(str)
        x_label = "Quarter"
    else: # month
        df['timebin'] = df['month']
        x_label = "Month"
        
    # Group by Service + Timebin
    grouped = df.groupby(['service', 'timebin'])
    
    # Calculate Metric Mean
    z_data = grouped[metric].mean().unstack(fill_value=0)
    
    # Calculate Frequent Events (for tooltip)
    def most_common_event(series):
        c = Counter([x for x in series if x != 'none'])
        return c.most_common(1)[0][0] if c else 'None'
    
    events = grouped['event'].agg(most_common_event).unstack(fill_value='None')

    # Prepare Hover Text
    hover_text = []
    for service in z_data.index:
        row_txt = []
        for time in z_data.columns:
            val = z_data.loc[service, time]
            evt = events.loc[service, time]
            row_txt.append(f"Service: {service}<br>{x_label}: {time}<br>Value: {val:.1f}<br>Event: {evt}")
        hover_text.append(row_txt)

    # Plot
    fig = go.Figure(data=go.Heatmap(
        z=z_data.values,
        x=z_data.columns,
        y=[s.replace('_',' ').title() for s in z_data.index],
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorscale='Viridis',
        colorbar=dict(title=metric.replace('_',' ').title())
    ))

    fig.update_layout(
        title=f"Seasonal Patterns ({metric}) by {x_label}",
        height=500,
        yaxis_autorange='reversed' # Service list top-to-bottom
    )
    return fig