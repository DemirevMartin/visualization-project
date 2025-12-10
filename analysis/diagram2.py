import pandas as pd
import numpy as np
from collections import Counter

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ------------------------
# Config / Data load
# ------------------------
CSV_PATH = "../data/services_weekly.csv"
df = pd.read_csv(CSV_PATH)

# Basic preprocessing
df['week'] = df['week'].astype(int)
if 'month' in df.columns:
    df['month'] = df['month'].astype(int)
else:
    df['month'] = ((df['week'] - 1) // 4 + 1).clip(1, 12)

numeric_cols = ['patients_admitted', 'patient_satisfaction', 'staff_morale']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)

all_services = sorted(df['service'].dropna().unique().tolist())
all_events = sorted(df['event'].dropna().unique().tolist())

# ------------------------
# Aggregation helper
# ------------------------
def aggregate_line(df_in, agg='weekly', services=None, events=None):
    df_f = df_in.copy()
    if services:
        df_f = df_f[df_f['service'].isin(services)]
    if events:
        df_f = df_f[df_f['event'].isin(events)]

    if agg == 'weekly':
        time_col = 'week'
    elif agg == 'monthly':
        time_col = 'month'
    elif agg == 'quarterly':
        time_col = 'quarter'
    else:
        time_col = 'week'

    grouped = df_f.groupby(['service', time_col]).agg(
        patients_admitted=('patients_admitted', 'sum'),
        patient_satisfaction=('patient_satisfaction', 'mean'),
        staff_morale=('staff_morale', 'mean')
    ).reset_index()

    return grouped, time_col

# ------------------------
# Dash App Layout
# ------------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H3("Seasonal Metrics – Multi-Metric Line Charts (Small Multiples)"),

    html.Div([
        # Aggregation
        html.Div([
            html.Label("Aggregation:"),
            dcc.RadioItems(id='agg', options=[
                {'label': 'Weekly', 'value': 'weekly'},
                {'label': 'Monthly', 'value': 'monthly'},
                {'label': 'Quarterly', 'value': 'quarterly'}
            ], value='weekly', inline=True)
        ], style={'display': 'inline-block', 'margin-right': '30px'}),

        # Metric multi-selection
        html.Div([
            html.Label("Metrics:"),
            dcc.Dropdown(id='metric', options=[
                {'label': 'Patients Admitted', 'value': 'patients_admitted'},
                {'label': 'Patient Satisfaction', 'value': 'patient_satisfaction'},
                {'label': 'Staff Morale', 'value': 'staff_morale'}
            ], value=['patients_admitted'], multi=True)
        ], style={'display': 'inline-block', 'width': '30%', 'margin-right': '30px'}),

        # Service filter
        html.Div([
            html.Label("Service filter:"),
            dcc.Dropdown(id='service-filter', options=[{'label': s, 'value': s} for s in all_services],
                         value=all_services, multi=True)
        ], style={'display': 'inline-block', 'width': '30%'})
    ], style={'padding': '8px', 'border': '1px solid #ddd', 'margin-bottom': '10px'}),

    html.Div([
        html.Label("Event filter:"),
        dcc.Dropdown(id='event-filter', options=[{'label': e, 'value': e} for e in all_events],
                     value=all_events, multi=True)
    ], style={'width': '50%', 'padding-bottom': '12px'}),

    html.Div(id='charts-container')
])

# ------------------------
# Callback to generate small-multiple line charts
# ------------------------
@app.callback(
    Output('charts-container', 'children'),
    Input('agg', 'value'),
    Input('metric', 'value'),
    Input('service-filter', 'value'),
    Input('event-filter', 'value'),
    Input({'type': 'service-chart', 'index': dash.ALL}, 'clickData')
)
def update_charts(agg, metrics_sel, services_sel, events_sel, click_list):
    grouped, time_col = aggregate_line(df, agg=agg, services=services_sel, events=events_sel)
    # Compute global x-axis range

    global_min = grouped[time_col].min()
    global_max = grouped[time_col].max()
    services_sorted = sorted(grouped['service'].unique())

    #Compute gloabl y-axis range
    # ---- Global Y range (based on selected metrics only) ----
    global_min_y = grouped[metrics_sel].min().min()
    #global_max_y = grouped[metrics_sel].max().max()

    # ---------------------------
    # Detect clicked metric line
    # ---------------------------
    clicked_metric = None
    if click_list:
        for click in click_list:
            if click and "curveNumber" in click:
                trace_index = click["curveNumber"]
                point = click["points"][0]
                clicked_metric = point["data"]["name"]  # metric name
                break

    charts = []

    for s in services_sorted:
        sub = grouped[grouped['service'] == s]
        fig = go.Figure()

        for idx, metric in enumerate(metrics_sel):

            # ------------ Highlight clicked metric ------------
            if clicked_metric == metric.replace('_', ' ').title():
                opacity = 1.0
                line_width = 4
            else:
                opacity = 0.2 if clicked_metric else 1.0
                line_width = 2

            fig.add_trace(go.Scatter(
                x=sub[time_col],
                y=sub[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(width=line_width),
                opacity=opacity,

                # ------------ Hover shows only the y-value ------------
                hovertemplate="%{y}<extra></extra>"
            ))

        fig.update_layout(
            title=f"{s} — Selected Metrics",
            height=300,
            margin=dict(l=40, r=10, t=40, b=40),
            xaxis_title=time_col.replace('_', ' ').title(),
            yaxis_title="Value",
            xaxis=dict(range=[global_min, global_max]),   # <--- LOCKED SCALE
            #yaxis=dict(range=[global_min_y, global_max_y])  # <--- LOCKED SCALE
        )

        charts.append(html.Div([
            dcc.Graph(
                id={'type': 'service-chart', 'index': s},
                figure=fig,
                clear_on_unhover=True
            )
        ], style={'width': '100%', 'border': '1px solid #ddd', 'margin-bottom': '10px'}))

    return charts


# ------------------------
# Run app
# ------------------------
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)