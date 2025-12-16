# diagram1_2_combined.py
# -------------------------------------------------------
# Diagram 1: Parcoords (unchanged)
# Diagram 2: Small multiples with METRIC-BASED BRUSHING
# -------------------------------------------------------

import math
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ------------------------
# Data load
# ------------------------
CSV_PATH = "../data/services_weekly.csv"
df = pd.read_csv(CSV_PATH)

# ------------------------
# Preprocessing
# ------------------------
df['week'] = df.get('week', 1).astype(int)
df['month'] = df.get('month', ((df['week'] - 1) // 4 + 1)).astype(int)
df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)

numeric_cols = [
    'patients_admitted', 'patient_satisfaction',
    'staff_morale', 'patients_refused',
    'patients_request', 'available_beds'
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

for c in ['service', 'event']:
    if c not in df.columns:
        df[c] = np.nan

FIXED_SERVICES = ['ICU', 'emergency', 'general_medicine', 'surgery']

# ------------------------
# Shortage logic
# ------------------------
df['availability_status'] = np.where(
    df['patients_request'].fillna(0) > df['available_beds'].fillna(0),
    'Shortage', 'Sufficient'
)
df['status_id'] = df['availability_status'].map({'Sufficient': 0, 'Shortage': 1})

COLORSCALE = [
    [0.0, '#1f77b4'],
    [0.5, '#1f77b4'],
    [0.5, '#d62728'],
    [1.0, '#d62728']
]

# ------------------------
# Aggregation helper
# ------------------------
def aggregate_line(df_in, agg, services, events):
    dff = df_in[df_in['service'].isin(services)].copy()
    if events:
        dff = dff[dff['event'].isin(events)]

    time_col = {'weekly': 'week', 'monthly': 'month', 'quarterly': 'quarter'}[agg]

    g = dff.groupby(['service', time_col]).agg(
        patients_admitted=('patients_admitted', 'sum'),
        patient_satisfaction=('patient_satisfaction', 'mean'),
        staff_morale=('staff_morale', 'mean'),
        patients_refused=('patients_refused', 'sum'),
        patients_request=('patients_request', 'sum'),
        available_beds=('available_beds', 'sum')
    ).reset_index()

    g['availability_status'] = np.where(
        g['patients_request'].fillna(0) > g['available_beds'].fillna(0),
        'Shortage', 'Sufficient'
    )
    g['status_id'] = g['availability_status'].map({'Sufficient': 0, 'Shortage': 1})
    return g, time_col

# ------------------------
# FIXED METRIC ORDER (IMPORTANT)
# ------------------------
METRICS = [
    'patients_admitted',
    'patient_satisfaction',
    'staff_morale',
    'patients_refused'
]

METRIC_LABELS = [
    'Patients Admitted',
    'Patient Satisfaction',
    'Staff Morale',
    'Patients Refused'
]

# ------------------------
# Dash App
# ------------------------
app = dash.Dash(__name__)
server = app.server

all_events = sorted(df['event'].dropna().unique())

app.layout = html.Div([

    html.H2("Combined Dashboard — Metric Brushing in Diagram 2", style={'textAlign': 'center'}),

    dcc.Store(id='global-metric-brush', data=[]),

    html.Div([
        html.Label("Time Granularity"),
        dcc.RadioItems(
            id='time-granularity',
            options=[
                {'label': 'Weekly', 'value': 'weekly'},
                {'label': 'Monthly', 'value': 'monthly'},
                {'label': 'Quarterly', 'value': 'quarterly'}
            ],
            value='weekly',
            inline=True
        ),
        html.Br(),
        html.Label("Event Filter"),
        dcc.Dropdown(
            id='event-filter',
            options=[{'label': e, 'value': e} for e in all_events],
            value=all_events,
            multi=True
        )
    ], style={'width': '90%', 'margin': 'auto'}),

    html.Hr(),

    html.Div([
        html.H3("Diagram 1 — Parcoords"),
        html.Div(id='d1-container')
    ]),

    html.Hr(),

    html.Div([
        html.H3("Diagram 2 — Small Multiples (Metric Brushing)"),
        html.Div(id='d2-container')
    ])
])

# ------------------------
# ✅ FIXED METRIC BRUSH CALLBACK
# ------------------------
@app.callback(
    Output('global-metric-brush', 'data'),
    Input({'type': 'd2-chart', 'index': dash.ALL}, 'clickData'),
    State('global-metric-brush', 'data'),
    prevent_initial_call=True
)
def update_metric_brush(click_list, stored):

    if not isinstance(stored, list):
        stored = []

    if not click_list:
        return stored

    for click in click_list:
        if click and "points" in click:
            point = click["points"][0]

            curve_idx = point.get("curveNumber", None)
            if curve_idx is None or curve_idx >= len(METRIC_LABELS):
                continue

            metric = METRIC_LABELS[curve_idx]

            # TOGGLE
            if metric in stored:
                stored.remove(metric)
            else:
                stored.append(metric)
            break

    return stored

# ------------------------
# Diagram 1 (unchanged)
# ------------------------
@app.callback(
    Output('d1-container', 'children'),
    Input('time-granularity', 'value'),
    Input('event-filter', 'value')
)
def update_d1(agg, events):
    g, _ = aggregate_line(df, agg, FIXED_SERVICES, events)
    figs = []

    for s in FIXED_SERVICES:
        sub = g[g['service'] == s]
        if sub.empty:
            continue

        fig = go.Figure(go.Parcoords(
            line=dict(
                color=sub['status_id'],
                colorscale=COLORSCALE,
                cmin=0, cmax=1
            ),
            dimensions=[
                dict(label='Satisfaction', values=sub['patient_satisfaction']),
                dict(label='Staff Morale', values=sub['staff_morale']),
                dict(label='Admitted', values=sub['patients_admitted']),
                dict(label='Refused', values=sub['patients_refused'])
            ]
        ))

        fig.update_layout(title=s, height=350)
        figs.append(html.Div(dcc.Graph(figure=fig),
                             style={'width': '48%', 'display': 'inline-block'}))

    return figs

# ------------------------
# Diagram 2 — METRIC BRUSHING
# ------------------------
@app.callback(
    Output('d2-container', 'children'),
    Input('time-granularity', 'value'),
    Input('event-filter', 'value'),
    Input('global-metric-brush', 'data')
)
def update_d2(agg, events, brushed_metrics):

    g, time_col = aggregate_line(df, agg, FIXED_SERVICES, events)

    selected = set(brushed_metrics or [])
    is_brushed = bool(selected)

    charts = []

    for s in sorted(g['service'].unique()):
        sub = g[g['service'] == s]
        fig = go.Figure()

        for m, label in zip(METRICS, METRIC_LABELS):
            is_selected = label in selected

            if is_brushed:
                opacity = 1.0 if is_selected else 0.15
                width = 4 if is_selected else 2
            else:
                opacity, width = 1.0, 2

            fig.add_trace(go.Scatter(
                x=sub[time_col],
                y=sub[m],
                mode='lines+markers',
                name=label,
                line=dict(width=width),
                opacity=opacity
            ))

        fig.update_layout(
            title=s,
            height=280,
            margin=dict(l=40, r=10, t=40, b=40),
            xaxis_title=time_col.title(),
            clickmode='event+select'
        )

        charts.append(html.Div(
            dcc.Graph(id={'type': 'd2-chart', 'index': s}, figure=fig),
            style={'border': '1px solid #ddd', 'marginBottom': '10px'}
        ))

    return charts

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
