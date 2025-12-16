# diagram1_2_combined.py
# -------------------------------------------------------
# Diagram 1: Parcoords (PCP)
# Diagram 2: Small multiples with METRIC + TIME brushing
# -------------------------------------------------------

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
# Fixed metric order
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

    html.H2("Linked Dashboard — Metric & Time Brushing"),

    dcc.Store(id='global-metric-brush', data=[]),
    dcc.Store(id='d2-time-selection', data=None),

    html.Div([
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
        dcc.Dropdown(
            id='event-filter',
            options=[{'label': e, 'value': e} for e in all_events],
            value=all_events,
            multi=True
        )
    ]),

    html.Hr(),

    html.Div(id='d1-container'),
    html.Hr(),

    html.Div(
    [
        html.Div(
            html.Button(
                "Reset selection",
                id="reset-selection-btn",
                n_clicks=0,
                style={
                    "marginBottom": "8px",
                    "padding": "6px 12px",
                    "fontWeight": "bold"
                }
            ),
            style={"textAlign": "right"}
        ),
        html.Div(id='d2-container')
    ]
)

])

# ------------------------
# Metric brushing (UNCHANGED)
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

    for click in click_list or []:
        if click and 'points' in click:
            idx = click['points'][0].get('curveNumber')
            if idx is not None and idx < len(METRIC_LABELS):
                metric = METRIC_LABELS[idx]
                if metric in stored:
                    stored.remove(metric)
                else:
                    stored.append(metric)
                break
    return stored

# ------------------------
# Rectangle time brushing (NEW)
# ------------------------
@app.callback(
    Output('d2-time-selection', 'data'),
    Input({'type': 'd2-chart', 'index': dash.ALL}, 'selectedData'),
    prevent_initial_call=True
)
def capture_rectangle(selected_list):
    if not selected_list:
        return None

    weeks, service = set(), None
    for sel in selected_list:
        if sel and 'points' in sel:
            for p in sel['points']:
                if 'x' in p:
                    weeks.add(int(p['x']))
                if 'customdata' in p:
                    service = p['customdata']

    if not weeks or not service:
        return None

    return {'service': service, 'weeks': sorted(weeks)}



# ------------------------
# Diagram 1 — PCP (LINKED)
# ------------------------
@app.callback(
    Output('d1-container', 'children'),
    Input('time-granularity', 'value'),
    Input('event-filter', 'value'),
    Input('d2-time-selection', 'data')
)
def update_d1(agg, events, selection):
    g, time_col = aggregate_line(df, agg, FIXED_SERVICES, events)

    if selection:
        g = g[
            (g['service'] == selection['service']) &
            (g[time_col].isin(selection['weeks']))
        ]

    figs = []
    for s in FIXED_SERVICES:
        sub = g[g['service'] == s]
        if sub.empty:
            continue

        fig = go.Figure(go.Parcoords(
            line=dict(color=sub['status_id'], colorscale=COLORSCALE),
            dimensions=[
                dict(label='Satisfaction', values=sub['patient_satisfaction']),
                dict(label='Staff Morale', values=sub['staff_morale']),
                dict(label='Admitted', values=sub['patients_admitted']),
                dict(label='Refused', values=sub['patients_refused'])
            ]
        ))
        fig.update_layout(title=s, height=450)
        figs.append(dcc.Graph(figure=fig))

    return html.Div(
        figs,
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(2, 1fr)',
            'gap': '12px'
        }
    )

# ------------------------
# Diagram 2 — Small multiples
# ------------------------
@app.callback(
    Output('d2-container', 'children'),
    Input('time-granularity', 'value'),
    Input('event-filter', 'value'),
    Input('global-metric-brush', 'data')
)
def update_d2(agg, events, brushed):
    g, time_col = aggregate_line(df, agg, FIXED_SERVICES, events)

    # Context (flu / donation / strike / none) per service & time
    context_map = (
    df[['service', time_col, 'event']]
    .drop_duplicates(subset=['service', time_col])
)

    # --- FIXED X-AXIS RANGE (global) ---
    x_min = g[time_col].min()
    x_max = g[time_col].max()

    selected = set(brushed or [])
    is_brushed = bool(selected)
    charts = []

    for s in sorted(g['service'].unique()):
        sub = g[g['service'] == s]
        fig = go.Figure()

        for m, label in zip(METRICS, METRIC_LABELS):
            sel = label in selected
            fig.add_trace(go.Scatter(
                x=sub[time_col],
                y=sub[m],
                name=label,
                mode='lines+markers',
                customdata=[s] * len(sub),
                opacity=1.0 if (not is_brushed or sel) else 0.15,
                line=dict(width=4 if sel else 2)
            ))

        fig.update_layout(
            title=s,
            dragmode='select',
            clickmode='event+select',
            xaxis=dict(range=[x_min, x_max])
        )

        charts.append(dcc.Graph(
            id={'type': 'd2-chart', 'index': s},
            figure=fig
        ))

    return charts
# ------------------------
# Reset button: clear metric & rectangle selections
# ------------------------
@app.callback(
    Output('global-metric-brush', 'data', allow_duplicate=True),
    Output('d2-time-selection', 'data', allow_duplicate=True),
    Input('reset-selection-btn', 'n_clicks'),
    prevent_initial_call=True
)

def reset_all_selections(n_clicks):
    if n_clicks:
        return [], None
    return dash.no_update, dash.no_update

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
