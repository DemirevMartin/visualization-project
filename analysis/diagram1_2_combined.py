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

SERVICE_LABELS = {
    'emergency': 'Emergency',
    'general_medicine': 'General Medicine',
    'ICU': 'ICU',
    'surgery': 'Surgery'
}

FIXED_SERVICES = list(SERVICE_LABELS.keys())
EVENT_LABELS = {
    'flu': 'Flu',
    'donation': 'Donation',
    'strike': 'Strike',
    'none': 'None'
}
FIXED_EVENTS = list(EVENT_LABELS.keys())


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
    dcc.Store(id='pcp-brush-store', data={}),
    


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
        options=[
            {'label': EVENT_LABELS.get(e, e), 'value': e}
            for e in all_events
        ],
        value=all_events,
        multi=True
    ),

        dcc.Dropdown(
        id='service-filter',
        options=[
            {'label': SERVICE_LABELS[s], 'value': s}
            for s in FIXED_SERVICES
        ],
        value=FIXED_SERVICES,
        multi=True,
        placeholder="Filter services"
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
from dash.exceptions import PreventUpdate
@app.callback(
    Output('d2-time-selection', 'data'),
    Input({'type': 'd2-chart', 'index': dash.ALL}, 'selectedData'),
    prevent_initial_call=True
)
def capture_rectangle(selected_list):

    if not selected_list or all(sel is None for sel in selected_list):
        raise PreventUpdate

    weeks, service = set(), None
    for sel in selected_list:
        if sel and 'points' in sel:
            for p in sel['points']:
                if 'x' in p:
                    weeks.add(int(p['x']))
                if 'customdata' in p:
                    service = p['customdata']

    if not weeks or not service:
        raise PreventUpdate

    return {'service': service, 'weeks': sorted(weeks)}

@app.callback(
    Output('pcp-brush-store', 'data'),
    Input({'type': 'pcp-chart', 'index': dash.ALL}, 'restyleData'),
    State('pcp-brush-store', 'data'),
    prevent_initial_call=True
)
def capture_pcp_brush(restyle_list, stored):
    if not stored:
        stored = {}

    for restyle in restyle_list or []:
        if not restyle:
            continue

        changes = restyle[0]   # what changed
        for k, v in changes.items():
            # Example key: "dimensions[1].constraintrange"
            if "constraintrange" in k:
                stored[k] = v

    return stored


# ------------------------
# Diagram 1 — PCP (LINKED)
# ------------------------
@app.callback(
    Output('d1-container', 'children'),
    Input('time-granularity', 'value'),
    Input('event-filter', 'value'),
    Input('service-filter', 'value'),  
    Input('d2-time-selection', 'data'),
    Input('pcp-brush-store', 'data')   
)
def update_d1(agg, events, services, selection, pcp_brush):
    services = services or FIXED_SERVICES
    g, time_col = aggregate_line(df, agg,services, events)

    selected_weeks = selection['weeks'] if selection else None
   
    if selected_weeks:
        pcp_mask = g[time_col].isin(selected_weeks)
    else:
        pcp_mask = np.ones(len(g), dtype=bool)
    
    figs = []

    for s in services:
        sub = g[g['service'] == s]
        if selected_weeks:
            pcp_mask = sub[time_col].isin(selected_weeks)
        else:
            pcp_mask = np.ones(len(sub), dtype=bool)

        if sub.empty:
            continue

        # ---- PCP dimensions (EXACTLY same metrics as before)
        dimensions = [
            dict(label='Satisfaction', values=sub['patient_satisfaction']),
            dict(label='Staff Morale', values=sub['staff_morale']),
            dict(label='Admitted', values=sub['patients_admitted']),
            dict(label='Refused', values=sub['patients_refused'])
        ]

        # ---- APPLY INTEGRATED BRUSHING (NEW, minimal)
        for k, v in (pcp_brush or {}).items():
            # k looks like: "dimensions[1].constraintrange"
            idx = int(k.split('[')[1].split(']')[0])
            dimensions[idx]['constraintrange'] = v
        pcp_color = sub['status_id'].astype(float) + np.where(pcp_mask, 0.5, 0.0)
        fig = go.Figure(go.Parcoords(
        line=dict(
            color=pcp_color,
            cmin=0.0,
            cmax=1.5,
            colorscale=[
                [0.0, '#c6dbef'],   # Sufficient (dim)
                [0.33, '#1f77b4'],  # Sufficient (selected)
                [0.66, '#f2b6b6'],  # Shortage (dim)
                [1.0, '#d62728']    # Shortage (selected)
            ],
            showscale=True,
            colorbar=dict(
                title="Availability",
                tickvals=[0.25, 1.25],
                ticktext=["Sufficient", "Shortage"]
            )
        ),
        dimensions=dimensions
    ))


        fig.update_layout(
            title=SERVICE_LABELS.get(s, s),
            height=450
        )

        figs.append(
            dcc.Graph(
                id={'type': 'pcp-chart', 'index': s},  # ← REQUIRED for linking
                figure=fig
            )
        )

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
    Input('service-filter', 'value'),   
    Input('global-metric-brush', 'data'),
    Input('pcp-brush-store', 'data'),
    State('d2-time-selection', 'data')   # ← ADD
)
def update_d2(agg, events,services, brushed, pcp_brush, selection):
    pcp_dim_to_metric = {
        0: 'patient_satisfaction',
        1: 'staff_morale',
        2: 'patients_admitted',
        3: 'patients_refused'
    }

    pcp_ranges = {}
    for k, v in (pcp_brush or {}).items():
        idx = int(k.split('[')[1].split(']')[0])
        if idx in pcp_dim_to_metric:
            pcp_ranges[idx] = v

    services = services or FIXED_SERVICES
    g, time_col = aggregate_line(df, agg, services, events)
        # ------------------------
    # Rectangle time bounds
    # ------------------------
    if selection and selection.get('weeks'):
        x0 = min(selection['weeks'])
        x1 = max(selection['weeks'])
    else:
        x0 = x1 = None

    x_axis_label = {
        'weekly': 'Week',
        'monthly': 'Month',
        'quarterly': 'Quarter'
    }[agg]

    # Context (flu / donation / strike / none) per service & time
    context_map = (
    df[['service', time_col, 'event']]
    .drop_duplicates(subset=['service', time_col])
)

    # --- FIXED X-AXIS RANGE (global) ---
    #x_min = g[time_col].min()
    #x_max = g[time_col].max()

    selected = set(brushed or [])
    is_brushed = bool(selected)
    charts = []

    for s in services:
        sub = g[g['service'] == s]
        fig = go.Figure()
        event_series = (
            df[
                (df['service'] == s) &
                (df[time_col].isin(sub[time_col]))
            ]
            .drop_duplicates(subset=[time_col])
            .set_index(time_col)['event']
            .reindex(sub[time_col])
        )

        for m, label in zip(METRICS, METRIC_LABELS):
            sel = label in selected
            pcp_mask = np.zeros(len(sub), dtype=bool)
            for idx, v in pcp_ranges.items():
                if pcp_dim_to_metric[idx] == m and v:

                    # v may be [lo, hi] OR [[lo, hi], [lo2, hi2], ...]
                    ranges = v if isinstance(v[0], (list, tuple)) else [v]

                    mask_part = np.zeros(len(sub), dtype=bool)
                    for lo, hi in ranges:
                        mask_part |= (sub[m] >= lo) & (sub[m] <= hi)

                    pcp_mask = mask_part

                
            fig.add_trace(go.Scatter(
            x=sub[time_col],
            y=sub[m],
            name=label,
            mode='lines+markers',

            # DO NOT TOUCH — required for brushing
            customdata=[s] * len(sub),

            # Event shown ONLY in hover
            hovertext=[
                EVENT_LABELS.get(ev, ev) if pd.notna(ev) else "None"
                for ev in event_series
            ],

            hovertemplate=(
                "Service: %{customdata}<br>"
                "Event: %{hovertext}<br>"
                "Value: %{y}<extra></extra>"
            ),

            opacity=1.0 if (not is_brushed or sel) else 0.15,
            line=dict(width=4 if sel else 2),
            # PCP highlight = bigger filled dots (NEW)
            marker=dict(
            size=np.where(pcp_mask, 12, 4)
        )
        ))

        # ------------------------
# Shaded time band (rectangle surrogate)
# ------------------------
        shapes = []

        if x0 is not None and x1 is not None:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(255, 165, 0, 0.25)",  # translucent orange
                    line=dict(width=0),
                    layer="below"
                )
            )

        fig.update_layout(
            title=SERVICE_LABELS.get(s, s),
            dragmode='select',
            clickmode='event+select',
            xaxis=dict(title=x_axis_label),
            yaxis=dict(
            title='Value'
            ),
            uirevision=f"line-chart-{s}",
            selectionrevision="keep-selection",
            shapes=shapes    
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
    Output('pcp-brush-store', 'data', allow_duplicate=True),
    Input('reset-selection-btn', 'n_clicks'),
    prevent_initial_call=True
)

def reset_all_selections(n_clicks):
    if n_clicks:
        return [], None, {}
    return dash.no_update, dash.no_update

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
