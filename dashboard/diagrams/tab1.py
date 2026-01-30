import dash
from dash import dcc, html, Input, Output, State, ALL, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate

from colors import COLORS_DICT

# ------------------------
# Constants & Helpers
# ------------------------
SERVICE_COLORS = {
    'emergency': COLORS_DICT['emergency'],
    'ICU': COLORS_DICT['ICU'],
    'surgery': COLORS_DICT['surgery'],
    'general_medicine': COLORS_DICT['general_medicine']
}

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

# Colorscale for PCP
COLORSCALE = [
    [0.0, COLORS_DICT['sufficient']],
    [0.5, COLORS_DICT['sufficient']],
    [0.5, COLORS_DICT['shortage']],
    [1.0, COLORS_DICT['shortage']]
]

METRIC_COLORS = [
    COLORS_DICT['patients_admitted'],
    COLORS_DICT['patient_satisfaction'],
    COLORS_DICT['staff_morale'],
    COLORS_DICT['patients_refused']
]

# Metric colors for line charts

# ------------------------
# Data Aggregation Function
# ------------------------
def aggregate_line(df_in, agg, services, events):
    """
    Aggregate weekly data to different time granularities (weekly/monthly/quarterly).
    Filters by selected services and events, then computes metrics for visualization.
    """
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

    # Calculate bed availability status: Shortage when requests exceed available beds
    g['availability_status'] = np.where(
        g['patients_request'].fillna(0) > g['available_beds'].fillna(0),
        'Shortage', 'Sufficient'
    )
    g['status_id'] = g['availability_status'].map({'Sufficient': 0, 'Shortage': 1})
    return g, time_col

# ------------------------
# Layout
# ------------------------
def create_layout(df):
    # Preprocessing
    if 'month' not in df.columns:
        df['month'] = ((df['week'] - 1) // 4 + 1).astype(int)
    if 'quarter' not in df.columns:
        df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
    
    all_events = sorted(df['event'].dropna().unique())

    return html.Div([
        html.H1("Patient Analysis", 
                style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif'}),
        
        dcc.Store(id='d1-global-metric-brush', data=[]),
        dcc.Store(id='d1-time-selection', data=None),
        dcc.Store(id='d1-pcp-brush-store', data={}),

        # --- Control Panel ---
        html.Div([
            html.Div([
                html.Label("Time Granularity:", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='d1-time-granularity',
                    options=[
                        {'label': ' Weekly', 'value': 'weekly'},
                        {'label': ' Monthly', 'value': 'monthly'},
                        {'label': ' Quarterly', 'value': 'quarterly'}
                    ],
                    value='weekly',
                    inline=True,
                    style={'marginTop': '5px'}
                ),
            ], style={'marginBottom': '15px'}),

            html.Div([
                html.Div([
                    html.Label("Events:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='d1-event-filter',
                        className='event-filter',
                        options=[
                            {'label': EVENT_LABELS.get(e, e), 'value': e}
                            for e in all_events
                        ],
                        value=all_events,
                        multi=True
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("Services:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='d1-service-filter',
                        className='service-filter',
                    options=[
                        {
                            "label": html.Span(
                                SERVICE_LABELS[s],
                                className=f"service-pill service-{s}"
                            ),
                            "value": s
                        }
                        for s in FIXED_SERVICES
                    ],
                        value=FIXED_SERVICES,
                        multi=True,
                        placeholder="Filter services"
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '4%'}),
            ]),
             
            # Reset Buttons
            html.Div([
                html.Button("Reset Selection", id="d1-reset-selection-btn", n_clicks=0, 
                            style={'cursor':'pointer', 'padding': '8px 20px', 'marginRight': '10px'}),
                html.Button("Reset All Filters", id="d1-reset-all-btn", n_clicks=0, 
                            style={'cursor':'pointer', 'padding': '8px 20px', 'marginLeft': '10px'}),
            ], style={'textAlign': 'center', 'marginTop': '15px'})

        ], style={'width': '90%', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px', 'marginBottom': '20px'}),

        html.Div(id='d1-container', style={'width': '95%', 'margin': '0 auto'}),
        
        html.Div([
            html.Div(id='d2-container')
        ], style={'width': '95%', 'margin': '0 auto', 'marginTop': '20px'})
    ])

# ------------------------
# Callbacks
# ------------------------
def register_callbacks(app, df):
    # Preprocessing
    df = df.copy()
    df['week'] = df.get('week', 1).astype(int)
    df['month'] = df.get('month', ((df['week'] - 1) // 4 + 1)).astype(int)
    df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
    
    all_events_callback = sorted(df['event'].dropna().unique())

    numeric_cols = [
        'patients_admitted', 'patient_satisfaction',
        'staff_morale', 'patients_refused',
        'patients_request', 'available_beds'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # ------------------------
    # Callback: Metric Brushing (Line Chart Interaction)
    # Clicking a metric line toggles its selection state
    # ------------------------
    @app.callback(
        Output('d1-global-metric-brush', 'data'),
        Input({'type': 'd2-chart', 'index': ALL}, 'clickData'),
        State('d1-global-metric-brush', 'data'),
        prevent_initial_call=True
    )
    def update_metric_brush(click_list, stored):
        # Initialize storage as empty list if not already a list
        if not isinstance(stored, list):
            stored = []

        # Iterate through click events from all line charts
        for click in click_list or []:
            # Verify click event has valid point data
            if click and 'points' in click:
                # Get the curve number (metric index) from the clicked point
                idx = click['points'][0].get('curveNumber')
                # Check if valid metric index
                if idx is not None and idx < len(METRIC_LABELS):
                    metric = METRIC_LABELS[idx]
                    # Toggle metric selection: remove if already selected, add if not
                    if metric in stored:
                        stored.remove(metric)
                    else:
                        stored.append(metric)
                    break  # Process only the first click event
        return stored

    # ------------------------
    # Callback: Rectangle Time Brushing
    # Capture time range selection via rectangular selection on line charts
    # ------------------------
    @app.callback(
        Output('d1-time-selection', 'data'),
        Input({'type': 'd2-chart', 'index': ALL}, 'selectedData'),
        prevent_initial_call=True
    )
    def capture_rectangle(selected_list):
        # Check if any selection was made; if not, don't update
        if not selected_list or all(sel is None for sel in selected_list):
            raise PreventUpdate

        weeks, service = set(), None
        # Extract all selected points from all line charts
        for sel in selected_list:
            # Verify selection data exists
            if sel and 'points' in sel:
                # Iterate through each selected point
                for p in sel['points']:
                    # Extract week/month/quarter from x-axis (time dimension)
                    if 'x' in p:
                        weeks.add(int(p['x']))
                    # Extract service identifier from customdata
                    if 'customdata' in p:
                        service = p['customdata']

        # Only return selection if both weeks and service are captured
        if not weeks or not service:
            raise PreventUpdate

        return {'service': service, 'weeks': sorted(weeks)}

    # ------------------------
    # Callback: Parallel Coordinates Plot (PCP) Brushing
    # Store constraint ranges when user brushes dimensions in PCP
    # ------------------------
    @app.callback(
        Output('d1-pcp-brush-store', 'data'),
        Input({'type': 'pcp-chart', 'index': ALL}, 'restyleData'),
        State('d1-pcp-brush-store', 'data'),
        prevent_initial_call=True
    )
    def capture_pcp_brush(restyle_list, stored):
        # Initialize storage dictionary if empty
        if not stored:
            stored = {}

        # Process restyle events from parallel coordinates charts
        for restyle in restyle_list or []:
            # Skip empty restyle events
            if not restyle:
                continue

            changes = restyle[0]
            # Store constraint ranges for each modified dimension
            for k, v in changes.items():
                # Key format example: "dimensions[1].constraintrange"
                if "constraintrange" in k:
                    # Keep only the most recent range value for this dimension
                    stored[k] = v[-1]
                else:
                    # Store other restyle properties as-is
                    stored[k] = v

        return stored

    # ------------------------
    # Callback: Update Parallel Coordinates Plots (D1)
    # Generates one PCP per service, showing metrics with bed availability coloring
    # ------------------------
    @app.callback(
        Output('d1-container', 'children'),
        Input('d1-time-granularity', 'value'),
        Input('d1-event-filter', 'value'),
        Input('d1-service-filter', 'value'),  
        Input('d1-time-selection', 'data'),
        Input('d1-pcp-brush-store', 'data')   
    )
    def update_d1(agg, events, services, selection, pcp_brush):
        services = services or FIXED_SERVICES
        g, time_col = aggregate_line(df, agg, services, events)

        selected_weeks = selection['weeks'] if selection else None
    
        if selected_weeks:
            pcp_mask = g[time_col].isin(selected_weeks)
        else:
            pcp_mask = np.ones(len(g), dtype=bool)
        
        figs = []  # Accumulate PCP figures for each service

        # Create one PCP chart for each selected service
        for s in services:
            # Filter aggregated data to current service
            sub = g[g['service'] == s]
            # Create mask to highlight time-selected data points in PCP
            if selected_weeks:
                pcp_mask = sub[time_col].isin(selected_weeks)
            else:
                pcp_mask = np.ones(len(sub), dtype=bool)  # All points highlighted if no time selection

            # Skip if no data for this service
            if sub.empty:
                continue

            # Build PCP dimensions for the four key metrics
            dimensions = [
                dict(label='Satisfaction', values=sub['patient_satisfaction']),
                dict(label='Staff Morale', values=sub['staff_morale']),
                dict(label='Admitted', values=sub['patients_admitted']),
                dict(label='Refused', values=sub['patients_refused'])
            ]

            # Apply stored brush constraints to PCP dimensions
            for k, v in (pcp_brush or {}).items():
                # Parse dimension index from key like "dimensions[1].constraintrange"
                idx = int(k.split('[')[1].split(']')[0])
                dimensions[idx]['constraintrange'] = v
            
            # Color encoding: base color by availability status, brightness by time selection
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
                height=520,
                margin=dict(l=70, r=70, t=40, b=60),
                font=dict(size=15)
            )

            service_color = SERVICE_COLORS.get(s, "#7f8c8d")

            figs.append(
                html.Div(
                    [
                        html.Div(
                            SERVICE_LABELS.get(s, s),
                            style={
                                "fontSize": "22px",
                                "fontWeight": "600",
                                "marginBottom": "4px",
                            },
                        ),
                        html.Div(
                            style={
                                "height": "4px",
                                "backgroundColor": service_color,
                                "marginBottom": "8px",
                            },
                        ),
                        dcc.Graph(
                            id={'type': 'pcp-chart', 'index': s},
                            figure=fig
                        )
                    ]
                )
            )

        return html.Div(
            figs,
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, 1fr)",
                "gridAutoRows": "580px",
                "alignItems": "start",
                "gap": "12px"
            }
        )

    # ------------------------
    # Callback: Update Small Multiples Line Charts (D2)
    # One line chart per service showing all metrics over time
    # Supports metric highlighting, time selection bands, and PCP brushing
    # ------------------------
    @app.callback(
        Output('d2-container', 'children'),
        Input('d1-time-granularity', 'value'),
        Input('d1-event-filter', 'value'),
        Input('d1-service-filter', 'value'),   
        Input('d1-global-metric-brush', 'data'),
        Input('d1-pcp-brush-store', 'data'),
        State('d1-time-selection', 'data')
    )
    def update_d2(agg, events, services, brushed, pcp_brush, selection):
        # Map PCP dimension indices to metric column names
        pcp_dim_to_metric = {
            0: 'patient_satisfaction',
            1: 'staff_morale',
            2: 'patients_admitted',
            3: 'patients_refused'
        }

        # Extract PCP brush ranges to highlight matching points in line charts
        pcp_ranges = {}
        # Convert stored PCP constraints to a dimension-indexed dictionary
        for k, v in (pcp_brush or {}).items():
            # Extract dimension index from constraint key
            idx = int(k.split('[')[1].split(']')[0])
            # Only store if this dimension maps to a metric we display
            if idx in pcp_dim_to_metric:
                pcp_ranges[idx] = v

        services = services or FIXED_SERVICES
        g, time_col = aggregate_line(df, agg, services, events)
        
        # Determine time range for shaded selection band on charts
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

        selected = set(brushed or [])
        is_brushed = bool(selected)
        charts = []

        for s in services:
            sub = g[g['service'] == s]
            fig = go.Figure()
            
            # Extract event data for hover tooltips
            event_series = (
                df[
                    (df['service'] == s) &
                    (df[time_col].isin(sub[time_col]))
                ]
                .drop_duplicates(subset=[time_col])
                .set_index(time_col)['event']
                .reindex(sub[time_col])
            )

            # Plot each of the four metrics with conditional highlighting based on brushing
            for idx, (m, label) in enumerate(zip(METRICS, METRIC_LABELS)):
                # Check if this metric was selected via click on line chart
                sel = label in selected
                pcp_mask = np.zeros(len(sub), dtype=bool)
                
                # Check if PCP brush constraints apply to this metric
                for dim_idx, v in pcp_ranges.items():
                    # If dimension constraint matches this metric
                    if pcp_dim_to_metric[dim_idx] == m and v:
                        # Normalize to list of ranges (may be single range or multiple)
                        ranges = v if isinstance(v[0], (list, tuple)) else [v]
                        mask_part = np.zeros(len(sub), dtype=bool)
                        # Create mask for points within any of the constraint ranges
                        for lo, hi in ranges:
                            mask_part |= (sub[m] >= lo) & (sub[m] <= hi)
                        pcp_mask = mask_part

                fig.add_trace(go.Scatter(
                    x=sub[time_col],
                    y=sub[m],
                    name=label,
                    mode='lines+markers',
                    customdata=[s] * len(sub),
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
                    line=dict(color=METRIC_COLORS[idx], width=4 if sel else 2),
                    marker=dict(
                        color=METRIC_COLORS[idx],
                        size=np.where(pcp_mask, 12, 4)
                    )
                ))

            # Shaded time band
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
                        fillcolor="rgba(255, 165, 0, 0.25)",
                        line=dict(width=0),
                        layer="below"
                    )
                )

            fig.update_layout(
                dragmode='select',
                clickmode='event+select',
                xaxis=dict(title=dict(text=x_axis_label, font=dict(size=17)), tickfont=dict(size=15)),
                yaxis=dict(title=dict(text='Value', font=dict(size=17)), tickfont=dict(size=15)),
                uirevision=f"line-chart-{s}",
                selectionrevision="keep-selection",
                shapes=shapes,
                margin=dict(l=70, r=70, t=40, b=60),
                font=dict(size=15),
                legend=dict(font=dict(size=15))
            )

            service_color = SERVICE_COLORS.get(s, "#7f8c8d")

            charts.append(
                html.Div(
                    [
                        html.Div(
                            SERVICE_LABELS.get(s, s),
                            style={
                                "fontSize": "22px",
                                "fontWeight": "600",
                                "marginBottom": "4px",
                                "marginTop": "20px"
                            },
                        ),
                        html.Div(
                            style={
                                "height": "4px",
                                "backgroundColor": service_color,
                                "marginBottom": "8px",
                            },
                        ),
                        dcc.Graph(
                            id={'type': 'd2-chart', 'index': s},
                            figure=fig
                        )
                    ]
                )
            )

        return charts

    # Reset buttons callback
    @app.callback(
        [Output('d1-global-metric-brush', 'data', allow_duplicate=True),
         Output('d1-time-selection', 'data', allow_duplicate=True),
         Output('d1-pcp-brush-store', 'data', allow_duplicate=True),
         Output('d1-time-granularity', 'value'),
         Output('d1-event-filter', 'value'),
         Output('d1-service-filter', 'value')],
        [Input('d1-reset-selection-btn', 'n_clicks'),
         Input('d1-reset-all-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def handle_resets(btn_selection, btn_all):
        # Determine which reset button was clicked
        triggered = ctx.triggered_id
        
        # Reset values for all interactive brushing stores (metric, time, PCP)
        res_stores = [[], None, {}]
        
        # Check if full reset (all filters) was triggered
        if triggered == 'd1-reset-all-btn':
            # Reset everything: stores + filters + granularity to initial state
            return res_stores + ['weekly', all_events_callback, FIXED_SERVICES]
        
        # Partial reset: clear interactive selections only, keep current filter settings
        return res_stores + [dash.no_update, dash.no_update, dash.no_update]

