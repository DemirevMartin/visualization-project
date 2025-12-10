# Combined Dash app — single shared time-granularity radio (Week/Month/Quarter)
# - Single shared Event filter
# - Services fixed to ICU, emergency, general_medicine, surgery
# - No week-range slider for Diagram 1
# - time-granularity drives both Diagram 1 and Diagram 2
#
# Core sum/mean aggregation logic preserved.

import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ------------------------
# Shared Config / Data load
# ------------------------
CSV_PATH = "../data/services_weekly.csv"
df = pd.read_csv(CSV_PATH)

# ------------------------
# Common preprocessing
# ------------------------
if 'week' in df.columns:
    df['week'] = df['week'].astype(int)
if 'month' in df.columns:
    df['month'] = df['month'].astype(int)
else:
    if 'week' in df.columns:
        df['month'] = ((df['week'] - 1) // 4 + 1).clip(1, 12)
    else:
        df['month'] = 1

numeric_cols = ['patients_admitted', 'patient_satisfaction', 'staff_morale', 'patients_refused']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)

# ensure optional columns exist to avoid KeyErrors
for c in ['patients_request', 'available_beds', 'service', 'event']:
    if c not in df.columns:
        df[c] = np.nan

# ------------------------
# Fixed services
# ------------------------
FIXED_SERVICES = ['ICU', 'emergency', 'general_medicine', 'surgery']

# ------------------------
# Parcoords (Figure 1) base config
# ------------------------
df['availability_status'] = df.apply(
    lambda row: 'Shortage' if row.get('patients_request', 0) > row.get('available_beds', 0) else 'Sufficient',
    axis=1
)
status_map = {'Sufficient': 0, 'Shortage': 1}
df['status_id'] = df['availability_status'].map(status_map).fillna(0).astype(int)

COLORS = ['#1f77b4', '#d62728']
COLORSCALE = [
    [0.0, COLORS[0]],
    [0.5, COLORS[0]],
    [0.5, COLORS[1]],
    [1.0, COLORS[1]]
]

metrics_config_fixed = [
    {'col': 'patient_satisfaction', 'label': 'Satisfaction', 'range': [0, 100]},
    {'col': 'staff_morale',         'label': 'Staff Morale', 'range': [0, 100]},
    {'col': 'patients_admitted',    'label': 'Admitted',     'range': [0, df['patients_admitted'].max()] if 'patients_admitted' in df.columns else [0,1]},
    {'col': 'patients_refused',     'label': 'Refusals',     'range': [0, df['patients_refused'].max()] if 'patients_refused' in df.columns else [0,1]}
]

# ------------------------
# Aggregation helper used by Small-Multiples (and reused for Parcoords when aggregating)
# (keeps sum/mean exactly the same as original)
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
        staff_morale=('staff_morale', 'mean'),
        patients_refused=('patients_refused', 'sum')
    ).reset_index()

    return grouped, time_col

# ------------------------
# Helper: apply fixed services and event filter; return filtered df
# ------------------------
def apply_common_filters(df_in, events_selected):
    df_f = df_in.copy()
    df_f = df_f[df_f['service'].isin(FIXED_SERVICES)]
    if events_selected:
        df_f = df_f[df_f['event'].isin(events_selected)]
    return df_f

# ------------------------
# Build Dash app and layout
# ------------------------
app = dash.Dash(__name__)

all_events = sorted(df['event'].dropna().unique().tolist())
week_min = int(df['week'].min()) if 'week' in df.columns else 1
week_max = int(df['week'].max()) if 'week' in df.columns else 1

app.layout = html.Div([
    html.H1("Combined Dashboard — Shared Time-Granularity Control", style={'textAlign': 'center'}),

    # ---------- Top controls: common event filter + single shared time-granularity radio ----------
    html.Div([
        html.Div([
            html.Label("Common Event filter (applies to both diagrams):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='common-event-filter',
                options=[{'label': e, 'value': e} for e in all_events],
                value=all_events,
                multi=True,
                clearable=False,
                placeholder="Select events (include 'PCP' if needed)"
            )
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),

        html.Div([
            html.Label("Time Granularity (shared):", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='time-granularity',
                options=[
                    {'label': 'Week', 'value': 'weekly'},
                    {'label': 'Month', 'value': 'monthly'},
                    {'label': 'Quarter', 'value': 'quarterly'}
                ],
                value='weekly',
                inline=True
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'width': '98%', 'margin': '10px auto', 'padding': '12px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'backgroundColor': '#fafafa'}),

    html.Hr(),

    # ========== Diagram 1: Parcoords (NO week slider now) ==========
    html.Div([
        html.H3("Diagram 1 — Parcoords (Bed Capacity & Performance)"),
        html.Div(id='d1-graphs-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
    ], style={'width': '95%', 'margin': '10px auto', 'padding': '10px', 'border': '1px solid #eee', 'borderRadius': '8px'}),

    html.Hr(),

    # ========== Diagram 2: Small Multiples (Aggregation controlled by same granularity) ==========
    html.Div([
        html.H3("Diagram 2 — Small Multiples (Seasonal Metrics)"),
        # aggregation radio removed from here, it's at top now
        html.Div(id='d2-charts-container')
    ], style={'width': '95%', 'margin': '10px auto', 'padding': '10px', 'border': '1px solid #eee', 'borderRadius': '8px', 'marginBottom': '30px'})
])

# ------------------------
# Callbacks
# ------------------------

# Diagram 1 callback (Parcoords) — uses shared time-granularity for aggregation/filtering
@app.callback(
    Output('d1-graphs-container', 'children'),
    Input('common-event-filter', 'value'),
    Input('time-granularity', 'value')
)
def update_d1_graphs(events_selected, time_granularity):
    # apply fixed services and events
    df_filtered = apply_common_filters(df, events_selected)

    # If granularity is weekly, keep original rows (week-based), else aggregate to month or quarter
    if time_granularity == 'weekly':
        d1_df = df_filtered.copy()
        # ensure the columns needed exist
    elif time_granularity == 'monthly':
        # aggregate to monthly using same sum/mean rules as aggregate_line
        d1_grouped, _ = aggregate_line(df_filtered, agg='monthly', services=FIXED_SERVICES, events=events_selected)
        # rename time col consistently for slicing into parcoords
        d1_df = d1_grouped.copy()
    else:  # 'quarterly'
        d1_grouped, _ = aggregate_line(df_filtered, agg='quarterly', services=FIXED_SERVICES, events=events_selected)
        d1_df = d1_grouped.copy()

    graphs = []
    # iterate through fixed services (only show those with data)
    for service in FIXED_SERVICES:
        subset = d1_df[d1_df['service'] == service]
        if subset.empty:
            continue

        # For aggregated versions (monthly/quarterly), the columns present are the aggregated ones
        # For weekly/raw, original columns exist — both cases we use the same metrics_config_fixed columns
        dimensions = []
        for conf in metrics_config_fixed:
            col = conf['col']
            vals = subset[col] if col in subset.columns else pd.Series(dtype=float)
            if not vals.empty:
                rmin, rmax = float(vals.min()), float(vals.max())
                if rmin == rmax:
                    rmin -= 0.5
                    rmax += 0.5
                rng = [rmin, rmax]
            else:
                rng = conf['range']
            dimensions.append(dict(range=rng, label=conf['label'], values=vals))

        if not dimensions:
            continue

        # Parcoords: color uses status_id where available; if aggregated df lacks status_id, compute fallback
        color_vals = subset['status_id'] if 'status_id' in subset.columns else np.zeros(len(subset))

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=color_vals,
                colorscale=COLORSCALE,
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title='Load Status',
                    tickvals=[0.25, 0.75],
                    ticktext=['Sufficient', 'Shortage'],
                    len=0.6,
                    thickness=15
                )
            ),
            dimensions=dimensions
        ))

        fig.update_layout(
            title=f"<b>{service.replace('_', ' ').title()}</b>",
            height=400,
            margin=dict(l=50, r=50, t=60, b=20)
        )

        graphs.append(html.Div(dcc.Graph(figure=fig),
                               style={'width': '48%', 'minWidth': '500px', 'padding': '10px'}))

    if not graphs:
        return html.Div("No data to display for the chosen fixed services and events.", style={'padding': '20px'})

    return graphs

# Diagram 2 callback (Small Multiples) — aggregation uses time-granularity chosen at top
@app.callback(
    Output('d2-charts-container', 'children'),
    Input('time-granularity', 'value'),
    Input('common-event-filter', 'value'),
    Input({'type': 'd2-service-chart', 'index': dash.ALL}, 'clickData')
)
def update_d2_charts(time_granularity, events_selected, click_list):
    # apply fixed services + events first
    df_filtered = apply_common_filters(df, events_selected)

    # map granularity to aggregation arg for aggregate_line
    agg_map = {'weekly': 'weekly', 'monthly': 'monthly', 'quarterly': 'quarterly'}
    agg_choice = agg_map.get(time_granularity, 'weekly')

    grouped, time_col = aggregate_line(df_filtered, agg=agg_choice, services=FIXED_SERVICES, events=events_selected)

    if grouped.empty:
        return [html.Div("No data for selected filters.", style={'padding': '10px'})]

    global_min = grouped[time_col].min()
    global_max = grouped[time_col].max()
    services_sorted = sorted(grouped['service'].unique())

    candidate_metrics = ['patients_admitted', 'patient_satisfaction', 'staff_morale', 'patients_refused']
    valid_metrics = [m for m in candidate_metrics if m in grouped.columns]
    global_min_y, global_max_y = None, None
    if valid_metrics:
        try:
            global_min_y = grouped[valid_metrics].min().min()
            global_max_y = grouped[valid_metrics].max().max()
            if pd.isna(global_min_y) or pd.isna(global_max_y):
                global_min_y, global_max_y = None, None
        except Exception:
            global_min_y, global_max_y = None, None

    clicked_metric = None
    if click_list:
        for click in click_list:
            if click and "points" in click and click["points"]:
                p = click["points"][0]
                clicked_metric = p.get("data", {}).get("name") or clicked_metric
                break

    charts = []
    for s in services_sorted:
        sub = grouped[grouped['service'] == s]
        fig = go.Figure()

        for metric in valid_metrics:
            metric_name = metric.replace('_', ' ').title()
            if clicked_metric and clicked_metric == metric_name:
                opacity = 1.0
                line_width = 4
            else:
                opacity = 0.2 if clicked_metric else 1.0
                line_width = 2

            fig.add_trace(go.Scatter(
                x=sub[time_col],
                y=sub[metric],
                mode='lines+markers',
                name=metric_name,
                line=dict(width=line_width),
                opacity=opacity,
                hovertemplate="%{y}<extra></extra>"
            ))

        fig.update_layout(
        title=f"{s} — Selected Metrics",
        height=300,
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis_title=time_col.replace('_', ' ').title(),
        yaxis_title="Value",
        xaxis=dict(range=[global_min, global_max]),
        yaxis=dict(autorange=True)
)


        charts.append(html.Div([dcc.Graph(
            id={'type': 'd2-service-chart', 'index': s},
            figure=fig,
            clear_on_unhover=True
        )], style={'width': '100%', 'border': '1px solid #ddd', 'margin-bottom': '10px'}))

    return charts

# ------------------------
# Run server
# ------------------------
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
