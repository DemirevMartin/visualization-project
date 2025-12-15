# dashboard_merged_full.py
# Full merged Dash app:
# - Restored shortage logic (aggregated + non-aggregated)
# - Global PCP filters with uniform slider divisions and fixed min/max display
# - Diagram 2 global brushing where clicking an empty area clears the brush (Option B)

import math
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

# ------------------------
# Shared Config / Data load
# ------------------------
CSV_PATH = "../data/services_weekly.csv"  # adjust path if needed
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

numeric_cols = ['patients_admitted', 'patient_satisfaction', 'staff_morale', 'patients_refused',
                'patients_request', 'available_beds']
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
# compute row-level availability_status if missing (used for weekly view)
if 'availability_status' not in df.columns:
    df['availability_status'] = np.where(
        df.get('patients_request', 0) > df.get('available_beds', 0),
        'Shortage',
        'Sufficient'
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
# Includes patients_request, available_beds so shortages survive aggregation.
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
        patients_refused=('patients_refused', 'sum'),

        # bring these back so shortages remain visible after aggregation
        patients_request=('patients_request', 'sum'),
        available_beds=('available_beds', 'sum')
    ).reset_index()

    # restore your SHORTAGE logic at aggregated level
    grouped["availability_status"] = np.where(
        grouped["patients_request"].fillna(0) > grouped["available_beds"].fillna(0),
        "Shortage",
        "Sufficient"
    )
    grouped["status_id"] = grouped["availability_status"].map({'Sufficient':0, 'Shortage':1})

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
# Helper: apply metric slider filters to grouped dataframe (post-aggregation)
# ------------------------
def apply_metric_filters_to_grouped(grouped_df, sat_range, morale_range, admitted_range, refused_range):
    g = grouped_df.copy()
    if sat_range is not None and 'patient_satisfaction' in g.columns:
        g = g[(g['patient_satisfaction'] >= sat_range[0]) & (g['patient_satisfaction'] <= sat_range[1])]
    if morale_range is not None and 'staff_morale' in g.columns:
        g = g[(g['staff_morale'] >= morale_range[0]) & (g['staff_morale'] <= morale_range[1])]
    if admitted_range is not None and 'patients_admitted' in g.columns:
        g = g[(g['patients_admitted'] >= admitted_range[0]) & (g['patients_admitted'] <= admitted_range[1])]
    if refused_range is not None and 'patients_refused' in g.columns:
        g = g[(g['patients_refused'] >= refused_range[0]) & (g['patients_refused'] <= refused_range[1])]
    return g

# ------------------------
# Helper: make uniform marks (n_divisions) for sliders
# ------------------------
def uniform_marks(min_v, max_v, divisions=4):
    # produce divisions+1 marks including min and max
    if min_v == max_v:
        return {int(min_v): str(int(min_v))}
    step = (max_v - min_v) / divisions
    marks = {}
    for i in range(divisions + 1):
        val = min_v + step * i
        ival = int(round(val))
        marks[ival] = str(ival)
    # ensure both min and max present exactly
    marks[int(min_v)] = str(int(min_v))
    marks[int(max_v)] = str(int(max_v))
    # sort keys
    ordered = {k: marks[k] for k in sorted(marks.keys())}
    return ordered

# ------------------------
# Build Dash app and layout
# ------------------------
app = dash.Dash(__name__)
server = app.server

all_events = sorted(df['event'].dropna().unique().tolist())
week_min = int(df['week'].min()) if 'week' in df.columns else 1
week_max = int(df['week'].max()) if 'week' in df.columns else 1

# compute slider ranges from raw data (used for slider bounds)
sat_min, sat_max = (0, 100)
if 'patient_satisfaction' in df.columns and not df['patient_satisfaction'].isna().all():
    sat_min = int(np.nanmin(df['patient_satisfaction']))
    sat_max = int(np.nanmax(df['patient_satisfaction']))

mor_min, mor_max = (0, 100)
if 'staff_morale' in df.columns and not df['staff_morale'].isna().all():
    mor_min = int(np.nanmin(df['staff_morale']))
    mor_max = int(np.nanmax(df['staff_morale']))

adm_min, adm_max = (0, int(df['patients_admitted'].max() if 'patients_admitted' in df.columns else 1))
ref_min, ref_max = (0, int(df['patients_refused'].max() if 'patients_refused' in df.columns else 1))

app.layout = html.Div([
    html.H1("Combined Dashboard — Shared Time-Granularity, Shortages, and Linked Brushing", style={'textAlign': 'center'}),

    # Store for global brushed metric in Diagram 2
    dcc.Store(id='global-metric-brush', data=None),

    # ---------- Top controls: common event filter + single shared time-granularity radio ----------
    html.Div([
        # event + granularity column
        html.Div([
            html.Label("Common Event filter (applies to both diagrams):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='common-event-filter',
                options=[{'label': e, 'value': e} for e in all_events],
                value=all_events,
                multi=True,
                clearable=False,
                placeholder="Select events (include 'PCP' if needed')"
            ),
            html.Br(),
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
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),

        # global slider column
        html.Div([
            html.Label("Global PCP Filters (applies to all services):", style={'fontWeight': 'bold'}),
            html.Div([
                html.Label("Patient Satisfaction"),
                dcc.RangeSlider(
                    id='slider-satisfaction',
                    min=sat_min, max=sat_max, step=1,
                    value=[sat_min, sat_max],
                    marks=uniform_marks(sat_min, sat_max, divisions=4)
                ),
                html.Br(),

                html.Label("Staff Morale"),
                dcc.RangeSlider(
                    id='slider-morale',
                    min=mor_min, max=mor_max, step=1,
                    value=[mor_min, mor_max],
                    marks=uniform_marks(mor_min, mor_max, divisions=4)
                ),
                html.Br(),

                html.Label("Patients Admitted"),
                dcc.RangeSlider(
                    id='slider-admitted',
                    min=adm_min, max=adm_max, step=max(1, (adm_max - adm_min)//100 or 1),
                    value=[adm_min, adm_max],
                    marks=uniform_marks(adm_min, adm_max, divisions=4)
                ),
                html.Br(),

                html.Label("Patients Refused"),
                dcc.RangeSlider(
                    id='slider-refused',
                    min=ref_min, max=ref_max, step=max(1, (ref_max - ref_min)//100 or 1),
                    value=[ref_min, ref_max],
                    marks=uniform_marks(ref_min, ref_max, divisions=4)
                ),
                html.Br(),

                html.Button("Reset Filters", id='reset-filters', n_clicks=0, style={'marginTop': '6px'})
            ], style={'padding': '8px', 'border': '1px solid #eee', 'borderRadius': '6px', 'backgroundColor': '#fff'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'width': '98%', 'margin': '10px auto', 'padding': '12px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'backgroundColor': '#fafafa'}),

    html.Hr(),

    # ========== Diagram 1: Parcoords (PCP) ==========
    html.Div([
        html.H3("Diagram 1 — Parcoords (Bed Capacity & Performance)"),
        html.Div(id='d1-graphs-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
    ], style={'width': '95%', 'margin': '10px auto', 'padding': '10px', 'border': '1px solid #eee', 'borderRadius': '8px'}),

    html.Hr(),

    # ========== Diagram 2: Small Multiples ==========
    html.Div([
        html.H3("Diagram 2 — Small Multiples (Seasonal Metrics)"),
        html.Div(id='d2-charts-container')
    ], style={'width': '95%', 'margin': '10px auto', 'padding': '10px', 'border': '1px solid #eee', 'borderRadius': '8px', 'marginBottom': '30px'})
])

# ------------------------
# Callbacks
# ------------------------

# Reset filters button sets sliders back to full range
@app.callback(
    Output('slider-satisfaction', 'value'),
    Output('slider-morale', 'value'),
    Output('slider-admitted', 'value'),
    Output('slider-refused', 'value'),
    Input('reset-filters', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    # reset to computed global bounds
    return [sat_min, sat_max], [mor_min, mor_max], [adm_min, adm_max], [ref_min, ref_max]

# ------------------------
# Global brushing state updater for Diagram 2
# Option B behavior: clicking empty space clears the brush.
# ------------------------
@app.callback(
    Output('global-metric-brush', 'data'),
    Input({'type': 'd2-service-chart', 'index': dash.ALL}, 'clickData'),
    State('global-metric-brush', 'data')
)
def update_global_brush(click_list, stored_metric):
    """
    Called when any of the small-multiples charts receive a click.
    Behavior (Option B): If a real line is clicked, set that metric as the brush.
    If a click results in no points (i.e., empty space) across all charts, clear the brush (None).
    """
    # If click_list is None or empty, do nothing (keep previous)
    if click_list is None:
        return stored_metric

    # If any chart clickData contains a point, pick that metric (first found)
    for click in click_list:
        if click and "points" in click and click["points"]:
            try:
                metric_name = click["points"][0]["data"]["name"]
                return metric_name
            except Exception:
                continue

    # If we reach here, click_list existed but none had points -> this is an "empty space click"
    # Clear the stored brush
    return None

# Diagram 1 callback (Parcoords) — uses shared time-granularity for aggregation/filtering
@app.callback(
    Output('d1-graphs-container', 'children'),
    Input('common-event-filter', 'value'),
    Input('time-granularity', 'value'),
    Input('slider-satisfaction', 'value'),
    Input('slider-morale', 'value'),
    Input('slider-admitted', 'value'),
    Input('slider-refused', 'value')
)
def update_d1_graphs(events_selected, time_granularity,
                     sat_range, morale_range, admitted_range, refused_range):
    # apply fixed services and events
    df_filtered = apply_common_filters(df, events_selected)

    # aggregate according to granularity (aggregate_line now includes request/beds + status_id)
    if time_granularity == 'weekly':
        d1_grouped, time_col = aggregate_line(df_filtered, agg='weekly', services=FIXED_SERVICES, events=events_selected)
        # Note: weekly aggregation will effectively be same as rows grouped by week/service
    elif time_granularity == 'monthly':
        d1_grouped, time_col = aggregate_line(df_filtered, agg='monthly', services=FIXED_SERVICES, events=events_selected)
    else:  # quarterly
        d1_grouped, time_col = aggregate_line(df_filtered, agg='quarterly', services=FIXED_SERVICES, events=events_selected)

    # apply metric filters to aggregated grouped df (this ensures the plotted lines obey slider filters)
    d1_df = apply_metric_filters_to_grouped(d1_grouped, sat_range, morale_range, admitted_range, refused_range)

    graphs = []
    for service in FIXED_SERVICES:
        subset = d1_df[d1_df['service'] == service]
        if subset.empty:
            continue

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

        # status_id always present in aggregated df due to aggregate_line
        color_vals = subset['status_id']

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
        return html.Div("No data to display for the chosen fixed services and filters.", style={'padding': '20px'})

    return graphs

# Diagram 2 callback (Small Multiples) — aggregation uses time-granularity chosen at top
# Now uses the stored global brush state to highlight the chosen metric across all panels.
@app.callback(
    Output('d2-charts-container', 'children'),
    Input('time-granularity', 'value'),
    Input('common-event-filter', 'value'),
    Input('slider-satisfaction', 'value'),
    Input('slider-morale', 'value'),
    Input('slider-admitted', 'value'),
    Input('slider-refused', 'value'),
    Input('global-metric-brush', 'data')
)
def update_d2_charts(time_granularity, events_selected,
                     sat_range, morale_range, admitted_range, refused_range,
                     global_brush_state):
    # apply fixed services + events first
    df_filtered = apply_common_filters(df, events_selected)

    # map granularity to aggregation arg for aggregate_line
    agg_map = {'weekly': 'weekly', 'monthly': 'monthly', 'quarterly': 'quarterly'}
    agg_choice = agg_map.get(time_granularity, 'weekly')

    grouped, time_col = aggregate_line(df_filtered, agg=agg_choice, services=FIXED_SERVICES, events=events_selected)

    # apply metric slider filters to grouped
    grouped = apply_metric_filters_to_grouped(grouped, sat_range, morale_range, admitted_range, refused_range)

    if grouped.empty:
        return [html.Div("No data for selected filters.", style={'padding': '10px'})]

    global_min = grouped[time_col].min()
    global_max = grouped[time_col].max()
    services_sorted = sorted(grouped['service'].unique())

    candidate_metrics = ['patients_admitted', 'patient_satisfaction', 'staff_morale', 'patients_refused']
    valid_metrics = [m for m in candidate_metrics if m in grouped.columns]

    clicked_metric = global_brush_state  # the stored metric name, e.g. "Patient Satisfaction"

    charts = []
    for s in services_sorted:
        sub = grouped[grouped['service'] == s]
        if sub.empty:
            continue

        fig = go.Figure()
        for metric in valid_metrics:
            metric_name = metric.replace('_', ' ').title()
            if clicked_metric and clicked_metric == metric_name:
                opacity = 1.0
                line_width = 4
            else:
                opacity = 0.15 if clicked_metric else 1.0
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
            yaxis=dict(autorange=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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
