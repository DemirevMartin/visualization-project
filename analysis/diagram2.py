# task2_seasonal_dashboard.py
import pandas as pd
import numpy as np
from collections import Counter

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ------------------------
# Config / Data load
# ------------------------
CSV_PATH = "./data/services_weekly.csv"  # adjust if needed

df = pd.read_csv(CSV_PATH)

# Basic preprocessing / types
df['week'] = df['week'].astype(int)
if 'month' in df.columns:
    df['month'] = df['month'].astype(int)
else:
    # derive month from week by approximate mapping (optional)
    df['month'] = ((df['week'] - 1) // 4 + 1).clip(1,12)

# ensure numeric fields
for col in ['patients_admitted', 'patient_satisfaction', 'staff_morale']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# create quarter
df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)

# unique domain lists
all_services = sorted(df['service'].dropna().unique().tolist())
all_events = sorted(df['event'].dropna().unique().tolist())

# ------------------------
# Helper: aggregate into grid
# ------------------------
def aggregate_grid(df_in, agg='weekly', metric='patients_admitted', services=None, events=None):
    """
    Returns:
      z: 2D array values (services x time_bins)
      x_labels: list of time labels (weeks/months/quarters)
      y_labels: list of services (rows)
      custom: 2D array of dicts for hover (same shape)
    """
    df_f = df_in.copy()
    if services:
        df_f = df_f[df_f['service'].isin(services)]
    if events:
        df_f = df_f[df_f['event'].isin(events)]

    if agg == 'weekly':
        time_col = 'week'
        x_bins = sorted(df_f['week'].unique())
        x_labels = [f"W{int(w)}" for w in x_bins]
        group_cols = ['service', 'week']
    elif agg == 'monthly':
        time_col = 'month'
        x_bins = list(range(1,13))
        x_labels = [f"M{m}" for m in x_bins]
        group_cols = ['service', 'month']
    elif agg == 'quarterly':
        time_col = 'quarter'
        x_bins = [1,2,3,4]
        x_labels = [f"Q{q}" for q in x_bins]
        group_cols = ['service', 'quarter']
    else:
        raise ValueError("agg must be weekly/monthly/quarterly")

    # compute aggregated metrics
    agg_funcs = {
        metric: 'sum' if metric.endswith('_admitted') or metric.endswith('_request') or metric.endswith('_refused') else 'mean',
        'staff_morale': 'mean',
        'event_list': lambda s: list(s)
    }

    grouped = df_f.groupby(group_cols).agg(
        metric_val=(metric, agg_funcs[metric]),
        staff_morale_avg=('staff_morale', 'mean'),
        event_list=('event', lambda s: list(s))
    ).reset_index()

    # prepare matrices
    y_labels = services if services else sorted(df_in['service'].unique().tolist())
    z = []
    custom = []

    for serv in y_labels:
        row = []
        rowc = []
        for tb in x_bins:
            cell = grouped[(grouped['service'] == serv) & (grouped[group_cols[1]] == tb)]
            if not cell.empty:
                val = cell['metric_val'].values[0]
                morale = cell['staff_morale_avg'].values[0]
                events_here = cell['event_list'].values[0]
                most_common_event = Counter(events_here).most_common(1)[0][0] if events_here else 'none'
                row.append(val if (pd.notna(val)) else np.nan)
                rowc.append({'staff_morale': float(morale) if pd.notna(morale) else None,
                             'most_event': most_common_event,
                             'service': serv, 'timebin': tb})
            else:
                row.append(np.nan)
                rowc.append({'staff_morale': None, 'most_event': None, 'service': serv, 'timebin': tb})
        z.append(row)
        custom.append(rowc)

    return np.array(z), x_labels, y_labels, custom

# ------------------------
# Build Dash app
# ------------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H3("Seasonal Patient Load & Satisfaction — Heatmap Calendar (Task 2)"),
    html.Div([
        html.Div([
            html.Label("Aggregation:"),
            dcc.RadioItems(id='agg', options=[
                {'label':'Weekly','value':'weekly'},
                {'label':'Monthly','value':'monthly'},
                {'label':'Quarterly','value':'quarterly'}
            ], value='weekly', inline=True)
        ], style={'display':'inline-block','margin-right':'30px'}),

        html.Div([
            html.Label("Metric:"),
            dcc.RadioItems(id='metric', options=[
                {'label':'Patients Admitted','value':'patients_admitted'},
                {'label':'Patient Satisfaction (avg)','value':'patient_satisfaction'}
            ], value='patients_admitted', inline=True)
        ], style={'display':'inline-block','margin-right':'30px'}),

        html.Div([
            html.Label("Service filter:"),
            dcc.Dropdown(id='service-filter', options=[{'label':s,'value':s} for s in all_services],
                         value=all_services, multi=True, placeholder="Select services")
        ], style={'width':'35%','display':'inline-block', 'verticalAlign':'top'})
    ], style={'padding':'8px','border':'1px solid #ddd','margin-bottom':'10px'}),

    html.Div([
        html.Label("Event filter:"),
        dcc.Dropdown(id='event-filter', options=[{'label':e,'value':e} for e in all_events],
                     value=all_events, multi=True, placeholder="Select events (e.g., flu)")
    ], style={'width':'50%','padding-bottom':'12px'}),

    dcc.Graph(id='heatmap', config={'displayModeBar': True}),
    html.Div(id='explain', style={'fontSize':12, 'color':'#555', 'marginTop':'6px'})
])

# ------------------------
# Callbacks
# ------------------------
@app.callback(
    Output('heatmap', 'figure'),
    Output('explain', 'children'),
    Input('agg', 'value'),
    Input('metric', 'value'),
    Input('service-filter', 'value'),
    Input('event-filter', 'value')
)
def update_heatmap(agg, metric, services_sel, events_sel):
    z, x_labels, y_labels, custom = aggregate_grid(df, agg=agg, metric=metric, services=services_sel, events=events_sel)

    # Build hover text from customdata
    # custom is shape (n_services, n_timebins) with dicts
    custom_text = []
    for ridx, row in enumerate(custom):
        crow = []
        for c in row:
            mor = c['staff_morale']
            ev = c['most_event']
            timebin = c['timebin']
            if mor is None and ev is None:
                txt = f"Service: {c['service']}<br>Time: {timebin}<br>No data"
            else:
                txt = f"Service: {c['service']}<br>Time: {timebin}<br>Staff morale (avg): {mor:.1f}<br>Most frequent event: {ev}"
            crow.append(txt)
        custom_text.append(crow)

    # create Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        text=custom_text,
        hovertemplate="%{text}<br><br>"+f"{metric}: "+"%{z}<extra></extra>",
        colorscale='Viridis',
        colorbar=dict(title=metric.replace('_',' ').title())
    ))

    fig.update_layout(
        title=f"Heatmap ({metric.replace('_',' ').title()}) aggregated by {agg}",
        xaxis_title="Time",
        yaxis_title="Service",
        yaxis_autorange='reversed',
        height=600
    )

    explain = f"Showing metric='{metric}' aggregated='{agg}'. Hover a cell for avg staff_morale and the most frequent event. Filters: {len(services_sel)} services, {len(events_sel)} events."
    return fig, explain

# ------------------------
# Run
# ------------------------
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
