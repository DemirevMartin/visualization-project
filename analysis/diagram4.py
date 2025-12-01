# task4_staff_allocation_load
import pandas as pd
import numpy as np
from collections import Counter

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ------------------------
# Config / Data load
# ------------------------
services_df = pd.read_csv("C:/Users/msmir/Desktop/Uni_TUE/Visualization/VisualizationProject/data/services_weekly.csv")
staff_schedule_df = pd.read_csv("C:/Users/msmir/Desktop/Uni_TUE/Visualization/VisualizationProject/data/staff_schedule.csv")

staff_schedule_present_df = staff_schedule_df[staff_schedule_df['present'] == 1]

df_agg_counts = (
    staff_schedule_present_df
    .groupby(['week', 'service', 'role'])['staff_id']
    .nunique()
    .reset_index(name='count')
)

df_role_counts = df_agg_counts.pivot_table(
    index=['week', 'service'],
    columns='role',
    values='count',
    fill_value=0
).reset_index()

df_role_counts = df_role_counts.rename(
    columns={
        col: f'{col}'
        for col in df_role_counts.columns
        if col not in ['week', 'service']
    }
)

df_role_counts = df_role_counts.rename(
    columns = {"nursing_assistant" : "nursing assistant"}
)

df_final = services_df.merge(
    df_role_counts,
    on=['week', 'service'],
    how='left'
)

# ------------------------
# Dropdown
# ------------------------

app = dash.Dash(__name__)

service_options = [{'label': s.capitalize(), 'value': s} for s in sorted(df_final['service'].unique())]
event_options = [{'label': e.capitalize(), 'value': e} for e in sorted(df_final['event'].unique())]

app.layout = html.Div([
    html.H1("Staff Allocation vs Patient Load"),
    
    html.Div([
        html.Label("Select Service:"),
        dcc.Dropdown(
            id='service-dropdown',
            options=service_options,
            value='emergency',
            clearable=False,
            style={"width": "300px"}
        )
    ], style={'display': 'inline-block', 'margin-right': '20px'}),

    html.Div([
        html.Label("Select Event:"),
        dcc.Dropdown(
            id='event-dropdown',
            options=event_options,
            value=None,
            multi=True,        # Allow selecting multiple events
            placeholder="All events",
            style={"width": "300px"}
        )
    ], style={'display': 'inline-block'}),

    dcc.Graph(id='staff-vs-patients')
])


# ----------------------------------------------------
# Callback
# ----------------------------------------------------
@app.callback(
    Output('staff-vs-patients', 'figure'),
    Input('service-dropdown', 'value'),
    Input('event-dropdown', 'value')
)

# ----------------------------------------------------
# Figure
# ----------------------------------------------------

def figure4(selected_service, selected_events):

    # Filter by service
    df_srv = df_final[df_final['service'] == selected_service]

    # Filter by events if selected
    if selected_events:
        df_srv = df_srv[df_srv['event'].isin(selected_events)]

    df_srv = df_srv.sort_values('week')

    roles_to_plot = ['doctor', 'nurse', 'nursing assistant']  
    colors = ['#636EFA', '#EF553B', '#00CC96']

    fig = go.Figure()

    # Add stacked bars
    for i, role in enumerate(roles_to_plot):
        if role in df_srv.columns:
            fig.add_trace(go.Bar(
                x=df_srv['week'],
                y=df_srv[role],
                name=role.capitalize(),
                marker_color=colors[i % len(colors)],
                hovertemplate='Week %{x}<br>%{y} ' + role + 's<extra></extra>'
            ))

    # Line for patients admitted
    fig.add_trace(go.Scatter(
        x=df_srv['week'],
        y=df_srv['patients_admitted'],
        name='Patients Admitted',
        mode='lines+markers',
        line=dict(color='black', width=3),
        yaxis='y2',
        hovertemplate='Week %{x}<br>%{y} Patients Admitted<extra></extra>'
    ))

    fig.update_layout(
        title=f"Staff Allocation vs Patient Load ({selected_service.capitalize()})",
        xaxis_title="Week",
        yaxis=dict(title="Staff Count"),
        yaxis2=dict(title="Patients Admitted", overlaying='y', side='right'),
        barmode='stack',
        legend=dict(x=1.1, y=1)
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)