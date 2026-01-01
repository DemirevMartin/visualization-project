# task4_staff_allocation_load
import pandas as pd
import numpy as np
from collections import Counter

import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------
# Config / Data load
# ------------------------
services_df = pd.read_csv("data/services_weekly.csv")
staff_schedule_df = pd.read_csv("data/staff_schedule.csv")

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
    html.Label("View Mode:"),
    dcc.RadioItems(
        id='view-mode',
        options=[
            {'label': 'By Role (single service)', 'value': 'role'},
            {'label': 'By Service (comparison)', 'value': 'service'}
        ],
        value='role',
        inline=True
    )
    ], style={'margin-bottom': '10px'}),

    html.Div([
        html.Label("Select Service(s):"),
        dcc.Dropdown(
    id='service-dropdown',
        options=service_options,
        value=['emergency'],
        multi=True,
        clearable=False,
        style={"width": "400px"}
        )
    ], style={'margin-right': '20px'}),

    html.Div([
        html.Label("Select Event:"),
        dcc.Dropdown(
            id='event-dropdown',
            options=event_options,
            value=None,
            multi=True,
            placeholder="All events",
            style={"width": "300px"}
        )
    ], style={'display': 'inline-block'}),

    html.Div([
        html.Label("Select Week Range:"),
        dcc.RangeSlider(
            id='week-slider',
            min=df_final['week'].min(),
            max=df_final['week'].max(),
            step=1,
            value=[
                df_final['week'].min(),
                df_final['week'].max()
            ],
            marks={int(i): str(int(i)) for i in df_final['week'].unique()},
            tooltip={"placement": "bottom", "always_visible": False}
        )
    ], style={'margin-top': '20px'}),

    dcc.Graph(
        id='staff-vs-patients',
        clear_on_unhover=True
    )
])


# ----------------------------------------------------
# Callback
# ----------------------------------------------------

@app.callback(
    Output('service-dropdown', 'multi'),
    Output('service-dropdown', 'value'),
    Input('view-mode', 'value')
)
def update_service_dropdown(view_mode):
    if view_mode == 'service':
        return True, ['emergency']  
    
@app.callback(
    Output('staff-vs-patients', 'figure'),
    Input('view-mode', 'value'),
    Input('service-dropdown', 'value'),
    Input('event-dropdown', 'value'),
    Input('week-slider', 'value'),
    Input('staff-vs-patients', 'hoverData')
)

# ----------------------------------------------------
# Figure
# ----------------------------------------------------

def figure4(view_mode, selected_services, selected_events, selected_weeks, hoverData):

    if isinstance(selected_services, str):
        selected_services = [selected_services]

    week_start, week_end = selected_weeks

    df = df_final[
        (df_final['week'] >= week_start) &
        (df_final['week'] <= week_end) &
        (df_final['service'].isin(selected_services))
    ]

    if selected_events:
        df = df[df['event'].isin(selected_events)]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Staff Allocation",
            "Patients Admitted"
        )
    )

    # =========================
    # VIEW 1 — BY ROLE
    # =========================
    if view_mode == 'role':

    # Aggregate across selected services
        df_agg = (
            df
            .groupby('week', as_index=False)
            .agg({
                'doctor': 'sum',
                'nurse': 'sum',
                'nursing assistant': 'sum',
                'patients_admitted': 'sum'
            })
            .sort_values('week')
        )

        roles = ['doctor', 'nurse', 'nursing assistant']
        colors = ['#636EFA', '#EF553B', '#00CC96']

        # --- Staff stacked by role ---
        for i, role in enumerate(roles):
            fig.add_trace(
                go.Bar(
                    x=df_agg['week'],
                    y=df_agg[role],
                    name=role.capitalize(),
                    marker_color=colors[i]
                ),
                row=1,
                col=1
            )

        # --- Patients (aggregated) ---
        fig.add_trace(
            go.Scatter(
                x=df_agg['week'],
                y=df_agg['patients_admitted'],
                name='Patients Admitted (All Services)',
                mode='lines+markers',
                line=dict(color='black', width=3)
            ),
            row=2,
            col=1
        )

        title_suffix = " + ".join([s.capitalize() for s in selected_services])

    # =========================
    # VIEW 2 — BY SERVICE
    # =========================
    else:
        services = selected_services
        colors = px.colors.qualitative.Set2

        # --- Staff (stacked by service) ---
        for i, srv in enumerate(services):
            df_srv = df[df['service'] == srv].sort_values('week')
            staff_total = (
                df_srv['doctor'] +
                df_srv['nurse'] +
                df_srv['nursing assistant']
            )

            fig.add_trace(
                go.Bar(
                    x=df_srv['week'],
                    y=staff_total,
                    name=srv.capitalize(),
                    marker_color=colors[i % len(colors)]
                ),
                row=1,
                col=1
            )

            # --- Patients lines ---
            fig.add_trace(
                go.Scatter(
                    x=df_srv['week'],
                    y=df_srv['patients_admitted'],
                    name=f"{srv.capitalize()} Patients",
                    mode='lines+markers'
                ),
                row=2,
                col=1
            )

        title_suffix = " vs ".join([s.capitalize() for s in services])

    # =========================
    # Layout
    # =========================
    fig.update_layout(
        title=f"Staff Allocation and Patient Load ({title_suffix})",
        barmode='stack',
        height=750,
        legend=dict(orientation="h", y=1.05)
    )

    fig.update_xaxes(title_text="Week", row=2, col=1)
    fig.update_yaxes(title_text="Staff Count", row=1, col=1)
    fig.update_yaxes(title_text="Patients Admitted", row=2, col=1)

    # Hover highlight
    if hoverData and 'points' in hoverData:
        week = hoverData['points'][0]['x']
        fig.add_vrect(
            x0=week - 0.5,
            x1=week + 0.5,
            fillcolor="rgba(200,200,200,0.3)",
            line_width=0,
            layer="below"
        )

    return fig


if __name__ == '__main__':
    app.run(debug=True)