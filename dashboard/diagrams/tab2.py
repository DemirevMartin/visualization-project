from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

# ===========================
# CONSTANTS
# ===========================
SERVICE_COLORS = {
    'emergency': '#d62728',
    'ICU': '#ff7f0e',
    'surgery': '#2ca02c',
    'general_medicine': '#1f77b4'
}

SERVICE_LABELS = {
    'emergency': 'Emergency',
    'ICU': 'ICU',
    'surgery': 'Surgery',
    'general_medicine': 'General Medicine'
}

# ===========================
# HELPER FUNCTIONS
# ===========================
def create_view1(df, selected_weeks=None, selected_services=None):
    """4-panel view with week/service linking"""
    
    # We work on a copy to avoid SettingWithCopy on the original df
    df = df.copy()

    if selected_weeks is not None and len(selected_weeks) > 0:
        df['week_selected'] = df['week'].isin(selected_weeks)
    else:
        df['week_selected'] = True

    if selected_services is not None and len(selected_services) > 0:
        df['service_selected'] = df['service'].isin(selected_services)
    else:
        df['service_selected'] = True

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Staff Presence vs Satisfaction', 'Workload Pressure vs Satisfaction',
                       'Department Performance', 'Service Overview'],
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "scatter"}]],
        vertical_spacing=0.18, horizontal_spacing=0.15
    )

    # TOP-LEFT: Presence vs Satisfaction
    for service, color in SERVICE_COLORS.items():
        service_data = df[df['service'] == service].copy()
        if len(service_data) == 0:
            continue

        selected = service_data[service_data['week_selected']]
        not_selected = service_data[~service_data['week_selected']]

        if len(not_selected) > 0:
            fig.add_trace(
                go.Scatter(
                    x=not_selected['presence_rate'] * 100, y=not_selected['patient_satisfaction'],
                    mode='markers', name=SERVICE_LABELS[service],
                    marker=dict(color=color, size=6, opacity=0.2),
                    customdata=np.column_stack((not_selected['week'], not_selected['service'])),
                    hovertemplate='<b>%{customdata[1]}</b><br>Week: %{customdata[0]}<br>Presence: %{x:.1f}%<br>Satisfaction: %{y:.1f}<extra></extra>',
                    legendgroup=service, showlegend=False
                ),
                row=1, col=1
            )
        
        if len(selected) > 0:
            fig.add_trace(
                go.Scatter(
                    x=selected['presence_rate'] * 100, y=selected['patient_satisfaction'],
                    mode='markers', name=SERVICE_LABELS[service],
                    marker=dict(color=color, size=14, opacity=1.0, line=dict(width=2, color='white')),
                    customdata=np.column_stack((selected['week'], selected['service'])),
                    hovertemplate='<b>%{customdata[1]}</b><br>Week: %{customdata[0]}<br>Presence: %{x:.1f}%<br>Satisfaction: %{y:.1f}<extra></extra>',
                    legendgroup=service, showlegend=True
                ),
                row=1, col=1
            )

    # TOP-RIGHT: Workload vs Satisfaction
    if 'patients_per_staff' in df.columns:
        workload_data = df[df['patients_per_staff'] < 15].copy()
        for service, color in SERVICE_COLORS.items():
            service_data = workload_data[workload_data['service'] == service]
            if len(service_data) == 0:
                continue
            
            selected = service_data[service_data['week_selected']]
            not_selected = service_data[~service_data['week_selected']]
            
            if len(not_selected) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=not_selected['patients_per_staff'], y=not_selected['patient_satisfaction'],
                        mode='markers', marker=dict(color=color, size=6, opacity=0.2),
                        customdata=np.column_stack((not_selected['week'], not_selected['service'])),
                        hovertemplate='<b>%{customdata[1]}</b><br>Week: %{customdata[0]}<br>Workload: %{x:.1f}<br>Sat: %{y:.1f}<extra></extra>',
                        legendgroup=service, showlegend=False
                    ),
                    row=1, col=2
                )
            
            if len(selected) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=selected['patients_per_staff'], y=selected['patient_satisfaction'],
                        mode='markers', marker=dict(color=color, size=14, opacity=1.0, line=dict(width=2, color='white')),
                        customdata=np.column_stack((selected['week'], selected['service'])),
                        hovertemplate='<b>%{customdata[1]}</b><br>Week: %{customdata[0]}<br>Workload: %{x:.1f}<br>Sat: %{y:.1f}<extra></extra>',
                        legendgroup=service, showlegend=False
                    ),
                    row=1, col=2
                )

    # BOTTOM-LEFT: Department Performance
    # Use 'avg_satisfaction' if available (from loader), or compute if missing
    if 'avg_satisfaction' in df.columns:
        dept_avg = df.groupby('service').agg({'patient_satisfaction': 'mean', 'avg_satisfaction': 'mean'}).reset_index()
    else:
         dept_avg = df.groupby('service').agg({'patient_satisfaction': 'mean'}).reset_index()
         dept_avg['avg_satisfaction'] = dept_avg['patient_satisfaction']
         
    dept_avg.columns = ['service', 'service_satisfaction', 'patient_satisfaction']
    dept_avg = dept_avg.sort_values('service_satisfaction', ascending=True)
    
    for _, row in dept_avg.iterrows():
        service = row['service']
        color = SERVICE_COLORS.get(service, '#7f8c8d')
        is_selected = (selected_services is None or len(selected_services) == 0 or service in selected_services)
        service_label = SERVICE_LABELS.get(service, service.replace('_', ' ').title())
        
        fig.add_trace(
            go.Bar(
                y=[service_label], x=[row['service_satisfaction']], orientation='h',
                marker=dict(color=color, opacity=1.0 if is_selected else 0.3),
                text=[f"{row['service_satisfaction']:.1f}"], textposition='inside', textfont=dict(size=12, color='white'),
                legendgroup=service, showlegend=False,
                hovertemplate=f'<b>{service_label}</b><br>Avg: %{{x:.1f}}<extra></extra>',
                customdata=[[service]]
            ),
            row=2, col=1
        )

    # BOTTOM-RIGHT: Service Overview
    service_summary = df.groupby('service').agg({
        'presence_rate': 'mean', 'patient_satisfaction': 'mean', 'patients_per_staff': 'mean'
    }).reset_index()

    for _, row in service_summary.iterrows():
        service = row['service']
        color = SERVICE_COLORS.get(service, '#7f8c8d')
        is_selected = (selected_services is None or len(selected_services) == 0 or service in selected_services)
        bubble_size = 20 + (row['patients_per_staff'] * 3)
        
        fig.add_trace(
            go.Scatter(
                x=[row['presence_rate'] * 100], y=[row['patient_satisfaction']],
                mode='markers+text',
                marker=dict(color=color, size=bubble_size if is_selected else bubble_size * 0.5, opacity=1.0 if is_selected else 0.3),
                text=[SERVICE_LABELS.get(service, service)] if is_selected else [''],
                textposition='top center', textfont=dict(size=13, color=color),
                legendgroup=service, showlegend=False,
                hovertemplate=f'<b>{SERVICE_LABELS.get(service, service)}</b><br>Presence: %{{x:.1f}}%<br>Satisfaction: %{{y:.1f}}<extra></extra>',
                customdata=[[service]]
            ),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Staff Presence Rate (%)", row=1, col=1, range=[0, 105])
    fig.update_xaxes(title_text="Patients per Staff", row=1, col=2, range=[0, 15])
    fig.update_xaxes(title_text="Average Satisfaction", row=2, col=1, range=[0, 105])
    fig.update_xaxes(title_text="Average Presence Rate (%)", row=2, col=2)
    
    fig.update_yaxes(title_text="Patient Satisfaction", row=1, col=1, range=[0, 105])
    fig.update_yaxes(title_text="Patient Satisfaction", row=1, col=2, range=[0, 105])
    fig.update_yaxes(title_text="Department", row=2, col=1, tickfont=dict(size=12))
    fig.update_yaxes(title_text="Average Satisfaction", row=2, col=2)
    
    fig.update_layout(
        height=800,
        title_text="<b>View 1: Staff Performance Analysis</b><br><sub>TOP panels linked by WEEK | BOTTOM panels linked by SERVICE</sub>",
        template='plotly_white', showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        hovermode='closest'
    )
    
    return fig

def create_view2(df, view_mode, selected_services, selected_events, selected_weeks, hoverData):
    """Temporal staff allocation view"""
    
    if isinstance(selected_services, str):
        selected_services = [selected_services]

    week_start, week_end = selected_weeks
    # Filter columns to only what we need to avoid issues
    df = df.copy()
    
    # Handle column naming differences (loader.py vs independent script)
    # loader uses 'nursing_assistant', script used 'nursing assistant'. 
    # We will standardize on 'nursing_assistant' for computation, but label nicely.
    if 'nursing_assistant' not in df.columns and 'nursing assistant' in df.columns:
        df['nursing_assistant'] = df['nursing assistant']
    elif 'nursing_assistant' in df.columns and 'nursing assistant' not in df.columns:
        # Just to be safe if some logic relies on space
        df['nursing assistant'] = df['nursing_assistant']

    df = df[(df['week'] >= week_start) & (df['week'] <= week_end) & (df['service'].isin(selected_services))]

    if selected_events:
        df = df[df['event'].isin(selected_events)]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=("Staff Allocation", "Patients Admitted")
    )

    if view_mode == 'role':
        # Grouping
        # Note: using 'nursing_assistant' (underscore)
        df_agg = df.groupby('week', as_index=False).agg({
            'doctor': 'sum', 'nurse': 'sum', 'nursing_assistant': 'sum', 'patients_admitted': 'sum'
        }).sort_values('week')

        roles = ['doctor', 'nurse', 'nursing_assistant']
        role_labels = {'doctor': 'Doctor', 'nurse': 'Nurse', 'nursing_assistant': 'Nursing Assistant'}
        colors = ['#636EFA', '#EF553B', '#00CC96']

        for i, role in enumerate(roles):
            fig.add_trace(
                go.Bar(x=df_agg['week'], y=df_agg[role], name=role_labels[role], marker_color=colors[i]),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(x=df_agg['week'], y=df_agg['patients_admitted'],
                      name='Patients Admitted', mode='lines+markers', line=dict(color='black', width=3)),
            row=2, col=1
        )
        title_suffix = " + ".join([s.capitalize() for s in selected_services])

    else:
        colors = px.colors.qualitative.Set2
        for i, srv in enumerate(selected_services):
            df_srv = df[df['service'] == srv].sort_values('week')
            # Sum roles to get total staff
            staff_total = df_srv['doctor'] + df_srv['nurse'] + df_srv['nursing_assistant']

            fig.add_trace(
                go.Bar(x=df_srv['week'], y=staff_total, name=srv.capitalize(), marker_color=colors[i % len(colors)]),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_srv['week'], y=df_srv['patients_admitted'],
                          name=f"{srv.capitalize()} Patients", mode='lines+markers'),
                row=2, col=1
            )
        title_suffix = " vs ".join([s.capitalize() for s in selected_services])

    if hoverData and 'points' in hoverData:
        week = hoverData['points'][0]['x']
        fig.add_vrect(x0=week - 0.5, x1=week + 0.5, fillcolor="rgba(200,200,200,0.3)", line_width=0, layer="below")

    fig.update_layout(
        title=f"<b>View 2: Staff Allocation Timeline</b><br><sub>{title_suffix}</sub>",
        barmode='stack', height=600, legend=dict(orientation="h", y=1.08),
        template='plotly_white'
    )
    fig.update_xaxes(title_text="Week", row=2, col=1)
    fig.update_yaxes(title_text="Staff Count", row=1, col=1)
    fig.update_yaxes(title_text="Patients Admitted", row=2, col=1)

    return fig


# ===========================
# LAYOUT
# ===========================
def create_layout(df):
    
    # Preprocessing or checking if columns like 'doctor', 'nurse', 'nursing_assistant' exist
    # (data loading is handled by loader.py, so df should be ready)
    
    all_services = sorted(df['service'].unique())
    service_options = [{'label': SERVICE_LABELS.get(s, s.capitalize()), 'value': s} for s in all_services]
    event_options = [{'label': e.capitalize(), 'value': e} for e in sorted(df['event'].astype(str).unique())]
    min_week = df['week'].min()
    max_week = df['week'].max()

    return html.Div([
        html.H1("Hospital Staff Analysis", 
                style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif'}),
        
        html.P("Comprehensive analysis of staff performance, satisfaction, and allocation.",
               style={'textAlign': 'center', 'color': '#555'}),

        # ========== View 1 ==========
        html.Div([
            html.H3("Staff Performance & Satisfaction Analysis", 
                    style={'color': '#2c3e50', 'borderBottom': '3px solid #3498db', 'paddingBottom': '10px'}),
            
            dcc.Store(id='dash1-store', data=None),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Week Range:", style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(id='dash1-week-slider', min=min_week, max=max_week,
                                   value=[1, 52], marks={i: str(i) for i in range(min_week, max_week + 1, 10)}, step=1),
                ], style={'width': '45%', 'display': 'inline-block', 'paddingRight': '2%', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label("Services:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='dash1-services', options=service_options, value=all_services, multi=True),
                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Button("Reset", id='dash1-reset', n_clicks=0,
                               style={'cursor':'pointer', 'padding': '5px 15px', 'marginTop': '15px'})
                ], style={'textAlign': 'center'})

            ], style={'width': '95%', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            html.Div(id='dash1-status'),
            dcc.Graph(id='dash1-graph', config={'displayModeBar': True, 'modeBarButtonsToAdd': ['select2d', 'lasso2d']}),
        ], style={'marginBottom': '40px', 'padding': '20px', 'backgroundColor': '#ffffff', 
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

        # ========== View 2 ==========
        html.Div([
            html.H3("Staff Allocation Timeline", 
                    style={'color': '#2c3e50', 'borderBottom': '3px solid #2ecc71', 'paddingBottom': '10px'}),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("View Mode:", style={'fontWeight': 'bold'}),
                    dcc.RadioItems(id='dash2-view-mode',
                                  options=[{'label': ' By Role', 'value': 'role'},
                                          {'label': ' By Service', 'value': 'service'}],
                                  value='role', inline=True, style={'marginTop': '5px'}),
                ], style={'width': '20%', 'display': 'inline-block', 'paddingRight': '2%', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label("Service(s):", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='dash2-services', options=service_options, value=['emergency'], 
                                multi=True, clearable=False),
                ], style={'width': '35%', 'display': 'inline-block', 'paddingRight': '2%', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label("Events:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='dash2-events', options=event_options, value=None, multi=True, placeholder="All events"),
                ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("Week Range:", style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(id='dash2-week-slider', min=min_week, max=max_week,
                                   value=[min_week, max_week],
                                   marks={int(i): str(int(i)) for i in df['week'].unique()}, step=1,
                                   tooltip={"placement": "bottom", "always_visible": False}),
                ], style={'marginTop': '20px'}),

            ], style={'width': '95%', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            dcc.Graph(id='dash2-graph', config={'displayModeBar': True}),
        ], style={'padding': '20px', 'backgroundColor': '#ffffff', 
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    ])


# ===========================
# CALLBACKS
# ===========================
def register_callbacks(app, df):
    
    # We can capture df in the closure for these callbacks
    # Or ensure create_view functions use the df passed here.
    
    # CALLBACKS - VIEW 1
    @app.callback(
        [Output('dash1-graph', 'figure'), Output('dash1-store', 'data')],
        [Input('dash1-week-slider', 'value'), Input('dash1-services', 'value'),
         Input('dash1-graph', 'clickData'), Input('dash1-graph', 'selectedData'), Input('dash1-reset', 'n_clicks')],
        [State('dash1-store', 'data')]
    )
    def update_dash1(weeks, services, click_data, selected_data, reset_clicks, current_state):
        if current_state is None:
            current_state = {'selected_weeks': None, 'selected_services': None}
        
        triggered = ctx.triggered_id
        
        if triggered == 'dash1-reset' and reset_clicks:
            current_state = {'selected_weeks': None, 'selected_services': None}
        elif triggered in ['dash1-week-slider', 'dash1-services']:
            current_state = {'selected_weeks': None, 'selected_services': None}
        elif triggered == 'dash1-graph' and click_data:
            if 'points' in click_data and len(click_data['points']) > 0:
                point = click_data['points'][0]
                if 'customdata' in point and point['customdata']:
                    custom = point['customdata']
                    try:
                        week = int(custom[0])
                        current_state['selected_weeks'] = None if current_state['selected_weeks'] == [week] else [week]
                    except (ValueError, TypeError):
                        service = custom[0]
                        current_state['selected_services'] = None if current_state['selected_services'] == [service] else [service]
        elif triggered == 'dash1-graph' and selected_data:
            if 'points' in selected_data:
                weeks_list = [int(p['customdata'][0]) for p in selected_data['points'] 
                             if 'customdata' in p and p.get('curveNumber', 0) < 16]
                current_state['selected_weeks'] = sorted(list(set(weeks_list))) if weeks_list else None
        
        if not services:
            return go.Figure(), current_state
        
        mask = (df['week'] >= weeks[0]) & (df['week'] <= weeks[1]) & (df['service'].isin(services))
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            return go.Figure(), current_state
        
        fig = create_view1(filtered_df, current_state['selected_weeks'], current_state['selected_services'])
        return fig, current_state

    @app.callback(
        Output('dash1-status', 'children'),
        Input('dash1-store', 'data')
    )
    def update_dash1_status(state):
        if state is None or (not state.get('selected_weeks') and not state.get('selected_services')):
            return html.Div("Showing all data", style={'padding': '8px', 'backgroundColor': '#ecf0f1', 
                                                        'borderRadius': '5px', 'marginBottom': '10px', 'fontSize': '14px'})
        
        parts = []
        if state.get('selected_weeks'):
            weeks = state['selected_weeks']
            week_text = f"Week {weeks[0]}" if len(weeks) == 1 else f"{len(weeks)} weeks"
            parts.append(html.Span([html.B("Weeks: "), week_text]))
        if state.get('selected_services'):
            services = [SERVICE_LABELS.get(s, s) for s in state['selected_services']]
            parts.append(html.Span([" | " if parts else "", html.B("Services: "), ', '.join(services)]))
        
        return html.Div(parts, style={'padding': '10px', 'backgroundColor': '#fff3cd', 'borderRadius': '5px', 
                                      'marginBottom': '10px', 'border': '1px solid #ffc107'})

    # CALLBACKS - VIEW 2
    @app.callback(
        Output('dash2-graph', 'figure'),
        [Input('dash2-view-mode', 'value'), Input('dash2-services', 'value'), Input('dash2-events', 'value'),
         Input('dash2-week-slider', 'value'), Input('dash2-graph', 'hoverData')]
    )
    def update_dash2(view_mode, services, events, weeks, hover_data):
        return create_view2(df, view_mode, services, events, weeks, hover_data)
