import dash 
from dash import html, dcc, Input, Output, State, ctx, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats

#load data
patients_df = pd.read_csv("../data/patients.csv")
services_df = pd.read_csv("../data/services_weekly.csv")
staff_df = pd.read_csv("../data/staff.csv")
schedule_df = pd.read_csv("../data/staff_schedule.csv")

patients_df['arrival_date'] = pd.to_datetime(patients_df['arrival_date'])
patients_df['departure_date'] = pd.to_datetime(patients_df['departure_date'])
patients_df['length_of_stay'] = (patients_df['departure_date'] - patients_df['arrival_date']).dt.days
patients_df['week'] = patients_df['arrival_date'].dt.isocalendar().week

#staff metrics
staff_metrics = schedule_df.groupby(['week', 'service']).agg({
    'present': ['sum', 'count', 'mean']
}).reset_index()
staff_metrics.columns = ['week', 'service', 'staff_present', 'total_staff', 'presence_rate']

#role presence
role_presence = schedule_df.groupby(['week', 'service', 'role']).agg({
    'present': 'sum'
}).reset_index()
role_pivot = role_presence.pivot_table(index=['week', 'service'], columns='role', values='present', fill_value=0).reset_index()
role_pivot.columns = ['week', 'service', 'doctor', 'nurse', 'nursing_assistant']

#patient outcomes
patient_outcomes = patients_df.groupby(['week', 'service']).agg({
    'satisfaction': ['mean', 'std', 'min', 'max'],
    'length_of_stay': ['mean', 'median', 'std', 'max'],
    'patient_id': 'count'
}).reset_index()
patient_outcomes.columns = ['week', 'service', 
                            'avg_satisfaction', 'std_satisfaction',
                            'min_satisfaction', 'max_satisfaction',
                            'avg_los', 'median_los', 'std_los', 'max_los',
                            'patient_count']

#merged data
df = services_df.merge(staff_metrics, on=['week', 'service'], how='left'
                 ).merge(role_pivot, on=['week', 'service'], how='left'
                 ).merge(patient_outcomes, on=['week', 'service'], how='left')

df['patients_per_staff'] = df['patients_admitted'] / df['staff_present'].replace(0, np.nan)
df['workload'] = df['patients_request'] / df['staff_present'].replace(0, np.nan)

SERVICE_COLORS = {
    'emergency': '#d62728',
    'ICU': '#ff7f0e',
    'surgery': '#2ca02c',
    'general_medicine': '#1f77b4'}

SERVICE_LABELS = {
    'emergency': 'Emergency',
    'ICU': 'ICU',
    'surgery': 'Surgery',
    'general_medicine': 'General Medicine'}

def create_dashboard(df, selected_weeks=None, selected_services=None):
    """
    Creates the content for Task 3: Staff vs Outcomes.
    Top two panels are linked by week selection.
    Bottom two panels are linked by service selection.
    """
    if selected_weeks is not None and len(selected_weeks) > 0:
        df['week_selected'] = df['week'].isin(selected_weeks)
    else:
        df['week_selected'] = True

    if selected_services is not None and len(selected_services) > 0:
        df['service_selected'] = df['service'].isin(selected_services)
    else:
        df['service_selected'] = True

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Staff Presence vs Satisfaction',
        'Workload Pressure vs Satisfaction',
        'Department Performance',
        'Service Overview'
    ], 
    specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "scatter"}]],
    vertical_spacing=0.18, horizontal_spacing=0.15)

    # TOP-LEFT: Presence vs Satisfaction (linked with top-right by WEEK)
    for service, color in SERVICE_COLORS.items():
        service_data = df[df['service'] == service].copy()
        if len(service_data) == 0:
            continue

        selected = service_data[service_data['week_selected']]
        not_selected = service_data[~service_data['week_selected']]

        if len(not_selected) > 0:
            fig.add_trace(
                go.Scatter(
                    x=not_selected['presence_rate'] * 100,
                    y=not_selected['patient_satisfaction'],
                    mode='markers',
                    name=SERVICE_LABELS[service],
                    marker=dict(color=color, size=6, opacity=0.2),
                    customdata=np.column_stack((not_selected['week'], not_selected['service'])),
                    hovertemplate='<b>%{customdata[1]}</b><br>Week: %{customdata[0]}<br>Presence: %{x:.1f}%<br>Satisfaction: %{y:.1f}<extra></extra>',
                    legendgroup=service,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        if len(selected) > 0:
            fig.add_trace(
                go.Scatter(
                    x=selected['presence_rate'] * 100,
                    y=selected['patient_satisfaction'],
                    mode='markers',
                    name=SERVICE_LABELS[service],
                    marker=dict(color=color, size=14, opacity=1.0, line=dict(width=2, color='white')),
                    customdata=np.column_stack((selected['week'], selected['service'])),
                    hovertemplate='<b>%{customdata[1]}</b><br>Week: %{customdata[0]}<br>Presence: %{x:.1f}%<br>Satisfaction: %{y:.1f}<extra></extra>',
                    legendgroup=service,
                    showlegend=True
                ),
                row=1, col=1
            )
    
    valid = df[['presence_rate', 'patient_satisfaction']].dropna()
    if len(valid) > 3:
        x_vals = valid['presence_rate'] * 100
        y_vals = valid['patient_satisfaction']
        slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_pred = slope * x_range + intercept

        fig.add_trace(
            go.Scatter(
                x=x_range, y=y_pred, mode='lines',
                line=dict(color='black', width=3, dash='dash'),
                showlegend=False,
                hovertemplate=f'r={r_value:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_vline(x=75, line_dash="dot", line_color="red", line_width=2, row=1, col=1)
        fig.add_vrect(x0=0, x1=75, fillcolor="red", opacity=0.05, layer="below", line_width=0, row=1, col=1)

    # TOP-RIGHT: Workload vs Satisfaction (linked with top-left by WEEK)
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
                        x=not_selected['patients_per_staff'],
                        y=not_selected['patient_satisfaction'],
                        mode='markers',
                        marker=dict(color=color, size=6, opacity=0.2),
                        customdata=np.column_stack((not_selected['week'], not_selected['service'])),
                        hovertemplate='<b>%{customdata[1]}</b><br>Week: %{customdata[0]}<br>Workload: %{x:.1f}<br>Sat: %{y:.1f}<extra></extra>',
                        legendgroup=service,
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            if len(selected) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=selected['patients_per_staff'],
                        y=selected['patient_satisfaction'],
                        mode='markers',
                        marker=dict(color=color, size=14, opacity=1.0, line=dict(width=2, color='white')),
                        customdata=np.column_stack((selected['week'], selected['service'])),
                        hovertemplate='<b>%{customdata[1]}</b><br>Week: %{customdata[0]}<br>Workload: %{x:.1f}<br>Sat: %{y:.1f}<extra></extra>',
                        legendgroup=service,
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        fig.add_vrect(x0=8, x1=15, fillcolor="red", opacity=0.15, layer="below", line_width=0, row=1, col=2)
        fig.add_vline(x=8, line_dash="dash", line_color="orange", line_width=2, row=1, col=2)

    # BOTTOM-LEFT: Department Performance (linked with bottom-right by SERVICE)
    dept_avg = df.groupby('service').agg({
        'patient_satisfaction': 'mean',
        'avg_satisfaction': 'mean'
    }).reset_index()
    dept_avg.columns = ['service', 'service_satisfaction', 'patient_satisfaction']
    dept_avg = dept_avg.sort_values('service_satisfaction', ascending=True)
    
    for _, row in dept_avg.iterrows():
        service = row['service']
        color = SERVICE_COLORS.get(service, '#7f8c8d')
        is_selected = (selected_services is None or len(selected_services) == 0 or service in selected_services)
        
        # Use full service label with more spacing
        service_label = SERVICE_LABELS.get(service, service.replace('_', ' ').title())
        
        fig.add_trace(
            go.Bar(
                y=[service_label],
                x=[row['service_satisfaction']],
                orientation='h',
                marker=dict(
                    color=color, 
                    opacity=1.0 if is_selected else 0.3, 
                    line=dict(width=2 if is_selected else 1, color='white')
                ),
                text=[f"{row['service_satisfaction']:.1f}"],
                textposition='inside',
                textfont=dict(size=12, color='white'),
                legendgroup=service,
                showlegend=False,
                hovertemplate=f'<b>{service_label}</b><br>Service Avg: %{{x:.1f}}<extra></extra>',
                customdata=[[service]]
            ),
            row=2, col=1
        )
        
        # Only add diamond marker if patient_satisfaction differs significantly from service_satisfaction
        if abs(row['patient_satisfaction'] - row['service_satisfaction']) > 0.01:
            fig.add_trace(
                go.Scatter(
                    x=[row['patient_satisfaction'] + 3.5],
                    y=[service_label],
                    mode='markers',
                    marker=dict(
                        symbol='diamond', 
                        size=10 if is_selected else 7, 
                        color='white', 
                        line=dict(width=2, color=color)
                    ),
                    showlegend=False,
                    hovertemplate=f'<b>{service_label}</b><br>Patient Avg: %{{x:.1f}}<extra></extra>',
                    customdata=[[service]]
                ),
                row=2, col=1
            )
    
    # BOTTOM-RIGHT: Service Overview (linked with bottom-left by SERVICE)
    service_summary = df.groupby('service').agg({
        'presence_rate': 'mean',
        'patient_satisfaction': 'mean',
        'patients_per_staff': 'mean'
    }).reset_index()
    
    # Calculate better ranges for visibility with wider spread
    presence_values = service_summary['presence_rate'] * 100
    satisfaction_values = service_summary['patient_satisfaction']
    
    presence_min = presence_values.min()
    presence_max = presence_values.max()
    presence_range = presence_max - presence_min
    
    satisfaction_min = satisfaction_values.min()
    satisfaction_max = satisfaction_values.max()
    satisfaction_range = satisfaction_max - satisfaction_min
    
    # Add significant padding (30%) to create visual separation
    x_min_panel4 = max(0, presence_min - (presence_range * 0.4))
    x_max_panel4 = min(100, presence_max + (presence_range * 0.4))
    y_min_panel4 = max(0, satisfaction_min - (satisfaction_range * 0.4))
    y_max_panel4 = min(100, satisfaction_max + (satisfaction_range * 0.4))

    for _, row in service_summary.iterrows():
        service = row['service']
        color = SERVICE_COLORS.get(service, '#7f8c8d')
        is_selected = (selected_services is None or len(selected_services) == 0 or service in selected_services)
        
        # Size is based on patients_per_staff - normalize for better visibility
        base_size = 20  # Base size for all bubbles
        size_multiplier = 3  # Reduced multiplier for less dramatic size differences
        bubble_size = base_size + (row['patients_per_staff'] * size_multiplier)
        
        fig.add_trace(
            go.Scatter(
                x=[row['presence_rate'] * 100],
                y=[row['patient_satisfaction']],
                mode='markers+text',
                marker=dict(
                    color=color, 
                    size=bubble_size if is_selected else bubble_size * 0.5, 
                    opacity=1.0 if is_selected else 0.3
                ),
                text=[SERVICE_LABELS.get(service, service)] if is_selected else [''],
                textposition='top center',
                textfont=dict(size=13 if is_selected else 10, color=color),
                legendgroup=service,
                showlegend=False,
                hovertemplate=f'<b>{SERVICE_LABELS.get(service, service)}</b><br>Presence: %{{x:.1f}}%<br>Satisfaction: %{{y:.1f}}<br>Patients/Staff: {row["patients_per_staff"]:.1f}<extra></extra>',
                customdata=[[service]]
            ),
            row=2, col=2
        )
    
    median_presence = service_summary['presence_rate'].median() * 100
    median_satisfaction = service_summary['patient_satisfaction'].median()
    fig.add_hline(y=median_satisfaction, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=2)
    fig.add_vline(x=median_presence, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=2)

    # Layout with improved bottom-right zoom
    fig.update_xaxes(title_text="Staff Presence Rate (%)", row=1, col=1, range=[0, 105])
    fig.update_xaxes(title_text="Patients per Staff", row=1, col=2, range=[0, 15])
    fig.update_xaxes(title_text="Average Satisfaction", row=2, col=1, range=[0, 105])
    fig.update_xaxes(title_text="Average Presence Rate (%)", row=2, col=2, range=[x_min_panel4, x_max_panel4])
    
    fig.update_yaxes(title_text="Patient Satisfaction", row=1, col=1, range=[0, 105])
    fig.update_yaxes(title_text="Patient Satisfaction", row=1, col=2, range=[0, 105])
    fig.update_yaxes(title_text="Department", row=2, col=1, tickfont=dict(size=12))
    fig.update_yaxes(title_text="Average Satisfaction", row=2, col=2, range=[y_min_panel4, y_max_panel4])
    
    fig.update_layout(
        height=1000,
        title_text="<b>Two-Way Linked Staff Analysis Dashboard</b><br>" +
                  "<sub>TOP panels linked by WEEK | BOTTOM panels linked by SERVICE</sub>",
        template='plotly_white',
        showlegend=True,
        legend=dict(title="<b>Services</b>", orientation="v", yanchor="top", y=0.98, xanchor="right", x=1.15),
        hovermode='closest'
    )
    
    return fig

app = dash.Dash(__name__)
app.title = "Dashboard Demo"
all_services = sorted(df['service'].unique())

app.layout = html.Div([
    html.Div([
        html.H1("Hospital Staff Analysis Dashboard",
                style={'color': '#2c3e50', 'marginBottom': '10px'})  
    ], style={'textAlign': 'center', 'padding':'20px', 'backgroundColor':'#ecf0f1'}),

    dcc.Store(id='demo-interaction-store', data=None),
    html.Div([
        html.Div([
            html.Label("Week Range:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='demo-slider',
                min=int(df['week'].min()),
                max=int(df['week'].max()),
                value=[1, 52],
                marks={i: str(i) for i in range(0, 53, 10)},
                step=1
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'paddingRight': '2%'}),
        
        html.Div([
            html.Label("Services:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='demo-dropdown',
                options=[{'label': s.replace('_',' ').title(), 'value': s} for s in all_services],
                value=all_services,
                multi=True
            ),
        ], style={'width': '45%', 'display': 'inline-block'}),
        
        html.Button(
            "Reset Selections",
            id='demo-reset-btn',
            n_clicks=0,
            style={
                'backgroundColor': '#e74c3c',
                'color': 'white',
                'border': 'none',
                'padding': '12px 24px',
                'borderRadius': '6px',
                'cursor': 'pointer',
                'marginLeft': '20px',
                'fontWeight': 'bold'
            }
        )
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '15px'}),
    
    # Status display
    html.Div(id='demo-status'),
    
    # Main content
    dcc.Graph(
        id='demo-graph',
        config={'displayModeBar': True, 'modeBarButtonsToAdd': ['select2d', 'lasso2d']}
    ),
    
    # Instructions
    html.Div([
        html.H4("How to Use:", style={'color': '#2c3e50'}),
        html.Ul([
            html.Li("TOP PANELS (linked by WEEK): Click any point or brush select → highlights same weeks in BOTH top panels"),
            html.Li("BOTTOM PANELS (linked by SERVICE): Click bars or bubbles → emphasizes same service in BOTH bottom panels"),
            html.Li("Top and bottom selections are independent - you can have both active simultaneously"),
            html.Li("Use filters to narrow data range, then interact with visualizations"),
            html.Li("Click Reset to clear all interactive selections")
        ]),
    ], style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '8px', 'border': '1px solid #3498db', 'margin': '20px'})
])

# Callbacks
@app.callback(
    [Output('demo-graph', 'figure'),
     Output('demo-interaction-store', 'data')],
    [Input('demo-slider', 'value'),
     Input('demo-dropdown', 'value'),
     Input('demo-graph', 'clickData'),
     Input('demo-graph', 'selectedData'),
     Input('demo-reset-btn', 'n_clicks')],
    [State('demo-interaction-store', 'data')]
)
def update_demo(weeks, services, click_data, selected_data, reset_clicks, current_state):
    """Main callback - handles all interactions"""
    
    if current_state is None:
        current_state = {'selected_weeks': None, 'selected_services': None}
    
    triggered = ctx.triggered_id
    
    # Reset
    if triggered == 'demo-reset-btn' and reset_clicks:
        current_state = {'selected_weeks': None, 'selected_services': None}
    
    # Filter changes clear selections
    elif triggered in ['demo-slider', 'demo-dropdown']:
        current_state = {'selected_weeks': None, 'selected_services': None}
    
    # Click events
    elif triggered == 'demo-graph' and click_data:
        if 'points' in click_data and len(click_data['points']) > 0:
            point = click_data['points'][0]
            curve_num = point.get('curveNumber', 0)
            
            if 'customdata' in point and point['customdata']:
                custom = point['customdata']
                
                # Check if first element is a week (integer) or service (string)
                try:
                    # Try to parse as week - if successful, it's a TOP panel click
                    week = int(custom[0])
                    if current_state['selected_weeks'] == [week]:
                        current_state['selected_weeks'] = None
                    else:
                        current_state['selected_weeks'] = [week]
                except (ValueError, TypeError):
                    # If parsing fails, it's a service name - BOTTOM panel click
                    service = custom[0]
                    if current_state['selected_services'] == [service]:
                        current_state['selected_services'] = None
                    else:
                        current_state['selected_services'] = [service]
    
    # Brush selection (only for top panels - week selection)
    elif triggered == 'demo-graph' and selected_data:
        if 'points' in selected_data:
            weeks_list = []
            for point in selected_data['points']:
                if point.get('curveNumber', 0) < 16 and 'customdata' in point:
                    week = int(point['customdata'][0])
                    if week not in weeks_list:
                        weeks_list.append(week)
            
            if weeks_list:
                current_state['selected_weeks'] = sorted(weeks_list)
    
    # Filter data
    if not services:
        return go.Figure(), current_state
    
    mask = (df['week'] >= weeks[0]) & (df['week'] <= weeks[1]) & (df['service'].isin(services))
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        return go.Figure(), current_state
    
    # Generate visualization
    fig = create_dashboard(
        filtered_df,
        selected_weeks=current_state['selected_weeks'],
        selected_services=current_state['selected_services']
    )
    
    return fig, current_state


@app.callback(
    Output('demo-status', 'children'),
    Input('demo-interaction-store', 'data')
)
def update_status(state):
    """Show current selection status"""
    
    if state is None or (not state.get('selected_weeks') and not state.get('selected_services')):
        return html.Div([
            html.Span(" ", style={'fontSize': '20px'}),
            html.Span("Showing all data. Click on visualizations to highlight specific weeks or services.",
                     style={'color': '#3498db', 'fontWeight': 'bold'})
        ], style={'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px', 'marginBottom': '15px'})
    
    parts = []
    
    if state.get('selected_weeks'):
        weeks = state['selected_weeks']
        week_text = f"Week {weeks[0]}" if len(weeks) == 1 else f"{len(weeks)} weeks ({min(weeks)}-{max(weeks)})"
        parts.append(
            html.Div([
                html.Span(" ", style={'fontSize': '18px'}),
                html.B("TOP panels (Week filter): ", style={'color': '#3498db'}),
                html.Span(week_text, style={'fontSize': '16px'})
            ])
        )
    
    if state.get('selected_services'):
        services = [SERVICE_LABELS.get(s, s.replace('_', ' ').title()) for s in state['selected_services']]
        parts.append(
            html.Div([
                html.Span(" ", style={'fontSize': '18px'}),
                html.B("BOTTOM panels (Service filter): ", style={'color': '#2ecc71'}),
                html.Span(', '.join(services), style={'fontSize': '16px'})
            ])
        )
    
    return html.Div(
        parts,
        style={
            'padding': '15px',
            'backgroundColor': '#fff3cd',
            'borderRadius': '8px',
            'border': '2px solid #ffc107',
            'marginBottom': '15px'
        }
    )

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8054)