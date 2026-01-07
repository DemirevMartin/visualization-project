import dash 
from dash import html, dcc, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats

# Load data
patients_df = pd.read_csv("../data/patients.csv")
services_df = pd.read_csv("../data/services_weekly.csv")
staff_df = pd.read_csv("../data/staff.csv")
schedule_df = pd.read_csv("../data/staff_schedule.csv")

# Prepare patient data
patients_df['arrival_date'] = pd.to_datetime(patients_df['arrival_date'])
patients_df['departure_date'] = pd.to_datetime(patients_df['departure_date'])
patients_df['length_of_stay'] = (patients_df['departure_date'] - patients_df['arrival_date']).dt.days
patients_df['week'] = patients_df['arrival_date'].dt.isocalendar().week

# Staff metrics
staff_metrics = schedule_df.groupby(['week', 'service']).agg({
    'present': ['sum', 'count', 'mean']
}).reset_index()
staff_metrics.columns = ['week', 'service', 'staff_present', 'total_staff', 'presence_rate']

# Role presence
staff_schedule_present_df = schedule_df[schedule_df['present'] == 1]
df_agg_counts = (
    staff_schedule_present_df
    .groupby(['week', 'service', 'role'])['staff_id']
    .nunique()
    .reset_index(name='count')
)
role_pivot = df_agg_counts.pivot_table(
    index=['week', 'service'],
    columns='role',
    values='count',
    fill_value=0
).reset_index()

# Patient outcomes
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

# Merged data
df = services_df.merge(staff_metrics, on=['week', 'service'], how='left'
                 ).merge(role_pivot, on=['week', 'service'], how='left'
                 ).merge(patient_outcomes, on=['week', 'service'], how='left')

df['patients_per_staff'] = df['patients_admitted'] / df['staff_present'].replace(0, np.nan)

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

ROLE_COLORS = ['#636EFA', '#EF553B', '#00CC96']

def create_integrated_figure(df, selected_weeks=None, selected_service='emergency', view_mode='role'):
    """
    Creates a 3-panel dashboard with Jittered scatter plots and 2-way highlighting.
    """
    # 1. DATA CLEANING & JITTERING
    plot_df = df[(df['staff_present'] > 0) & (df['patient_satisfaction'] > 0)].copy()

    np.random.seed(42) 
    plot_df['sat_jitter'] = plot_df['patient_satisfaction'] + np.random.uniform(-0.8, 0.8, len(plot_df))
    plot_df['pres_jitter'] = (plot_df['presence_rate'] * 100) + np.random.uniform(-0.5, 0.5, len(plot_df))
    plot_df['work_jitter'] = plot_df['patients_per_staff'] + np.random.uniform(-0.1, 0.1, len(plot_df))

    # 2. SUBPLOT SETUP
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.45, 0.55],
        column_widths=[0.5, 0.5],
        specs=[
            [{"type": "xy"}, {"type": "xy"}], 
            [{"secondary_y": True, "colspan": 2}, None]
        ],
        subplot_titles=[
            'Staff Presence vs Patient Satisfaction',
            'Workload Pressure vs Patient Satisfaction',
            f'Staff Allocation & Patient Load - {SERVICE_LABELS.get(selected_service, selected_service)}'
        ],
        vertical_spacing=0.15, horizontal_spacing=0.10
    )
    
    # 3. TOP SCATTER PLOTS (Dynamic Focus & Highlighting)
    all_srv = list(SERVICE_COLORS.keys())
    # Ensure selected service is drawn last so it is on top
    draw_order = [s for s in all_srv if s != selected_service] + [selected_service]

    for service in draw_order:
        color = SERVICE_COLORS[service]
        s_data = plot_df[plot_df['service'] == service]
        if s_data.empty: continue
            
        is_sel_srv = (service == selected_service)
        base_opacity = 0.85 if is_sel_srv else 0.15
        base_size = 12 if is_sel_srv else 7
        
        for col, x_col in [(1, 'pres_jitter'), (2, 'work_jitter')]:
            # --- TRACE 1: THE REGULAR DATA POINTS ---
            fig.add_trace(go.Scatter(
                x=s_data[x_col],
                y=s_data['sat_jitter'],
                mode='markers',
                name=SERVICE_LABELS.get(service, service),
                marker=dict(
                    color=color, size=base_size, opacity=base_opacity,
                    line=dict(width=0.5, color='white')
                ),
                # We pack the week into customdata[0] for the callback to read
                customdata=np.column_stack((s_data['week'], s_data['service'], s_data['patient_satisfaction'])),
                hovertemplate='<b>%{customdata[1]}</b><br>Week: %{customdata[0]}<br>Sat: %{customdata[2]:.1f}<extra></extra>',
                legendgroup=service,
                showlegend=(col == 1)
            ), row=1, col=col)
            
            # --- TRACE 2: THE 2-WAY HIGHLIGHTER (The missing link) ---
            # If a week is selected (from anywhere), we draw a highlight ring on the scatter dot
            if selected_weeks:
                sel_dots = s_data[s_data['week'].isin(selected_weeks)]
                if not sel_dots.empty:
                    fig.add_trace(go.Scatter(
                        x=sel_dots[x_col],
                        y=sel_dots['sat_jitter'],
                        mode='markers',
                        marker=dict(
                            color=color, size=base_size + 8, 
                            line=dict(width=3, color='yellow'),
                            opacity=1.0
                        ),
                        # Match the customdata so clicking the highlight also works
                        customdata=np.column_stack((sel_dots['week'], sel_dots['service'], sel_dots['patient_satisfaction'])),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=1, col=col)

    # 4. TREND LINE
    valid_corr = plot_df[['presence_rate', 'patient_satisfaction']].dropna()
    if len(valid_corr) > 3:
        x_vals = valid_corr['presence_rate'] * 100
        y_vals = valid_corr['patient_satisfaction']
        slope, intercept, _, _, _ = stats.linregress(x_vals, y_vals)
        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig.add_trace(go.Scatter(x=x_range, y=slope*x_range + intercept, mode='lines',
                                line=dict(color='black', width=1.5, dash='dot'), 
                                name='Overall Trend', showlegend=False), row=1, col=1)

    # 5. BOTTOM TIMELINE: Staff Allocation
    df_srv = plot_df[plot_df['service'] == selected_service].sort_values('week')
    
    if view_mode == 'role':
        roles = ['doctor', 'nurse', 'nursing_assistant']
        for role, r_color in zip(roles, ROLE_COLORS):
            if role in df_srv.columns:
                fig.add_trace(go.Bar(
                    x=df_srv['week'], y=df_srv[role], 
                    name=role.replace('_', ' ').capitalize(), 
                    marker_color=r_color,
                    customdata=df_srv['week'] # Callback reads this when bar is clicked
                ), row=2, col=1)
    else:
        fig.add_trace(go.Bar(
            x=df_srv['week'], y=df_srv['patients_per_staff'] * df_srv['staff_present'], 
            name='Workload', marker_color='#95a5a6',
            customdata=df_srv['week']
        ), row=2, col=1)

    # Patient Load Line (Secondary Axis)
    fig.add_trace(go.Scatter(
        x=df_srv['week'], y=df_srv['patients_admitted'], 
        name='Patient Load', mode='lines+markers', 
        line=dict(color='black', width=3),
        marker=dict(size=8, symbol='diamond'),
        customdata=df_srv['week'] # Allows clicking the line to highlight too
    ), row=2, col=1, secondary_y=True)

    # 6. LAYOUT & AXIS STYLING
    fig.update_layout(
        height=850, 
        template='plotly_white', 
        barmode='stack', 
        margin=dict(r=150, t=100),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )

    # FIX FOR "PACKED" RIGHT PANEL:
    # Set the range explicitly to [0, 10] to zoom in on the majority of data
    fig.update_xaxes(title_text="Workload (Patients/Staff)", range=[0, 10], row=1, col=2)
    fig.update_yaxes(title_text="Satisfaction Score", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Satisfaction Score", range=[0, 105], row=1, col=2)
    
    # 2-way Selection highlight in timeline
    if selected_weeks:
        for week in selected_weeks:
            fig.add_vrect(
                x0=week-0.4, x1=week+0.4, 
                fillcolor="yellow", opacity=0.3, 
                layer="below", line_width=0, row=2, col=1
            )

    return fig

# Initialize app
app = dash.Dash(__name__)
app.title = "Hospital Staff Dashboard"
all_services = sorted(df['service'].unique())

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Hospital Staff Performance & Allocation Analysis",
                style={'color': '#2c3e50', 'marginBottom': '5px'})
    ], style={'textAlign': 'center', 'padding':'20px', 'backgroundColor':'#ecf0f1'}),

    # Store for interaction state
    dcc.Store(id='state-store', data={'selected_weeks': None, 'selected_service': 'emergency'}),
    
    # Filters - Compact Row
    html.Div([
        html.Div([
            html.Label("📅 Week Range:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RangeSlider(
                id='week-slider',
                min=int(df['week'].min()),
                max=int(df['week'].max()),
                value=[1, 52],
                marks={i: str(i) for i in range(0, 53, 10)},
                step=1,
                tooltip={"placement": "bottom", "always_visible": False}
            ),
        ], style={'width': '35%', 'display': 'inline-block', 'paddingRight': '2%', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("🏥 Focus Service:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='service-selector',
                options=[{'label': SERVICE_LABELS.get(s, s), 'value': s} for s in all_services],
                value='emergency',
                clearable=False,
                style={'width': '200px'}
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'paddingRight': '2%', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("📊 Timeline View:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='view-mode',
                options=[
                    {'label': ' By Role', 'value': 'role'},
                    {'label': ' Total Staff', 'value': 'total'}
                ],
                value='role',
                inline=True,
                style={'marginTop': '5px'}
            )
        ], style={'width': '20%', 'display': 'inline-block', 'paddingRight': '2%', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Button("🔄 Reset", id='reset-btn', n_clicks=0,
                       style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                              'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer',
                              'fontWeight': 'bold', 'marginTop': '20px'})
        ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'right'})
    ], style={'padding': '15px 20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '10px'}),
    
    # Status display
    html.Div(id='status-display'),
    
    # Main visualization
    dcc.Graph(id='main-graph', config={'displayModeBar': True}),
    
    # Insights box
    html.Div([
        html.H4("💡 Key Insights:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
        html.Div(id='insights-text', style={'fontSize': '14px', 'lineHeight': '1.8'})
    ], style={'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '8px',
              'border': '1px solid #3498db', 'margin': '20px 0'})
])


@app.callback(
    [Output('main-graph', 'figure'),
     Output('state-store', 'data'),
     Output('insights-text', 'children')],
    [Input('week-slider', 'value'),
     Input('service-selector', 'value'),
     Input('view-mode', 'value'),
     Input('main-graph', 'clickData'),
     Input('reset-btn', 'n_clicks')],
    [State('state-store', 'data')]
)
def update_dashboard(week_range, service_selector, view_mode, click_data, reset_clicks, current_state):
    """Main callback handling all interactions"""
    
    if current_state is None:
        current_state = {'selected_weeks': None, 'selected_service': 'emergency'}
    
    triggered = ctx.triggered_id
    
    # Reset
    if triggered == 'reset-btn':
        current_state['selected_weeks'] = None
        # We keep the service selector's current value
        current_state['selected_service'] = service_selector
    
    # Filter changes
    elif triggered in ['week-slider', 'service-selector', 'view-mode']:
        current_state['selected_service'] = service_selector
        if triggered == 'week-slider':
            current_state['selected_weeks'] = None
    
    # Click interactions
    elif triggered == 'main-graph' and click_data:
        point = click_data['points'][0]
        # We extract the week number from customdata.
        # Note: In our figure function, we must ensure BOTH scatter and bars 
        # have the week number as the first element in customdata.
        if 'customdata' in point:
            cd = point['customdata']
            try:
                if isinstance(cd, (list, np.ndarray)):
                    clicked_week = int(cd[0])
                else:
                    clicked_week = int(cd)
                    
                # Toggle logic: Clicking the same week twice deselects it
                if current_state.get('selected_weeks') == [clicked_week]:
                    current_state['selected_weeks'] = None
                else:
                    current_state['selected_weeks'] = [clicked_week]
            except (IndexError, TypeError, ValueError):
                pass
    
    # Filter data
    mask = (df['week'] >= week_range[0]) & (df['week'] <= week_range[1])
    filtered_df = df[mask].copy()
    
    # Generate figure
    fig = create_integrated_figure(
        filtered_df,
        selected_weeks=current_state['selected_weeks'],
        selected_service=current_state['selected_service'],
        view_mode=view_mode
    )
    
    # Generate insights
    insights = generate_insights(filtered_df, current_state['selected_service'], current_state['selected_weeks'])
    
    return fig, current_state, insights


def generate_insights(df, selected_service, selected_weeks):
    """Generate dynamic insights based on current selection"""
    
    insights = []
    
    # Overall correlation insight
    valid = df[['presence_rate', 'patient_satisfaction']].dropna()
    if len(valid) > 3:
        corr = valid['presence_rate'].corr(valid['patient_satisfaction'])
        insights.append(
            html.Div([
                html.Span("📈 ", style={'fontSize': '18px'}),
                html.Span(f"Staff presence shows a "),
                html.B(f"{'strong' if abs(corr) > 0.6 else 'moderate' if abs(corr) > 0.3 else 'weak'} positive correlation", 
                       style={'color': '#27ae60' if corr > 0.6 else '#f39c12'}),
                html.Span(f" (r={corr:.3f}) with patient satisfaction across all services.")
            ])
        )
    
    # Service-specific insight
    srv_data = df[df['service'] == selected_service]
    if len(srv_data) > 0:
        avg_presence = srv_data['presence_rate'].mean() * 100
        avg_satisfaction = srv_data['patient_satisfaction'].mean()
        insights.append(
            html.Div([
                html.Span("🏥 ", style={'fontSize': '18px'}),
                html.B(SERVICE_LABELS.get(selected_service, selected_service), style={'color': SERVICE_COLORS.get(selected_service)}),
                html.Span(f" averages {avg_presence:.1f}% staff presence with {avg_satisfaction:.1f} satisfaction score.")
            ], style={'marginTop': '10px'})
        )
    
    # Week-specific insight
    if selected_weeks and len(selected_weeks) > 0:
        week = selected_weeks[0]
        week_data = srv_data[srv_data['week'] == week]
        if len(week_data) > 0:
            row = week_data.iloc[0]
            insights.append(
                html.Div([
                    html.Span("📍 ", style={'fontSize': '18px'}),
                    html.Span(f"Week {week}: "),
                    html.Span(f"{int(row.get('doctor', 0))} doctors, {int(row.get('nurse', 0))} nurses, "),
                    html.Span(f"{int(row.get('nursing_assistant', 0))} assistants served {int(row['patients_admitted'])} patients "),
                    html.Span(f"(satisfaction: {row['patient_satisfaction']:.1f}).")
                ], style={'marginTop': '10px', 'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '5px'})
            )
    
    return insights if insights else html.Div("Select weeks or services to see specific insights.", 
                                              style={'color': '#7f8c8d', 'fontStyle': 'italic'})


@app.callback(
    Output('status-display', 'children'),
    Input('state-store', 'data')
)
def update_status(state):
    """Display current selection status"""
    
    if not state:
        return html.Div()
    
    parts = []
    
    if state.get('selected_service'):
        service_name = SERVICE_LABELS.get(state['selected_service'], state['selected_service'])
        parts.append(
            html.Span([
                html.B("Viewing: ", style={'color': '#2ecc71'}),
                html.Span(service_name, style={'fontSize': '15px'})
            ])
        )
    
    if state.get('selected_weeks'):
        weeks = state['selected_weeks']
        week_text = f"Week {weeks[0]}" if len(weeks) == 1 else f"Weeks {min(weeks)}-{max(weeks)}"
        parts.append(
            html.Span([
                html.Span(" • ", style={'margin': '0 8px', 'color': '#bdc3c7'}),
                html.B("Selected: ", style={'color': '#3498db'}),
                html.Span(week_text, style={'fontSize': '15px'})
            ])
        )
    
    if not parts:
        return html.Div([
            html.Span("ℹ️ ", style={'fontSize': '16px'}),
            html.Span("Click any point or bar to investigate specific weeks",
                     style={'color': '#7f8c8d', 'fontSize': '14px'})
        ], style={'padding': '8px 15px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px', 'marginBottom': '10px'})
    
    return html.Div(
        parts,
        style={
            'padding': '10px 15px',
            'backgroundColor': '#fff3cd',
            'borderRadius': '6px',
            'border': '1px solid #ffc107',
            'marginBottom': '10px',
            'fontSize': '14px'
        }
    )


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8055)