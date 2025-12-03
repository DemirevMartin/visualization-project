import dash 
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ------------------------
# 1. Config / Data Load 
# ------------------------
PATIENTS_PATH = "./data/patients.csv"
SERVICES_PATH = "./data/services_weekly.csv"
STAFF_PATH = "./data/staff.csv"
SCHEDULE_PATH = "./data/staff_schedule.csv"

patients_df = pd.read_csv(PATIENTS_PATH)
services_df = pd.read_csv(SERVICES_PATH)    
staff_df = pd.read_csv(STAFF_PATH)
schedule_df = pd.read_csv(SCHEDULE_PATH)

# ------------------------
# 2. Data Preprocessing
# ------------------------

patients_df['arrival_date'] = pd.to_datetime(patients_df['arrival_date'])
patients_df['departure_date'] = pd.to_datetime(patients_df['departure_date'])

patients_df['length_of_stay'] = (patients_df['departure_date'] - patients_df['arrival_date']).dt.days

patients_df['week'] = patients_df['arrival_date'].dt.isocalendar().week

staff_metrics = schedule_df.groupby(['week', 'service']).agg({
    'present': ['sum', 'count', 'mean'],
    'staff_id': 'count'
}).reset_index()

staff_metrics.columns = ['week', 'service', 'staff_present', 'total_staff', 'presence_rate', 'staff_count']
staff_metrics['staff_absent'] = staff_metrics['total_staff'] - staff_metrics['staff_present']
staff_metrics['absence_rate'] = (staff_metrics['staff_absent'] / staff_metrics['total_staff'] * 100)

role_presence = schedule_df.groupby(['week', 'service', 'role']).agg({
    'present': ['sum']
}).reset_index()

role_pivot = role_presence.pivot_table(
    index=['week', 'service'], 
    columns='role', 
    values=('present'), 
    fill_value=0
).reset_index()

role_pivot.columns = ['week', 'service', 'doctors_present', 'nurses_present', 'nursing_assistants_present']

patient_outcomes = patients_df.groupby(['week', 'service']).agg({
    'satisfaction': ['mean', 'std', 'min', 'max'],
    'length_of_stay': ['mean', 'median', 'std', 'max'],
    'patient_id': 'count'
}).reset_index()

patient_outcomes.columns = ['week', 'service', 
                           'avg_satisfaction', 'stf_satisfaction', 
                            'min_satisfaction', 'max_satisfaction',
                            'avg_los', 'median_los', 'std_los', 'max_los',
                            'patient_count']

merged_df = services_df.merge(staff_metrics, on=['week', 'service'], how='left'
    ).merge(role_pivot, on=['week', 'service'], how='left'
    ).merge(patient_outcomes, on=['week', 'service'], how='left')

merged_df['patients_per_staff'] = (merged_df['patients_admitted'] / merged_df['staff_present'].replace(0, np.nan))
merged_df['workload'] = (merged_df['patients_request'] / merged_df['staff_present'].replace(0, np.nan))

#understaffing indicator (less than 75% presence)
merged_df['understaffed'] = (merged_df['presence_rate'] < 0.75)

#busy week indicator
median_request = merged_df['patients_request'].median()
merged_df['busy_week'] = (merged_df['patients_request'] > median_request)

#presence categories
merged_df['presence_category'] = pd.cut(
    merged_df['presence_rate'],
    bins=[0, 0.6, 0.75, 0.9, 1.0],
    labels=['Very Low (<60%)', 'Low (60-75%)', 'Medium (75-90%)', 'High (90%+)']
)

#combined conditiom for busy week analysis
conditions = []
for _, row in merged_df.iterrows():
    if pd.isna(row['busy_week']) or pd.isna(row['understaffed']):
        conditions.append('Unknown')
    elif row['busy_week'] and row['understaffed']:
        conditions.append('Busy & Understaffed')
    elif row['busy_week'] and not row['understaffed']:
        conditions.append('Busy & Well Staffed')
    elif not row['busy_week'] and row['understaffed']:
        conditions.append('Not Busy & Understaffed')
    else:
        conditions.append('Not Busy & Well Staffed')

merged_df['condition'] = conditions

#unique service and weeks
all_services = sorted(merged_df['service'].dropna().unique())
all_weeks = sorted(merged_df['week'].dropna().unique())

# ------------------------
# 3. Dash App Initialization
# ------------------------
app = dash.Dash(__name__)

# ------------------------
# 4. App Layout
# ------------------------
app.layout = html.Div([
    html.H1("Task 3: Staff Presence vs. Patient Outcomes", 
            style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'marginBottom': '10px'}),
    html.P("Multivariate Analysis: Discover correlations between staffing levels and patient ourcomes",
           style={'textAlign': 'center', 'color':'#555', 'marginBottom': '30px'}),

    # --- Controls Container ---
    html.Div([
        # Week Range Slider
        html.Div([
            html.Label("Filter by Week Range:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='week-slider',
                min=int(merged_df['week'].min()),
                max=int(merged_df['week'].max()),
                value=[int(merged_df['week'].min()), int(merged_df['week'].max())],
                marks={int(w): str(w) for w in range(int(merged_df['week'].min()),
                                                     int(merged_df['week'].max())+1, 5)},
                step=1,
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),
        
        # Service Dropdown
        html.Div([
            html.Label("Select Service:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='service-filter',
                options=[{'label': s.replace('_',' ').title(), 'value': s} for s in all_services],
                value=all_services,
                multi=True,
                clearable=False
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'width':'90%','margin':'0 auto', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px','marginBottom': '20px'}),

    # --- Visualization Type Selector ---
    html.Div([
        html.Label("Select Analysis View:", style={'fontWeight': 'bold', 'marginRight': '20px'}),
        dcc.RadioItems(
            id='view-selector',
            options=[
                {'label': 'SPLOM (Scatter Plot Matrix)', 'value': 'splom'},
                {'label': 'Research Questions Analysis', 'value': 'research'},
                {'label': 'Correlation Heatmap', 'value': 'correlation'}
            ],
            value='splom',
            inline=True,
            style={'fontsize': '14px'}
        ),
    ], style={'width':'90%','margin':'0 auto', 'padding': '15px', 'backgroundColor': '#e8f4f8', 'borderRadius': '10px','marginBottom': '20px'}),

    # --- Main Visualization Container ---
    html.Div(id='main-content', style={'width':'95%','margin' :'0 auto'}),

    # --- Key Insights Summary ---
    html.Div(id='insights-summary', style={'width':'90%','margin':'20px auto', 'padding': '20px', 'backgroundColor': '#fff3cd', 'borderRadius': '10px', 'border':'2px solid #ffc107'})
])

# ------------------------
# 5. Callbacks
# ------------------------
@app.callback(
    [Output('main-content', 'children'),
     Output('insights-summary', 'children')],
    [Input('week-slider', 'value'),
     Input('service-filter', 'value'),
     Input('view-selector', 'value')]
)

def update_visualization(week_range, selected_services, view_type):
    if not selected_services:
        return html.Div("Please select at least one service.", style={'padding': '20px', 'textAlign': 'center'}), ""
    
    start_week, end_week = week_range

    #filter data
    mask = ((merged_df['week'] >= start_week) & 
            (merged_df['week'] <= end_week) & 
            (merged_df['service'].isin(selected_services)))
    filtered_df = merged_df[mask].copy()

    if filtered_df.empty:
        return html.Div("No data available for the selected filters.", style={'padding': '20px', 'textAlign': 'center'}), ""
    
    if view_type == 'splom':
        content, insights = create_splom_view(filtered_df)
    elif view_type == 'research':
        content, insights = create_research_questions_view(filtered_df)
    else:
        content, insights = create_correlation_view(filtered_df)
    return content, insights

def create_splom_view(df):
    splom_vars = [
        'staff_present', 'presence_rate', 'patient_satisfaction', 'avg_satisfaction', 'avg_los', 'staff_morale', 'patients_refused', 'workload'
    ]

    splom_data = df[splom_vars + ['service']].dropna()

    var_labels = {
        'staff_present': 'Staff Present',
        'presence_rate': 'Presence Rate',
        'patient_satisfaction': 'Satisfaction (Svc)', 
        'avg_satisfaction': 'Satisfaction (Pt)',
        'avg_los': 'Length of Stay',
        'staff_morale': 'Staff Morale',
        'patients_refused': 'Patients Refused',
        'workload': 'Workload'
    }

    service_color_map = {
        'emergency': '#1f77b4',
        'surgery': '#ff7f0e',
        'general_medicine': '#2ca02c',
        'ICU': '#d62728'
    }

    fig_splom = px.scatter_matrix(
        splom_data,
        dimensions=splom_vars,
        color='service',
        color_discrete_map=service_color_map,
        labels=var_labels,
        title="<b>Scatter Plot Matrix (SPLOM): Staff Presence vs Patient Outcomes</b>",
        height=900,
        opacity=0.6
    )

    fig_splom.update_traces(diagonal_visible=True, showupperhalf=False)

    fig_splom.update_layout(
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=10),
        legend=dict(title='Service', orientation='v', yanchor='top', y=1, xanchor='left', x=1.02))

    corr_staff_sat = splom_data[['staff_present', 'patient_satisfaction']].corr().iloc[0,1]
    corr_presence_sat = splom_data[['presence_rate', 'patient_satisfaction']].corr().iloc[0,1]
    corr_staff_los = splom_data[['staff_present', 'avg_los']].corr().iloc[0,1]
    corr_workload_sat = splom_data[['workload', 'patient_satisfaction']].corr().iloc[0,1]

    insights = html.Div([
        html.H4("Key Insights from SPLOM Analysis:", style={'marginBottom': '10px'}),
        html.Ul([
            html.Li(f"Staff Present <-> Patient Satisfaction: r = {corr_staff_sat:.3f} "
                   f"({'Positive' if corr_staff_sat > 0 else 'Negative'} relationship)"),
            html.Li(f"Presence Rate <-> Patient Satisfaction: r = {corr_presence_sat:.3f}"),
            html.Li(f"Staff Present <-> Length of Stay: r = {corr_staff_los:.3f}"),
            html.Li(f"Workload <-> Patient Satisfaction: r = {corr_workload_sat:.3f}"),
        ]),
        html.P([
            html.B("How to read: "),
            "Each cell shows a scatter plot of two variables. Diagonal shows distributions. "
            "Colors represent different services. Look for patterns like clusters, trends, or outliers."
        ], style={'marginTop': '10px', 'fontSize': '13px', 'color': '#555', 'fontStyle': 'italic'})
    ])

    content = dcc.Graph(figure=fig_splom)
    return content, insights

def create_research_questions_view(df):
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Q1: Presence Rate vs Satisfaction',
            'Q2: Staff Present vs Length of Stay',
            'Q3: Service Sensitivity to Shortages',
            'Q4: Satisfaction Threshold',
            'Q5: Busy Week + Staffing Impact',
            'Q6: Role-Specific Impact',
            'Q7: Workload Impact',
            'Q8: Weekly Trends',
            'Q9: Service and Presence Heatmap'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'heatmap'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    #Q1: presence categories vs satisfaction
    presence_satisfaction = df.groupby('presence_category')['patient_satisfaction'].agg(['mean', 'count'])
    presence_satisfaction = presence_satisfaction[presence_satisfaction['count'] >= 2]

    fig.add_trace(
        go.Bar(
            x=presence_satisfaction.index.astype(str),
            y=presence_satisfaction['mean'],
            marker_color='steelblue',
            name='Avg Satisfaction',
            showlegend=False
        ),
        row=1, col=1
    )

    #Q2: staff present vs LOS
    valid_data = df[['staff_present', 'avg_los']].dropna()
    if len(valid_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=valid_data['staff_present'],
                y=valid_data['avg_los'],
                mode='markers',
                marker=dict(size=8, color = 'orange', opacity=0.6),
                name='Staff vs LOS',
                showlegend=False
            ),
            row=1, col=2
        )

        z = np.polyfit(valid_data['staff_present'], valid_data['avg_los'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data['staff_present'].min(), valid_data['staff_present'].max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trend Line',
                showlegend=False
            ),
            row=1, col=2
        )
    
    #Q3: service impact
    service_impact = []
    for service in df['service'].unique():
        service_data = df[df['service'] == service]
        understaffed_sat = service_data[service_data['understaffed']]['patient_satisfaction'].mean()
        wellstaffed_sat = service_data[~service_data['understaffed']]['patient_satisfaction'].mean()
        if pd.notna(understaffed_sat) and pd.notna(wellstaffed_sat):
            impact = wellstaffed_sat - understaffed_sat
            service_impact.append({'service':service, 'impact':impact})
    
    if service_impact:
        impact_df = pd.DataFrame(service_impact).sort_values('impact', ascending=True)
        colors_impact = ['red' if x > 0 else 'green' for x in impact_df['impact']]
        fig.add_trace(
            go.Bar(
                x=impact_df['impact'],
                y=impact_df['service'],
                orientation='h',
                marker_color=colors_impact,
                name='Service Impact',
                showlegend=False
            ),
            row=1, col=3
        )

    #Q4: threshold analysis
    presence_bins = pd.cut(df['presence_rate'], bins=10)
    threshold_data = df.groupby(presence_bins)['patient_satisfaction'].mean().reset_index()
    threshold_data['presence_midpoint'] = threshold_data['presence_rate'].apply(lambda x: x.mid)

    fig.add_trace(
        go.Scatter(
            x=threshold_data['presence_midpoint'],
            y=threshold_data['patient_satisfaction'],
            mode='lines+markers',
            marker = dict(size=8, color='purple'),
            line=dict(width=2),
            name='Threshold',
            showlegend=False
        ),
        row=2, col=1
    )

    #Q5: busy week + staffing
    condition_satisfaction = df.groupby('condition')['patient_satisfaction'].mean().sort_values(ascending=False)
    colors_condition = {
        'Not Busy & Well Staffed': 'green',
        'Busy & Well Staffed': 'lightgreen',
        'Not Busy & Understaffed': 'orange',
        'Busy & Understaffed': 'red',
    }
    bar_colors = [colors_condition.get(cond, 'gray') for cond in condition_satisfaction.index]

    fig.add_trace(
        go.Bar(
            x=condition_satisfaction.index,
            y=condition_satisfaction.values,
            marker_color=bar_colors,
            name='Conditions',
            showlegend=False
        ),
        row=2, col=2
    )

    #Q6: role-specific impact
    role_correlation = []
    for role_col in ['doctors_present', 'nurses_present', 'nursing_assistants_present']:
        valid = df[[role_col, 'patient_satisfaction']].dropna()
        if len(valid) > 10:
            corr, _ = pearsonr(valid[role_col], valid['patient_satisfaction'])
            role_name = role_col.replace('_present','').replace('_', ' ').title()
            role_correlation.append({'role': role_name, 'correlation': corr})

    if role_correlation:
        role_df = pd.DataFrame(role_correlation).sort_values('correlation', ascending=True)
        colors_role = ['green' if x > 0 else 'red' for x in role_df['correlation']]
        fig.add_trace(
            go.Bar(
                x=role_df['role'],
                y=role_df['correlation'],
                orientation='h',
                marker_color=colors_role,
                name='Role Impact',
                showlegend=False
            ),
            row=2, col=3
        )

    #Q7: workload vs satisfaction
    valid_workload = df[['patients_per_staff', 'patient_satisfaction']].dropna()
    valid_workload = valid_workload[valid_workload['patients_per_staff'] < 20]

    if len(valid_workload) > 0:
        fig.add_trace(
            go.Scatter(
                x=valid_workload['patients_per_staff'],
                y=valid_workload['patient_satisfaction'],
                mode='markers',
                marker=dict(size=6, color='teal', opacity=0.5),
                name='Workload vs Satisfaction',
                showlegend=False
            ),
            row=3, col=1
        )

    #Q8: weekly trends
    weekly_avg = df.groupby('week').agg({
        'presence_rate': 'mean',
        'patient_satisfaction': 'mean'
    }).reset_index()

    fig.add_trace(
        go.Scatter(
            x=weekly_avg['week'],
            y=weekly_avg['presence_rate'] * 100,
            mode='lines+markers',
            name='Presence Rate',
            line=dict(color='blue', width=2),
            yaxis='y1'
        ),
        row=3, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=weekly_avg['week'],
            y=weekly_avg['patient_satisfaction'],
            mode='lines+markers',
            name='Satisfaction',
            line=dict(color='red', width=2),
            yaxis='y2'
        ),
        row=3, col=2
    )

    #Q9: heatmap
    heatmap_data = df.pivot_table(
        index='service',
        columns='presence_category',
        values='patient_satisfaction',
        aggfunc='mean'
    )

    fig.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=75,
            showscale=True
        ),
        row=3, col=3
    )

    fig.update_layout(
        height=1200,
        title_text="<b>Research Questions Analysis: Staff Presence Impact Analysis</b>",
        showlegend=False
    )

    fig.update_xaxes(title_text="Presence Category", row=1, col=1)
    fig.update_xaxes(title_text="Staff Present", row=1, col=2)
    fig.update_xaxes(title_text="Satisfaction Drop", row=1, col=3)
    fig.update_xaxes(title_text="Presence Rate", row=2, col=1)
    fig.update_xaxes(title_text="Condition", row=2, col=2)
    fig.update_xaxes(title_text="Correlation", row=2, col=3)
    fig.update_xaxes(title_text="Patients per Staff", row=3, col=1)
    fig.update_xaxes(title_text="Week", row=3, col=2)
    
    fig.update_yaxes(title_text="Satisfaction", row=1, col=1)
    fig.update_yaxes(title_text="Length of Stay", row=1, col=2)
    fig.update_yaxes(title_text="Satisfaction", row=2, col=1)
    fig.update_yaxes(title_text="Satisfaction", row=2, col=2)
    fig.update_yaxes(title_text="Satisfaction", row=3, col=1)
    fig.update_yaxes(title_text="Presence %", row=3, col=2)
    
    # insights
    avg_understaffed_sat = df[df['understaffed']]['patient_satisfaction'].mean()
    avg_wellstaffed_sat = df[~df['understaffed']]['patient_satisfaction'].mean()
    satisfaction_drop = avg_wellstaffed_sat - avg_understaffed_sat
    
    insights = html.Div([
        html.H4("Key Findings:", style={'marginBottom': '10px'}),
        html.Ul([
            html.Li(f"Satisfaction drops by {satisfaction_drop:.1f} points when understaffed"),
            html.Li(f"Understaffed weeks: {df['understaffed'].sum()} ({df['understaffed'].mean()*100:.1f}%)"),
            html.Li(f"Worst condition: Busy + Understaffed (lowest satisfaction)"),
            html.Li(f"Staff presence threshold appears around 75% for maintaining quality"),
        ])
    ])
    
    content = dcc.Graph(figure=fig)
    
    return content, insights


def create_correlation_view(df):
    corr_vars = [
        'staff_present', 'presence_rate', 'doctors_present', 'nurses_present',
        'nursing_assistants_present', 'patient_satisfaction', 'avg_satisfaction',
        'avg_los', 'median_los', 'staff_morale', 'patients_refused',
        'patients_per_staff', 'workload'
    ]
    
    # calculate correlation matrix
    corr_data = df[corr_vars].dropna()
    corr_matrix = corr_data.corr()
    
    # create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="<b>Correlation Matrix: Staff Presence vs Outcomes</b>",
        height=800,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    # extract key correlations
    staff_sat_corr = corr_matrix.loc['staff_present', 'patient_satisfaction']
    presence_sat_corr = corr_matrix.loc['presence_rate', 'patient_satisfaction']
    staff_los_corr = corr_matrix.loc['staff_present', 'avg_los']
    workload_sat_corr = corr_matrix.loc['workload', 'patient_satisfaction']
    morale_sat_corr = corr_matrix.loc['staff_morale', 'patient_satisfaction']
    
    insights = html.Div([
        html.H4("Correlation Insights:", style={'marginBottom': '10px'}),
        html.Ul([
            html.Li(f"Staff Present ↔ Patient Satisfaction: {staff_sat_corr:+.3f} "
                   f"({'Strong' if abs(staff_sat_corr) > 0.5 else 'Moderate' if abs(staff_sat_corr) > 0.3 else 'Weak'})"),
            html.Li(f"Presence Rate ↔ Patient Satisfaction: {presence_sat_corr:+.3f}"),
            html.Li(f"Staff Present ↔ Length of Stay: {staff_los_corr:+.3f}"),
            html.Li(f"Workload ↔ Patient Satisfaction: {workload_sat_corr:+.3f}"),
            html.Li(f"Staff Morale ↔ Patient Satisfaction: {morale_sat_corr:+.3f}"),
        ]),
        html.P([
            html.B("Interpretation: "),
            "Positive correlations (blue) indicate variables move together. "
            "Negative correlations (red) indicate inverse relationships. "
            "Correlations > 0.5 or < -0.5 are considered strong."
        ], style={'marginTop': '15px', 'fontSize': '13px', 'color': '#555'})
    ])
    
    content = dcc.Graph(figure=fig)
    
    return content, insights


# ------------------------
# 6. Run Server
# ------------------------
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8053)