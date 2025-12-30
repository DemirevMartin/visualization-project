import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash import html, dcc
from scipy.stats import pearsonr

def create_splom_view(df):
    """
    Creates the Scatter Plot Matrix (SPLOM) for Task 3.
    """
    splom_vars = [
        'staff_present', 'presence_rate', 'patient_satisfaction', 
        'avg_satisfaction', 'avg_los', 'staff_morale', 
        'patients_refused', 'workload'
    ]
    
    # Filter only existing columns
    valid_vars = [v for v in splom_vars if v in df.columns]
    splom_data = df[valid_vars + ['service']].dropna()

    dimensions = []
    var_labels = {
        'staff_present': 'Staff Count',
        'presence_rate': 'Presence %',
        'patient_satisfaction': 'Sat (Svc)', 
        'avg_satisfaction': 'Sat (Pt)',
        'avg_los': 'LOS',
        'staff_morale': 'Morale',
        'patients_refused': 'Refusals',
        'workload': 'Workload'
    }

    for var in valid_vars:
        dimensions.append(dict(label=var_labels.get(var, var), values=splom_data[var]))

    # Map colors
    service_map = {s: i for i, s in enumerate(sorted(df['service'].unique()))}
    splom_data['service_id'] = splom_data['service'].map(service_map)
    
    # Use distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=splom_data['service_id'],
            colorscale=[[i/(len(service_map)-1 or 1), c] for i, c in enumerate(colors[:len(service_map)])],
            showscale=True,
            colorbar=dict(title='Service', tickvals=list(service_map.values()), ticktext=list(service_map.keys()))
        ),
        dimensions=dimensions
    ))

    fig.update_layout(
        title="<b>Multivariate Staffing Analysis (Parallel Coordinates)</b>",
        height=600,
        margin=dict(l=60, r=60, t=80, b=40)
    )
    
    # Insights
    corr_staff_sat = df[['staff_present', 'patient_satisfaction']].corr().iloc[0,1]
    
    insights = html.Div([
        html.H4("Key SPLOM Insights:"),
        html.Ul([
            html.Li(f"Staff Presence vs Satisfaction correlation: r = {corr_staff_sat:.3f}"),
            html.Li("Use the interactive axes above to filter ranges and see connections across metrics.")
        ])
    ])

    return dcc.Graph(figure=fig), insights

def create_research_view(df):
    """
    Creates the Multi-Panel Research Questions view.
    """
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Presence vs Satisfaction', 'Staff vs LOS', 'Service Sensitivity',
            'Satisfaction Threshold', 'Condition Impact', 'Role Impact',
            'Workload Impact', 'Weekly Trends', 'Presence Heatmap'
        ),
        vertical_spacing=0.1, horizontal_spacing=0.08
    )

    # Q1: Presence vs Satisfaction
    if 'presence_category' in df.columns:
        p_sat = df.groupby('presence_category')['patient_satisfaction'].mean()
        fig.add_trace(go.Bar(x=p_sat.index.astype(str), y=p_sat.values, marker_color='#3366cc', name='Sat by Presence'), row=1, col=1)

    # Q2: Staff vs LOS
    fig.add_trace(go.Scatter(x=df['staff_present'], y=df['avg_los'], mode='markers', marker=dict(color='orange', opacity=0.6), name='Staff vs LOS'), row=1, col=2)

    # Q3: Service Impact (Well-staffed vs Understaffed)
    if 'understaffed' in df.columns:
        impacts = []
        for s in df['service'].unique():
            sub = df[df['service'] == s]
            diff = sub[~sub['understaffed']]['patient_satisfaction'].mean() - sub[sub['understaffed']]['patient_satisfaction'].mean()
            impacts.append({'service': s, 'diff': diff})
        imp_df = pd.DataFrame(impacts).sort_values('diff')
        fig.add_trace(go.Bar(y=imp_df['service'], x=imp_df['diff'], orientation='h', marker_color='teal', name='Impact'), row=1, col=3)

    # Q4: Threshold
    # (Simplified for brevity)
    fig.add_trace(go.Scatter(x=df['presence_rate'], y=df['patient_satisfaction'], mode='markers', marker=dict(size=4, color='purple'), name='Threshold'), row=2, col=1)

    # Q5: Condition
    if 'condition' in df.columns:
        cond_sat = df.groupby('condition')['patient_satisfaction'].mean().sort_values()
        fig.add_trace(go.Bar(x=cond_sat.index, y=cond_sat.values, marker_color='crimson', name='Condition'), row=2, col=2)

    # Q6: Role Impact (Correlation)
    roles = ['doctor', 'nurse', 'nursing_assistant']
    corrs = []
    for r in roles:
        if r in df.columns:
            c = df[[r, 'patient_satisfaction']].corr().iloc[0,1]
            corrs.append({'role': r, 'c': c})
    c_df = pd.DataFrame(corrs)
    if not c_df.empty:
        fig.add_trace(go.Bar(x=c_df['role'], y=c_df['c'], marker_color='green', name='Role Corr'), row=2, col=3)

    # Q7: Workload
    if 'workload' in df.columns:
        fig.add_trace(go.Scatter(x=df['workload'], y=df['patient_satisfaction'], mode='markers', marker=dict(color='brown'), name='Workload'), row=3, col=1)

    # Q8: Weekly Trend
    daily = df.groupby('week')[['patient_satisfaction', 'presence_rate']].mean().reset_index()
    fig.add_trace(go.Scatter(x=daily['week'], y=daily['patient_satisfaction'], mode='lines', line=dict(color='blue'), name='Sat Trend'), row=3, col=2)
    fig.add_trace(go.Scatter(x=daily['week'], y=daily['presence_rate']*100, mode='lines', line=dict(color='red', dash='dot'), name='Pres %'), row=3, col=2)

    # Q9: Heatmap
    if 'presence_category' in df.columns:
        hm = df.pivot_table(index='service', columns='presence_category', values='patient_satisfaction')
        fig.add_trace(go.Heatmap(z=hm.values, x=hm.columns.astype(str), y=hm.index, colorscale='RdYlGn'), row=3, col=3)

    fig.update_layout(height=900, showlegend=False, title_text="<b>Comprehensive Staffing Research Analysis</b>")
    
    return dcc.Graph(figure=fig), html.Div([html.H4("Analysis"), html.P("This matrix explores 9 key dimensions of staffing impact.")])

def create_correlation_view(df):
    """
    Creates the Correlation Heatmap.
    """
    # Select numeric columns
    cols = ['staff_present', 'presence_rate', 'doctor', 'nurse', 'workload', 
            'patient_satisfaction', 'avg_los', 'staff_morale', 'patients_refused']
    
    valid_cols = [c for c in cols if c in df.columns]
    corr_matrix = df[valid_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=valid_cols,
        y=valid_cols,
        colorscale='RdBu', zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        colorbar=dict(title="Pearson r")
    ))

    fig.update_layout(title="<b>Correlation Matrix</b>", height=600)
    
    return dcc.Graph(figure=fig), html.Div([html.H4("Correlation Map"), html.P("Red = Negative Correlation, Blue = Positive Correlation.")])