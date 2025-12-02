import plotly.graph_objects as go

ROLES = ['doctor', 'nurse', 'nursing_assistant']
COLORS = ['#636EFA', '#EF553B', '#00CC96']

def create_allocation_chart(df, service_name):
    """Generates Stacked Bar (Staff) + Line (Patients)."""
    
    df = df.sort_values('week')
    
    fig = go.Figure()

    # 1. Stacked Bars for Staff Roles
    for i, role in enumerate(ROLES):
        if role in df.columns:
            fig.add_trace(go.Bar(
                x=df['week'],
                y=df[role],
                name=role.replace('_',' ').title(),
                marker_color=COLORS[i % len(COLORS)]
            ))

    # 2. Line for Patient Load
    fig.add_trace(go.Scatter(
        x=df['week'],
        y=df['patients_admitted'],
        name='Admitted Patients',
        mode='lines',
        line=dict(color='black', width=3),
        yaxis='y2'
    ))

    fig.update_layout(
        title=f"Staff Allocation vs Load: {service_name}",
        yaxis=dict(title="Staff Count"),
        yaxis2=dict(title="Patients", overlaying='y', side='right'),
        barmode='stack',
        height=400,
        legend=dict(orientation="h", y=1.1)
    )
    return fig