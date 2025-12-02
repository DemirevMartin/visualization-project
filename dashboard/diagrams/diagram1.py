import plotly.graph_objects as go

COLORS = ['#1f77b4', '#d62728'] # Blue, Red
COLORSCALE = [[0.0, COLORS[0]], [0.5, COLORS[0]], [0.5, COLORS[1]], [1.0, COLORS[1]]]

METRICS_CONFIG = [
    {'col': 'patient_satisfaction', 'label': 'Satisfaction', 'range': [0, 100]}, 
    {'col': 'staff_morale',         'label': 'Staff Morale', 'range': [0, 100]}, 
    # Ranges for counts will be dynamic based on data max
    {'col': 'patients_admitted',    'label': 'Admitted'}, 
    {'col': 'patients_refused',     'label': 'Refusals'}   
]

def create_parallel_coords(df, service_name):
    """Generates the Parallel Coordinates plot for a single service."""
    
    # Dynamic range for counts
    dimensions = []
    for conf in METRICS_CONFIG:
        # If range is fixed, use it. If not, calculate max from data.
        r = conf.get('range', [0, df[conf['col']].max()])
        dimensions.append(dict(range=r, label=conf['label'], values=df[conf['col']]))

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df['status_id'],
            colorscale=COLORSCALE,
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(
                title='Status', tickvals=[0.25, 0.75],
                ticktext=['Sufficient', 'Shortage'], len=0.6, thickness=15
            )
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title=f"<b>{service_name}</b>",
        height=350,
        margin=dict(l=40, r=40, t=50, b=20)
    )
    return fig