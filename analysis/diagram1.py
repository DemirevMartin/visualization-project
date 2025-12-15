import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd

# ------------------------
# 1. Config / Data Load
# ------------------------
CSV_PATH = "../data/services_weekly.csv"

df = pd.read_csv(CSV_PATH)

# ------------------------
# 2. Data Preprocessing
# ------------------------

# Define Availability Status
df['availability_status'] = df.apply(
    lambda row: 'Shortage' if row['patients_request'] > row['available_beds'] else 'Sufficient', 
    axis=1
)

# Map Status to Color IDs: 0 = Sufficient (Blue), 1 = Shortage (Red)
status_map = {'Sufficient': 0, 'Shortage': 1}
df['status_id'] = df['availability_status'].map(status_map)

# Distinct Colors for the lines
# Blue for Good, Red for Bad
COLORS = ['#1f77b4', '#d62728'] 

# Create Discrete Colorscale
COLORSCALE = [
    [0.0, COLORS[0]],
    [0.5, COLORS[0]],
    [0.5, COLORS[1]],
    [1.0, COLORS[1]]
]

# METRICS CONFIGURATION
# We do NOT normalize. We define specific ranges for the axes to make them comparable.
metrics_config = [
    {'col': 'patient_satisfaction', 'label': 'Satisfaction', 'range': [0, 100]}, # Fixed 0-100 score
    {'col': 'staff_morale',         'label': 'Staff Morale', 'range': [0, 100]}, # Fixed 0-100 score
    {'col': 'patients_admitted',    'label': 'Admitted',     'range': [0, df['patients_admitted'].max()]}, # 0 to Max
    {'col': 'patients_refused',     'label': 'Refusals',     'range': [0, df['patients_refused'].max()]}   # 0 to Max
]

# List of all unique services for the dropdown
all_services = sorted(df['service'].unique())

# ------------------------
# 3. Dash App Initialization
# ------------------------
app = dash.Dash(__name__)

# ------------------------
# 4. App Layout
# ------------------------
app.layout = html.Div([
    html.H1("Task 1: Bed Capacity & Service Performance (Raw Values)", 
            style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'marginBottom': '30px'}),
    
    # --- Controls Container ---
    html.Div([
        # Week Range Slider
        html.Div([
            html.Label("Filter by Week Range:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='week-slider',
                min=df['week'].min(),
                max=df['week'].max(),
                value=[df['week'].min(), df['week'].max()],
                marks={str(w): str(w) for w in range(df['week'].min(), df['week'].max()+1, 5)},
                step=1
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),
        
        # Service Dropdown
        html.Div([
            html.Label("Select Services to Display:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='service-filter',
                options=[{'label': s.replace('_', ' ').title(), 'value': s} for s in all_services],
                value=all_services,
                multi=True,
                clearable=False
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
    ], style={'width': '90%', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # --- Graphs Container ---
    html.Div(id='graphs-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
])

# ------------------------
# 5. Callbacks
# ------------------------
@app.callback(
    Output('graphs-container', 'children'),
    [Input('week-slider', 'value'),
     Input('service-filter', 'value')]
)
def update_graphs(week_range, selected_services):
    if not selected_services:
        return html.Div("Please select at least one service.", style={'padding': '20px'})

    start_week, end_week = week_range
    
    # Filter by Week
    mask = (df['week'] >= start_week) & (df['week'] <= end_week)
    filtered_df = df[mask]
    
    graphs = []
    
    for service in sorted(selected_services):
        subset = filtered_df[filtered_df['service'] == service]
        
        if subset.empty:
            continue
            
        dimensions = [
            dict(range=conf['range'], label=conf['label'], values=subset[conf['col']]) 
            for conf in metrics_config
        ]

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=subset['status_id'],
                colorscale=COLORSCALE,
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title='Load Status',
                    tickvals=[0.25, 0.75],
                    ticktext=['Sufficient', 'Shortage'],
                    len=0.6,
                    thickness=15
                )
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title=f"<b>{service.replace('_', ' ').title()}</b>",
            height=400,
            margin=dict(l=50, r=50, t=60, b=20)
        )
        
        graphs.append(html.Div(
            dcc.Graph(figure=fig),
            style={'width': '48%', 'minWidth': '600px', 'padding': '10px'}
        ))
    
    return graphs

# ------------------------
# 6. Run Server
# ------------------------
if __name__ == '__main__':
    app.run(debug=True)