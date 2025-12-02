import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Import custom modules
from loader import load_data
# Ensure you have files diagram1.py, diagram2.py, etc. in the 'dashboard/diagrams' folder
from diagrams import diagram1, diagram2, diagram3, diagram4, diagram5

# 1. Load Data (Rich Dataset)
df = load_data()
all_services = sorted(df['service'].unique())

# 2. Init App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# 3. Layout
app.layout = html.Div([
    html.Div([
        html.H1("Hospital Operations Dashboard"),
        html.P("Integrated Analytics for Capacity, Seasonality, Staffing, and Strategy.")
    ], className="header"),

    dcc.Tabs([
        # --- TASK 1: CAPACITY ---
        dcc.Tab(label='1. Bed Capacity', children=[
            html.Div([
                html.Label("Filter Week Range:", className="control-label"),
                dcc.RangeSlider(
                    id='t1-slider', min=df['week'].min(), max=df['week'].max(), step=1,
                    value=[1, 52], marks={i: str(i) for i in range(0, 53, 5)}
                ),
                dcc.Dropdown(
                    id='t1-dropdown', options=[{'label': s, 'value': s} for s in all_services],
                    value=all_services, multi=True
                ),
                html.Div(id='t1-graphs-container', style={'display': 'flex', 'flexWrap': 'wrap'})
            ], className="card")
        ]),

        # --- TASK 2: SEASONALITY ---
        dcc.Tab(label='2. Seasonality', children=[
            html.Div([
                dcc.Dropdown(
                    id='t2-metric',
                    options=[
                        {'label': 'Admissions', 'value': 'patients_admitted'},
                        {'label': 'Satisfaction', 'value': 'patient_satisfaction'},
                        {'label': 'Staff Morale', 'value': 'staff_morale'}
                    ], value='patients_admitted', clearable=False
                ),
                dcc.RadioItems(
                    id='t2-agg', options=[{'label': 'Month', 'value': 'month'}, {'label': 'Quarter', 'value': 'quarter'}],
                    value='month', inline=True
                ),
                dcc.Graph(id='t2-heatmap')
            ], className="card")
        ]),

        # --- TASK 3: STAFF & OUTCOMES (New Integration) ---
        dcc.Tab(label='3. Staff vs Outcomes', children=[
            html.Div([
                html.Div([
                    html.Label("Week Filter:", className="control-label"),
                    dcc.RangeSlider(
                        id='t3-slider', min=df['week'].min(), max=df['week'].max(), step=1,
                        value=[1, 52], marks={i: str(i) for i in range(0, 53, 10)}
                    ),
                    html.Label("Service Filter:", className="control-label"),
                    dcc.Dropdown(
                        id='t3-dropdown', options=[{'label': s, 'value': s} for s in all_services],
                        value=all_services, multi=True
                    ),
                    html.Label("Analysis View:", className="control-label"),
                    dcc.RadioItems(
                        id='t3-view',
                        options=[
                            {'label': 'SPLOM (Multivariate)', 'value': 'splom'},
                            {'label': 'Research Questions (Grid)', 'value': 'research'},
                            {'label': 'Correlations', 'value': 'correlation'}
                        ], value='splom', inline=True
                    )
                ], className="control-panel"),
                
                html.Div(id='t3-content'),
                html.Div(id='t3-insights', style={'padding': '15px', 'backgroundColor': '#fff3cd', 'marginTop': '15px'})
            ], className="card")
        ]),

        # --- TASK 4: STAFF ALLOCATION ---
        dcc.Tab(label='4. Staff Roles', children=[
            html.Div([
                dcc.Dropdown(
                    id='t4-service', options=[{'label': s, 'value': s} for s in all_services],
                    value='emergency', clearable=False
                ),
                dcc.Graph(id='t4-chart')
            ], className="card")
        ]),

        # --- TASK 5: CLUSTERING ---
        dcc.Tab(label='5. Operational Strategy', children=[
            html.Div([
                html.Label("Number of Clusters (k):"),
                dcc.Slider(id='t5-slider', min=2, max=5, step=1, value=3),
                html.Div([
                    html.Div(dcc.Graph(id='t5-bubble'), style={'width': '58%', 'display': 'inline-block'}),
                    html.Div(dcc.Graph(id='t5-dna'), style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ]),
                dcc.Graph(id='t5-timeline')
            ], className="card")
        ]),
    ], style={'fontFamily': 'Arial'})
])

# --- CALLBACKS ---

# Task 1
@app.callback(Output('t1-graphs-container', 'children'), [Input('t1-slider', 'value'), Input('t1-dropdown', 'value')])
def update_t1(weeks, services):
    if not services: return html.Div("Select a service.")
    sub = df[(df['week'] >= weeks[0]) & (df['week'] <= weeks[1])]
    graphs = []
    for s in services:
        s_data = sub[sub['service'] == s]
        if s_data.empty: continue
        graphs.append(html.Div(dcc.Graph(figure=diagram1.create_parallel_coords(s_data, s)), style={'width': '48%', 'padding': '5px'}))
    return graphs

# Task 2
@app.callback(Output('t2-heatmap', 'figure'), [Input('t2-metric', 'value'), Input('t2-agg', 'value')])
def update_t2(metric, agg):
    return diagram2.create_seasonal_heatmap(df.copy(), metric, agg)

# Task 3 (NEW)
@app.callback([Output('t3-content', 'children'), Output('t3-insights', 'children')],
              [Input('t3-slider', 'value'), Input('t3-dropdown', 'value'), Input('t3-view', 'value')])
def update_t3(weeks, services, view):
    if not services: return html.Div("Select a service"), ""
    mask = (df['week'] >= weeks[0]) & (df['week'] <= weeks[1]) & (df['service'].isin(services))
    sub = df[mask].copy()
    if sub.empty: return html.Div("No data"), ""
    
    if view == 'splom': return diagram3.create_splom_view(sub)
    elif view == 'research': return diagram3.create_research_view(sub)
    else: return diagram3.create_correlation_view(sub)

# Task 4
@app.callback(Output('t4-chart', 'figure'), [Input('t4-service', 'value')])
def update_t4(service):
    return diagram4.create_allocation_chart(df[df['service'] == service], service)

# Task 5
@app.callback([Output('t5-bubble', 'figure'), Output('t5-dna', 'figure'), Output('t5-timeline', 'figure')],
              [Input('t5-slider', 'value')])
def update_t5(k):
    df_c, centroids = diagram5.perform_clustering(df, k)
    if k == 2: colors = ['#2ca02c', '#d62728']
    elif k == 3: colors = ['#2ca02c', '#ff7f0e', '#d62728']
    else: colors = px.colors.sequential.Viridis[::-1]
    
    return (diagram5.create_bubble_chart(df_c, k, colors),
            diagram5.create_dna_heatmap(centroids, k),
            diagram5.create_timeline(df_c, k, colors))

if __name__ == '__main__':
    app.run(debug=True)