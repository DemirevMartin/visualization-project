import dash
from dash import dcc, html, Input, Output

from loader import load_data
from diagrams import diagram1, diagram3, diagram5

# 1. Load Data
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
        # --- TASK 1 ---
        dcc.Tab(label='1. Linked Views', children=diagram1.create_layout(df)),

        # --- TASK 3 ---
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

        # --- TASK 5 ---
        dcc.Tab(label='5. Operational Strategy', children=diagram5.create_layout(df)),
    ], style={'fontFamily': 'Arial'})
])

# --- CALLBACKS ---

# Task 1
diagram1.register_callbacks(app, df)

# Task 3
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

# Task 5
diagram5.register_callbacks(app, df)

if __name__ == '__main__':
    app.run(debug=True)