import dash
from dash import dcc, html

from diagrams import tab1, tab2, tab3
from loader import load_data
 
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
        dcc.Tab(label='Capacity & Seasonality Analysis', children=tab1.create_layout(df)),

        # --- TASK 2 ---
        dcc.Tab(label='Staff Performance & Allocation', children=tab2.create_layout(df)),

        # --- TASK 3 ---
        dcc.Tab(label='Strategic Operational Clustering', children=tab3.create_layout(df)),
    ], style={'fontWeight': 'bold', 'fontSize': '18px'})
])

# --- CALLBACKS ---

# Task 1
tab1.register_callbacks(app, df)

# Task 2
tab2.register_callbacks(app, df)

# Task 3
tab3.register_callbacks(app, df)

if __name__ == '__main__':
    app.run(debug=True)