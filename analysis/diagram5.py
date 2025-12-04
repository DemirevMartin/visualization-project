import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------
# 1. Config / Data Load
# ------------------------
CSV_PATH = "../data/services_weekly.csv"

df = pd.read_csv(CSV_PATH)

# ------------------------
# 2. Features
# ------------------------
# Occupancy Rate
df['occupancy_rate'] = df.apply(
    lambda x: (x['patients_admitted'] / x['available_beds']) * 100 if x['available_beds'] > 0 else 0, 
    axis=1
)

# ------------------------
# 3. Dash App Initialization
# ------------------------
app = dash.Dash(__name__)

# ------------------------
# 4. App Layout
# ------------------------
app.layout = html.Div([
    html.H1("Strategic Operational Clustering (Complete View)", 
            style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif'}),
    
    html.P("Identify operational patterns. Cluster 0 = Lowest Stress, Cluster K = Highest Stress.",
           style={'textAlign': 'center', 'color': '#555'}),

    # Control
    html.Div([
        html.Label("Number of Clusters (k):", style={'fontWeight': 'bold'}),
        dcc.Slider(
            id='k-slider',
            min=2, max=5, value=3,
            marks={i: str(i) for i in range(2, 6)},
            step=1
        ),
    ], style={'width': '50%', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}),

    # --- Main Dashboard Area ---
    html.Div([
        # LEFT COLUMN: Bubble Chart (The "State Space")
        html.Div([
            html.H3("1. Operational State Map", style={'textAlign': 'center', 'fontSize': '16px'}),
            dcc.Graph(id='bubble-chart')
        ], style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # RIGHT COLUMN: Heatmaps (DNA & Timeline)
        html.Div([
            html.H3("2. Cluster DNA (Profile)", style={'textAlign': 'center', 'fontSize': '16px'}),
            dcc.Graph(id='heatmap-dna', style={'height': '300px'}),
            
            html.H3("3. Timeline (Patterns)", style={'textAlign': 'center', 'fontSize': '16px', 'marginTop': '20px'}),
            dcc.Graph(id='heatmap-timeline', style={'height': '350px'})
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
    ], style={'width': '95%', 'margin': '0 auto', 'marginTop': '20px'})
])

# ------------------------
# 5. Callbacks
# ------------------------
@app.callback(
    [Output('bubble-chart', 'figure'),
     Output('heatmap-dna', 'figure'),
     Output('heatmap-timeline', 'figure')],
    [Input('k-slider', 'value')]
)
def update_clustering(k):
    # ----------------------------------------------------
    # 1. Feature Selection
    # ----------------------------------------------------
    df_viz = df.copy()
    
    # Strictly use Rates/Scores to avoid "Size" bias
    cluster_features = ['occupancy_rate', 'patient_satisfaction', 'staff_morale']
    feature_labels = ['Occupancy %', 'Pt Satisfaction', 'Staff Morale']

    # Normalize
    scaler = StandardScaler()
    data_matrix = scaler.fit_transform(df_viz[cluster_features])
    
    # ----------------------------------------------------
    # 2. Run K-Means
    # ----------------------------------------------------
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_clusters = kmeans.fit_predict(data_matrix)
    
    # ----------------------------------------------------
    # 3. RE-ORDER CLUSTERS (Sort by Occupancy)
    # ----------------------------------------------------
    # Calculate average Occupancy for each raw cluster ID
    df_temp = pd.DataFrame({'raw_cluster': raw_clusters, 'occupancy': df_viz['occupancy_rate']})
    cluster_stats = df_temp.groupby('raw_cluster')['occupancy'].mean().sort_values().reset_index()
    
    # Map Raw ID -> Sorted ID (0 = Lowest Occupancy, K = Highest Occupancy)
    mapping = {row['raw_cluster']: i for i, row in cluster_stats.iterrows()}
    
    # Apply mapping
    df_viz['cluster'] = df_temp['raw_cluster'].map(mapping)
    df_viz['cluster_label'] = df_viz['cluster'].astype(str)

    # Define Colors: Green -> Orange -> Red
    if k == 2:
        colors = ['#2ca02c', '#d62728']
    elif k == 3:
        colors = ['#2ca02c', "#0e66ff", '#d62728']
    elif k == 4:
        colors = ['#2ca02c', "#0e66ff", '#d62728', "#43008b"]
    else:
        colors = ['#2ca02c', "#0e66ff", '#d62728', "#43008b", "#F32ADB"]

    # ----------------------------------------------------
    # 4. Generate Figures
    # ----------------------------------------------------
    
    # --- FIGURE 1: STRATEGIC BUBBLE CHART (Resurrected) ---
    fig_bubble = px.scatter(
        df_viz,
        x='occupancy_rate',
        y='patient_satisfaction',
        size='patients_admitted', # Size = Volume
        color='cluster_label',
        symbol='cluster_label',
        color_discrete_map={str(i): c for i, c in enumerate(colors) if i < len(colors)},
        category_orders={"cluster_label": [str(i) for i in range(k)]},
        title="<b>State Space:</b> Strain vs. Quality (Size=Volume)",
        hover_data=['service', 'week', 'staff_morale']
    )
    fig_bubble.update_layout(
        height=700, # Tall to match the two heatmaps on the right
        legend_title="Stress Level",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # --- FIGURE 2: CLUSTER DNA HEATMAP ---
    # Z-Scores for heatmap intensity
    df_scaled = pd.DataFrame(data_matrix, columns=cluster_features)
    df_scaled['cluster'] = df_viz['cluster']
    centroids = df_scaled.groupby('cluster')[cluster_features].mean()
    
    # Real values for text overlay
    real_centroids = df_viz.groupby('cluster')[cluster_features].mean().round(1)
    text_values = real_centroids.values.astype(str)
    
    fig_dna = go.Figure(data=go.Heatmap(
        z=centroids.values,
        x=feature_labels,
        y=[f"Cluster {i}" for i in range(k)],
        colorscale='RdBu_r', 
        zmid=0,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    fig_dna.update_layout(
        title="<b>Cluster DNA</b> (Red=High Stress)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(autorange="reversed")
    )

    # --- FIGURE 3: TIMELINE HEATMAP ---
    fig_timeline = px.scatter(
        df_viz, x='week', y='service', color='cluster_label', symbol='cluster_label',
        color_discrete_map={str(i): c for i, c in enumerate(colors) if i < len(colors)},
        category_orders={"cluster_label": [str(i) for i in range(k)]},
        title="<b>Timeline:</b> Crisis Patterns",
        hover_data=['occupancy_rate']
    )
    fig_timeline.update_traces(marker=dict(size=12, symbol='square'))
    fig_timeline.update_layout(
        height=350,
        xaxis_title="Week", 
        yaxis_title="Service",
        showlegend=False # Legend already on bubble chart
    )

    return fig_bubble, fig_dna, fig_timeline

# ------------------------
# 6. Run Server
# ------------------------
if __name__ == '__main__':
    app.run(debug=True)