import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ['occupancy_rate', 'patient_satisfaction', 'staff_morale']
LABELS = ['Occupancy %', 'Satisfaction', 'Morale']

def perform_clustering(df, k):
    """Runs KMeans and returns DF with 'cluster' column + centroids."""
    scaler = StandardScaler()
    matrix = scaler.fit_transform(df[FEATURES])
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_clusters = kmeans.fit_predict(matrix)
    
    # Sort Clusters by Occupancy (0 = Low Stress, K = High Stress)
    df_temp = pd.DataFrame({'raw': raw_clusters, 'occ': df['occupancy_rate']})
    mapping = {r: i for i, r in enumerate(df_temp.groupby('raw')['occ'].mean().sort_values().index)}
    
    df_out = df.copy()
    df_out['cluster'] = df_temp['raw'].map(mapping).astype(str)
    
    # Calculate Centroids (Scaled for Heatmap)
    df_scaled = pd.DataFrame(matrix, columns=FEATURES)
    df_scaled['cluster'] = df_out['cluster']
    centroids = df_scaled.groupby('cluster')[FEATURES].mean()
    
    return df_out, centroids

def create_bubble_chart(df, k, colors):
    fig = px.scatter(
        df, x='occupancy_rate', y='patient_satisfaction', size='patients_admitted',
        color='cluster', symbol='cluster',
        color_discrete_sequence=colors,
        category_orders={"cluster": [str(i) for i in range(k)]},
        title="<b>State Space:</b> Strain vs Quality (Size=Volume)",
        hover_data=['service', 'week']
    )
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_dna_heatmap(centroids, k):
    fig = go.Figure(data=go.Heatmap(
        z=centroids.values,
        x=LABELS,
        y=[f"Cluster {i}" for i in range(k)],
        colorscale='RdBu_r', zmid=0,
        text=centroids.round(1).values, texttemplate="%{text}",
        colorbar=dict(title="Deviation")
    ))
    fig.update_layout(
        title="<b>Cluster DNA</b> (Red=Stress)", height=250,
        yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_timeline(df, k, colors):
    fig = px.scatter(
        df, x='week', y='service', color='cluster', symbol='cluster',
        color_discrete_sequence=colors,
        category_orders={"cluster": [str(i) for i in range(k)]},
        title="<b>Timeline:</b> Crisis Patterns"
    )
    fig.update_traces(marker=dict(size=12, symbol='square'))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig