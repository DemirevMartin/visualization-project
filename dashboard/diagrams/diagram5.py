import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------
# Layout
# ------------------------
def create_layout(df):
    # Preprocessing specific to this diagram (if not in loader)
    # Ensure occupancy_rate exists (loader has it, but let's be safe or use it)
    if 'occupancy_rate' not in df.columns:
        df['occupancy_rate'] = df.apply(
            lambda x: (x['patients_admitted'] / x['available_beds']) * 100 if x['available_beds'] > 0 else 0, 
            axis=1
        )

    available_services = sorted(df['service'].unique())
    available_events = sorted(df['event'].astype(str).unique())
    min_week = df['week'].min()
    max_week = df['week'].max()

    return html.Div([
        html.H1("Strategic Operational Clustering", 
                style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif'}),
        
        html.P("Identify operational patterns. Cluster 0 = Lowest Stress, Cluster K = Highest Stress.",
               style={'textAlign': 'center', 'color': '#555'}),

        # --- Control Panel ---
        html.Div([
            # Row 1: K-Means & Cluster Focus
            html.Div([
                html.Div([
                    html.Label("Number of Clusters (k):", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='d5-k-slider',
                        min=2, max=5, value=3,
                        marks={i: str(i) for i in range(2, 6)},
                        step=1
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label("Focus on Cluster:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='d5-cluster-filter',
                        className='cluster-filter',
                        options=[], # Populated by callback
                        placeholder="Select a cluster to highlight...",
                        clearable=True
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '4%'}),
            ], style={'marginBottom': '20px'}),

            # Row 2: Filters (Service, Event, Week)
            html.Div([
                html.Div([
                    html.Label("Filter Services:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='d5-service-filter',
                        className='service-filter',
                        options=[{'label': s, 'value': s} for s in available_services],
                        multi=True,
                        placeholder="All Services"
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("Filter Events:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='d5-event-filter',
                        className='event-filter',
                        options=[{'label': e, 'value': e} for e in available_events],
                        multi=True,
                        placeholder="All Events"
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'}),

                html.Div([
                    html.Label("Week Range:", style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(
                        id='d5-week-slider',
                        min=min_week, max=max_week,
                        value=[min_week, max_week],
                        marks={i: str(i) for i in range(min_week, max_week+1, 5)}
                    ),
                ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'}),
            ]),
            
            # Row 3: Reset
            html.Div([
                html.Button("Reset All Filters", id="d5-reset-filters-btn", n_clicks=0, style={'cursor':'pointer', 'padding': '5px 15px'}),
            ], style={'textAlign': 'center', 'marginTop': '15px'}),
            
        ], style={'width': '90%', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px', 'marginBottom': '20px'}),

        # --- Main Dashboard Area ---
        html.Div([
            # LEFT COLUMN: Bubble Chart
            html.Div([
                html.Div([
                    html.H3("State Space (Select Region to Filter)", style={'display': 'inline-block', 'fontSize': '16px', 'marginRight': '10px'}),
                    html.Button("Clear Selection", id="d5-clear-selection-btn", n_clicks=0, style={'cursor':'pointer', 'fontSize': '12px'}),
                ], style={'textAlign': 'center'}),
                dcc.Graph(id='d5-bubble-chart')
            ], style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # RIGHT COLUMN: Heatmap and Timeline
            html.Div([
                html.H3("Cluster DNA", style={'textAlign': 'center', 'fontSize': '16px'}),
                dcc.Graph(id='d5-heatmap-dna', style={'height': '300px'}),
                
                html.H3("Timeline (Filtered)", style={'textAlign': 'center', 'fontSize': '16px', 'marginTop': '20px'}),
                dcc.Graph(id='d5-heatmap-timeline', style={'height': '350px'})
            ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
        ], style={'width': '95%', 'margin': '0 auto'}),

        # BOTTOM ROW: Drill-down
        html.Div([
            html.Hr(),
            html.H3("Capacity Drill-down (Filtered)", style={'textAlign': 'center', 'fontSize': '16px'}),
            html.P("Detailed view of services operating >95% capacity.",
                   style={'textAlign': 'center', 'color': '#777', 'fontSize': '12px'}),
            dcc.Graph(id='d5-capacity-drilldown', style={'height': '600px'})
        ], style={'width': '95%', 'margin': '0 auto', 'marginTop': '30px', 'paddingBottom': '50px'})
    ])

# ------------------------
# Callbacks
# ------------------------
def register_callbacks(app, df):
    
    # Preprocessing for callbacks
    df = df.copy()
    if 'occupancy_rate' not in df.columns:
        df['occupancy_rate'] = df.apply(
            lambda x: (x['patients_admitted'] / x['available_beds']) * 100 if x['available_beds'] > 0 else 0, 
            axis=1
        )
    
    min_week = df['week'].min()
    max_week = df['week'].max()

    @app.callback(
        [Output('d5-bubble-chart', 'figure'),
         Output('d5-heatmap-dna', 'figure'),
         Output('d5-heatmap-timeline', 'figure'),
         Output('d5-capacity-drilldown', 'figure'),
         Output('d5-cluster-filter', 'options'),
         Output('d5-service-filter', 'value'),
         Output('d5-event-filter', 'value'),
         Output('d5-week-slider', 'value'),
         Output('d5-cluster-filter', 'value'),
         Output('d5-bubble-chart', 'selectedData')],
        [Input('d5-k-slider', 'value'),
         Input('d5-service-filter', 'value'),
         Input('d5-event-filter', 'value'),
         Input('d5-week-slider', 'value'),
         Input('d5-cluster-filter', 'value'),
         Input('d5-bubble-chart', 'selectedData'),
         Input('d5-reset-filters-btn', 'n_clicks'),
         Input('d5-clear-selection-btn', 'n_clicks'),
         Input('d5-bubble-chart', 'restyleData')],
        [State('d5-bubble-chart', 'figure')]
    )
    def update_dashboard(k, selected_services, selected_events, week_range, focus_cluster, bubble_selected, 
                         reset_clicks, clear_clicks, restyle_data, current_figure):
        
        triggered_id = ctx.triggered_id
        
        # Default return values for controls (no change)
        ret_services = no_update
        ret_events = no_update
        ret_weeks = no_update
        ret_cluster_focus = no_update
        ret_selected_data = no_update
        
        # Logic to override inputs if Reset/Clear triggered
        if triggered_id == 'd5-reset-filters-btn':
            selected_services = []
            selected_events = []
            week_range = [min_week, max_week]
            focus_cluster = None
            bubble_selected = None
            
            # Set return values to update UI
            ret_services = []
            ret_events = []
            ret_weeks = [min_week, max_week]
            ret_cluster_focus = None
            ret_selected_data = None
            
        elif triggered_id == 'd5-clear-selection-btn':
            bubble_selected = None
            ret_selected_data = None

        # ----------------------------------------------------
        # 1. Global Clustering (On Full Data)
        # ----------------------------------------------------
        df_viz = df.copy()
        
        cluster_features = ['occupancy_rate', 'patient_satisfaction', 'staff_morale']
        feature_labels = ['Occupancy %', 'Pt Satisfaction', 'Staff Morale']

        scaler = StandardScaler()
        data_matrix = scaler.fit_transform(df_viz[cluster_features])
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        raw_clusters = kmeans.fit_predict(data_matrix)
        
        # Sort Clusters by Occupancy
        df_temp = pd.DataFrame({'raw_cluster': raw_clusters, 'occupancy': df_viz['occupancy_rate']})
        cluster_stats = df_temp.groupby('raw_cluster')['occupancy'].mean().sort_values().reset_index()
        mapping = {row['raw_cluster']: i for i, row in cluster_stats.iterrows()}
        
        df_viz['cluster'] = df_temp['raw_cluster'].map(mapping)
        df_viz['cluster_label'] = df_viz['cluster'].astype(str)

        # Define Colors
        if k == 2:
            colors = ['#2ca02c', '#d62728']
        elif k == 3:
            colors = ['#2ca02c', "#0e66ff", '#d62728']
        elif k == 4:
            colors = ['#2ca02c', "#0e66ff", '#d62728', "#43008b"]
        else:
            colors = ['#2ca02c', "#0e66ff", '#d62728', "#43008b", "#F32ADB"]
        
        color_map = {str(i): c for i, c in enumerate(colors) if i < len(colors)}
        
        # Update Cluster Dropdown Options
        cluster_options = [{'label': f"Cluster {i}", 'value': str(i)} for i in range(k)]

        # ----------------------------------------------------
        # 2. Apply User Filters (Service, Event, Week)
        # ----------------------------------------------------
        mask = pd.Series([True] * len(df_viz))
        
        if selected_services:
            mask &= df_viz['service'].isin(selected_services)
        
        if selected_events:
            mask &= df_viz['event'].astype(str).isin(selected_events)
            
        if week_range:
            mask &= (df_viz['week'] >= week_range[0]) & (df_viz['week'] <= week_range[1])
            
        df_view = df_viz[mask].copy()

        # ----------------------------------------------------
        # 3. Apply Interactive Filters (Selection, Focus, Legend)
        # ----------------------------------------------------
        
        # Determine Visible Clusters from Legend (restyleData)
        visible_clusters = set([str(i) for i in range(k)])
        
        if current_figure and triggered_id != 'd5-k-slider' and triggered_id != 'd5-reset-filters-btn':
            if len(current_figure['data']) == k:
                for i, trace in enumerate(current_figure['data']):
                    vis = trace.get('visible', True)
                    if vis == 'legendonly':
                        if str(i) in visible_clusters:
                            visible_clusters.remove(str(i))
                
                if triggered_id == 'd5-bubble-chart' and restyle_data:
                    update_dict = restyle_data[0]
                    trace_indices = restyle_data[1]
                    
                    if 'visible' in update_dict:
                        new_vis = update_dict['visible']
                        val = new_vis[0] if isinstance(new_vis, list) else new_vis
                        
                        for idx in trace_indices:
                            cluster_id = str(idx)
                            if val == 'legendonly':
                                if cluster_id in visible_clusters:
                                    visible_clusters.remove(cluster_id)
                            else:
                                visible_clusters.add(cluster_id)

        # Create Highlight Mask
        highlight_mask = pd.Series([True] * len(df_view), index=df_view.index)
        
        if focus_cluster is not None:
            highlight_mask &= (df_view['cluster_label'] == focus_cluster)
            
        if bubble_selected and bubble_selected.get('points'):
            selected_indices = [p['customdata'][0] for p in bubble_selected['points'] if 'customdata' in p]
            if selected_indices:
                highlight_mask &= df_view.index.isin(selected_indices)
                
        highlight_mask &= df_view['cluster_label'].isin(visible_clusters)

        df_highlight = df_view[highlight_mask].copy()

        # ----------------------------------------------------
        # 4. Generate Figures
        # ----------------------------------------------------
        
        # --- FIGURE 1: BUBBLE CHART ---
        if triggered_id == 'd5-bubble-chart' and restyle_data:
            fig_bubble = no_update
        else:
            df_view['opacity'] = highlight_mask.map({True: 1.0, False: 0.1})
            df_view['index_col'] = df_view.index
            
            fig_bubble = px.scatter(
                df_view,
                x='occupancy_rate',
                y='patient_satisfaction',
                size='patients_admitted',
                color='cluster_label',
                symbol='cluster_label',
                color_discrete_map=color_map,
                category_orders={"cluster_label": [str(i) for i in range(k)]},
                title="<b>State Space:</b> Stress vs Quality",
                hover_data=['service', 'week', 'staff_morale'],
                custom_data=['index_col']
            )
            
            for trace in fig_bubble.data:
                cluster_val = trace.name
                cluster_df = df_view[df_view['cluster_label'] == cluster_val]
                if not cluster_df.empty:
                    trace.marker.opacity = cluster_df['opacity'].values
                    
                if cluster_val not in visible_clusters:
                    trace.visible = 'legendonly'
            
            fig_bubble.update_layout(
                height=700,
                legend_title="Stress Level",
                margin=dict(l=40, r=40, t=40, b=40),
                clickmode='event+select'
            )
        
        # --- FIGURE 2: CLUSTER DNA HEATMAP ---
        df_scaled = pd.DataFrame(data_matrix, columns=cluster_features)
        df_scaled['cluster'] = df_viz['cluster']
        centroids = df_scaled.groupby('cluster')[cluster_features].mean()
        
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
            title="<b>Cluster DNA</b> (Global Definition)",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(autorange="reversed")
        )

        # --- FIGURE 3: TIMELINE HEATMAP ---
        if df_highlight.empty:
            fig_timeline = go.Figure()
            fig_timeline.update_layout(title="No data selected")
        else:
            fig_timeline = px.scatter(
                df_highlight, x='week', y='service', color='cluster_label', symbol='cluster_label',
                color_discrete_map=color_map,
                category_orders={"cluster_label": [str(i) for i in range(k)]},
                title="<b>Timeline:</b> Crisis Patterns (Filtered)",
                hover_data=['occupancy_rate']
            )
            fig_timeline.update_traces(marker=dict(size=12, symbol='square'))
            fig_timeline.update_layout(
                height=350,
                xaxis_title="Week", 
                yaxis_title="Service",
                showlegend=False,
                xaxis=dict(range=[min_week, max_week])
            )

        # --- FIGURE 4: CAPACITY DRILL-DOWN ---
        df_drill = df_highlight[df_highlight['occupancy_rate'] >= 95].copy()
        
        if df_drill.empty:
            fig_drill = go.Figure()
            fig_drill.update_layout(
                title="No high-occupancy data in selection",
                height=600
            )
        else:
            fig_drill = px.scatter(
                df_drill,
                x='patient_satisfaction',
                y=pd.Series([1]*len(df_drill)), 
                color='cluster_label',
                facet_row='service', 
                color_discrete_map=color_map,
                category_orders={"cluster_label": [str(i) for i in range(k)]},
                title="<b>Drill-down:</b> Anatomy of the '100% Occupancy' Wall (Filtered)",
                hover_data=['week', 'patients_admitted', 'staff_morale']
            )
            
            fig_drill.update_layout(
                height=600, 
                xaxis_title="Patient Satisfaction",
                showlegend=False,
                yaxis={'visible': False, 'showticklabels': False},
                margin=dict(l=120)
            )
            fig_drill.update_yaxes(matches=None, showticklabels=False, visible=False)
            fig_drill.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            
            fig_drill.for_each_annotation(lambda a: a.update(
                text=a.text.split("=")[-1],
                x=-0.01,
                xanchor='right',
                textangle=-90
            ))

        return fig_bubble, fig_dna, fig_timeline, fig_drill, cluster_options, \
               ret_services, ret_events, ret_weeks, ret_cluster_focus, ret_selected_data