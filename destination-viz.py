import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import random
import numpy as np

# Set page config
st.set_page_config(page_title="YAML Destinations Visualizer", layout="wide")

def generate_dummy_data():
    """Generate dummy YAML data with folders, files, and destinations"""
    
    # Fake destination keys
    destination_keys = [
        'sqlservermapping', 'database_config', 'api_endpoints', 'redis_cache',
        'message_queue', 'file_storage', 'log_aggregator', 'metrics_collector',
        'auth_service', 'user_management', 'payment_gateway', 'email_service',
        'notification_hub', 'search_engine', 'cdn_config', 'backup_service',
        'monitoring_alerts', 'load_balancer', 'security_scanner', 'data_warehouse',
        'etl_pipeline', 'reporting_service', 'audit_logger', 'config_manager',
        'secrets_vault', 'container_registry', 'deployment_pipeline', 'test_runner',
        'code_quality', 'dependency_scanner', 'artifact_repository', 'ci_cd_hooks',
        'kubernetes_cluster', 'docker_registry', 'helm_charts', 'ingress_controller',
        'service_mesh', 'observability_stack', 'tracing_service', 'feature_flags',
        'rate_limiter', 'circuit_breaker', 'health_checks', 'graceful_shutdown',
        'session_store', 'workspace_config', 'tenant_isolation', 'multi_region',
        'disaster_recovery', 'compliance_checker', 'data_classification'
    ]
    
    # Fake folders (sources)
    folders = [
        'user-service', 'payment-service', 'notification-service', 'auth-service',
        'inventory-service', 'order-service', 'analytics-service', 'admin-portal',
        'mobile-api', 'web-frontend', 'batch-jobs', 'data-pipeline',
        'monitoring-stack', 'security-service', 'reporting-service'
    ]
    
    # Possible destination values
    environments = ['dev', 'uat', 'prod']
    
    data = []
    
    for folder in folders:
        # Each folder has 2-5 files
        num_files = random.randint(2, 5)
        
        for i in range(num_files):
            filename = f"{folder}/config_{i+1}.yaml"
            
            # Each file has 1-4 destinations
            num_destinations = random.randint(1, 4)
            selected_destinations = random.sample(destination_keys, num_destinations)
            
            for dest in selected_destinations:
                # Each destination has 1-3 environment values
                num_envs = random.randint(1, 3)
                selected_envs = random.sample(environments, num_envs)
                
                for env in selected_envs:
                    data.append({
                        'folder': folder,
                        'file': filename,
                        'destination': dest,
                        'environment': env
                    })
    
    return pd.DataFrame(data)

def create_network_graph(df, selected_folder=None):
    """Create NetworkX graph and convert to Plotly"""
    
    # Filter data if folder is selected
    if selected_folder and selected_folder != "All Folders":
        df_filtered = df[df['folder'] == selected_folder]
    else:
        df_filtered = df
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes and edges
    for _, row in df_filtered.iterrows():
        file_node = row['file']
        dest_node = f"{row['destination']} ({row['environment']})"
        
        # Add edge between file and destination
        G.add_edge(file_node, dest_node)
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Separate nodes by type
    file_nodes = [node for node in G.nodes() if node.endswith('.yaml')]
    dest_nodes = [node for node in G.nodes() if not node.endswith('.yaml')]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    file_x = [pos[node][0] for node in file_nodes]
    file_y = [pos[node][1] for node in file_nodes]
    file_text = [node.split('/')[-1] for node in file_nodes]  # Just filename
    
    dest_x = [pos[node][0] for node in dest_nodes]
    dest_y = [pos[node][1] for node in dest_nodes]
    dest_text = dest_nodes
    
    file_trace = go.Scatter(
        x=file_x, y=file_y,
        mode='markers+text',
        hoverinfo='text',
        text=file_text,
        textposition="middle center",
        hovertext=[f"File: {node}<br>Connections: {G.degree(node)}" for node in file_nodes],
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        name='Files'
    )
    
    dest_trace = go.Scatter(
        x=dest_x, y=dest_y,
        mode='markers+text',
        hoverinfo='text',
        text=[text.split(' (')[0][:15] + '...' if len(text.split(' (')[0]) > 15 else text.split(' (')[0] for text in dest_text],
        textposition="middle center",
        hovertext=[f"Destination: {node}<br>Used by: {G.degree(node)} files" for node in dest_nodes],
        marker=dict(
            size=25,
            color='lightcoral',
            line=dict(width=2, color='darkred')
        ),
        name='Destinations'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, file_trace, dest_trace],
                    layout=go.Layout(
                        title=dict(
                            text=f"File-Destination Network {'(' + selected_folder + ')' if selected_folder and selected_folder != 'All Folders' else ''}",
                            font=dict(size=16)
                        ),
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Blue = YAML Files, Red = Destinations",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(size=12, color="grey")
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    return fig

def create_folder_summary(df):
    """Create folder summary charts"""
    
    # Destinations per folder
    folder_dest_count = df.groupby('folder')['destination'].nunique().reset_index()
    folder_dest_count.columns = ['Folder', 'Unique Destinations']
    
    fig1 = px.bar(folder_dest_count, x='Folder', y='Unique Destinations',
                  title="Number of Unique Destinations per Folder",
                  color='Unique Destinations',
                  color_continuous_scale='viridis')
    fig1.update_layout(xaxis_tickangle=45)
    
    # Most popular destinations
    dest_popularity = df['destination'].value_counts().head(10).reset_index()
    dest_popularity.columns = ['Destination', 'Usage Count']
    
    fig2 = px.bar(dest_popularity, x='Usage Count', y='Destination',
                  title="Top 10 Most Used Destinations",
                  orientation='h',
                  color='Usage Count',
                  color_continuous_scale='plasma')
    
    return fig1, fig2

def main():
    st.title("🗂️ YAML Destinations Visualizer")
    st.markdown("Visualize relationships between YAML files and their destinations")
    
    # Generate or load data
    if 'data' not in st.session_state:
        with st.spinner("Generating dummy data..."):
            st.session_state.data = generate_dummy_data()
    
    df = st.session_state.data
    
    # Sidebar
    st.sidebar.header("Filters")
    
    # Folder selector
    folders = ['All Folders'] + sorted(df['folder'].unique().tolist())
    selected_folder = st.sidebar.selectbox("Select Folder", folders)
    
    # Environment filter
    environments = ['All Environments'] + sorted(df['environment'].unique().tolist())
    selected_env = st.sidebar.selectbox("Select Environment", environments)
    
    # Apply environment filter
    if selected_env != 'All Environments':
        df = df[df['environment'] == selected_env]
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Network Graph", "📈 Folder Analysis", "📋 Data Table"])
    
    with tab1:
        st.header("Network Visualization")
        
        # Show network graph
        fig = create_network_graph(df, selected_folder)
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        
        filtered_df = df[df['folder'] == selected_folder] if selected_folder != 'All Folders' else df
        
        with col1:
            st.metric("Total Files", filtered_df['file'].nunique())
        with col2:
            st.metric("Total Destinations", filtered_df['destination'].nunique())
        with col3:
            st.metric("Total Connections", len(filtered_df))
        with col4:
            st.metric("Folders", filtered_df['folder'].nunique())
    
    with tab2:
        st.header("Folder Analysis")
        
        fig1, fig2 = create_folder_summary(df)
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Environment breakdown
        env_breakdown = df.groupby(['folder', 'environment']).size().reset_index(name='count')
        fig3 = px.bar(env_breakdown, x='folder', y='count', color='environment',
                      title="Environment Distribution by Folder",
                      barmode='stack')
        fig3.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.header("Raw Data")
        
        # Show filtered data
        display_df = df[df['folder'] == selected_folder] if selected_folder != 'All Folders' else df
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="yaml_destinations.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()