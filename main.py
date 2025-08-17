import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

def gyroid_surface(x, y, z, t=0.5):
    """Gyroid TPMS equation"""
    return np.sin(x)*np.cos(y) + np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x) - t

def schwarz_p_surface(x, y, z, t=0.5):
    """Schwarz P TPMS equation"""
    return np.cos(x) + np.cos(y) + np.cos(z) - t

def schwarz_d_surface(x, y, z, t=0.5):
    """Schwarz D (Diamond) TPMS equation"""
    return (np.sin(x)*np.sin(y)*np.sin(z) + 
            np.sin(x)*np.cos(y)*np.cos(z) + 
            np.cos(x)*np.sin(y)*np.cos(z) + 
            np.cos(x)*np.cos(y)*np.sin(z)) - t

def neovius_surface(x, y, z, t=0.5):
    """Neovius TPMS equation"""
    return (3*(np.cos(x) + np.cos(y) + np.cos(z)) + 
            4*np.cos(x)*np.cos(y)*np.cos(z)) - t

def fischer_koch_surface(x, y, z, t=0.5):
    """Fischer-Koch TPMS equation"""
    return (np.cos(2*x)*np.sin(y)*np.cos(z) + 
            np.cos(x)*np.cos(2*y)*np.sin(z) + 
            np.sin(x)*np.cos(y)*np.cos(2*z)) - t

def lidinoid_surface(x, y, z, t=0.5):
    """Lidinoid TPMS equation"""
    return (np.sin(2*x)*np.cos(y)*np.sin(z) + 
            np.sin(x)*np.sin(2*y)*np.cos(z) + 
            np.cos(x)*np.sin(y)*np.sin(2*z) - 
            (np.cos(2*x)*np.cos(2*y) + 
             np.cos(2*y)*np.cos(2*z) + 
             np.cos(2*z)*np.cos(2*x))/2) - t

def create_tpms_skeletal_structure(tpms_type, grid_size=32, domain_size=2*np.pi, 
                                 thickness_param=0.0, connection_threshold=0.5):
    """
    Create skeletal structure from TPMS by finding critical points and connections
    """
    # TPMS function mapping
    tpms_functions = {
        'Gyroid': gyroid_surface,
        'Schwarz P': schwarz_p_surface,
        'Schwarz D': schwarz_d_surface,
        'Neovius': neovius_surface,
        'Fischer-Koch': fischer_koch_surface,
        'Lidinoid': lidinoid_surface
    }
    
    tpms_func = tpms_functions[tpms_type]
    
    # Create 3D grid
    x = np.linspace(0, domain_size, grid_size)
    y = np.linspace(0, domain_size, grid_size)
    z = np.linspace(0, domain_size, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Evaluate TPMS function
    surface_values = tpms_func(X, Y, Z, thickness_param)
    
    # Find points near the surface (skeletal approximation)
    surface_mask = np.abs(surface_values) < connection_threshold
    
    # Extract coordinates of surface points
    surface_indices = np.where(surface_mask)
    vertices = np.column_stack([X[surface_indices], 
                               Y[surface_indices], 
                               Z[surface_indices]])
    
    # Create connections between nearby vertices to form skeleton
    edges = create_skeleton_connections(vertices, max_connection_distance=0.5)
    
    return vertices, edges

def create_skeleton_connections(vertices, max_connection_distance=0.5):
    """Create edges between nearby vertices to form a skeletal structure"""
    edges = []
    n_vertices = len(vertices)
    
    # For performance, limit to smaller subsets for large vertex counts
    if n_vertices > 1000:
        # Sample vertices for connections to avoid memory issues
        sample_indices = np.random.choice(n_vertices, min(1000, n_vertices), replace=False)
        sample_vertices = vertices[sample_indices]
        
        # Calculate distances between sampled vertices
        distances = cdist(sample_vertices, sample_vertices)
        
        # Create connections for nearby points
        for i in range(len(sample_vertices)):
            for j in range(i+1, len(sample_vertices)):
                if distances[i, j] < max_connection_distance:
                    edges.append([sample_indices[i], sample_indices[j]])
    else:
        # Full connection calculation for smaller vertex sets
        distances = cdist(vertices, vertices)
        
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                if distances[i, j] < max_connection_distance:
                    edges.append([i, j])
    
    return edges

def create_tpms_beam_approximation(tpms_type, unit_cells=2, beams_per_cell=20):
    """
    Alternative approach: Create beam-based approximation of TPMS
    by sampling the surface and creating beam connections
    """
    tpms_functions = {
        'Gyroid': gyroid_surface,
        'Schwarz P': schwarz_p_surface,
        'Schwarz D': schwarz_d_surface,
        'Neovius': neovius_surface,
        'Fischer-Koch': fischer_koch_surface,
        'Lidinoid': lidinoid_surface
    }
    
    tpms_func = tpms_functions[tpms_type]
    
    vertices = []
    edges = []
    vertex_count = 0
    
    # Create unit cells
    cell_size = 2 * np.pi
    
    for x in range(unit_cells):
        for y in range(unit_cells):
            for z in range(unit_cells):
                # Sample points within each unit cell
                cell_vertices = sample_tpms_surface(
                    tpms_func, 
                    offset=[x * cell_size, y * cell_size, z * cell_size],
                    cell_size=cell_size,
                    n_samples=beams_per_cell
                )
                
                # Translate vertices
                translated_vertices = cell_vertices + np.array([x, y, z]) * cell_size
                vertices.extend(translated_vertices)
                
                # Create connections within the cell
                cell_edges = create_cell_connections(len(cell_vertices), vertex_count)
                edges.extend(cell_edges)
                
                vertex_count += len(cell_vertices)
    
    return np.array(vertices), edges

def sample_tpms_surface(tpms_func, offset=[0, 0, 0], cell_size=2*np.pi, n_samples=20):
    """Sample points on TPMS surface within a unit cell"""
    vertices = []
    
    # Use marching cubes approach simplified - sample grid and find surface points
    grid_res = int(n_samples**(1/3)) + 2
    x = np.linspace(offset[0], offset[0] + cell_size, grid_res)
    y = np.linspace(offset[1], offset[1] + cell_size, grid_res)
    z = np.linspace(offset[2], offset[2] + cell_size, grid_res)
    
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            for k in range(len(z)-1):
                # Check if surface crosses this cube
                cube_corners = [
                    [x[i], y[j], z[k]], [x[i+1], y[j], z[k]],
                    [x[i], y[j+1], z[k]], [x[i+1], y[j+1], z[k]],
                    [x[i], y[j], z[k+1]], [x[i+1], y[j], z[k+1]],
                    [x[i], y[j+1], z[k+1]], [x[i+1], y[j+1], z[k+1]]
                ]
                
                values = [tpms_func(c[0], c[1], c[2]) for c in cube_corners]
                
                # If surface crosses this cube (sign change), add center point
                if min(values) <= 0 <= max(values):
                    center = [(x[i] + x[i+1])/2, (y[j] + y[j+1])/2, (z[k] + z[k+1])/2]
                    vertices.append(center)
                    
                    if len(vertices) >= n_samples:
                        break
            if len(vertices) >= n_samples:
                break
        if len(vertices) >= n_samples:
            break
    
    return np.array(vertices[:n_samples]) if vertices else np.array([[0, 0, 0]])

def create_cell_connections(n_vertices, vertex_offset):
    """Create connections between vertices in a cell"""
    edges = []
    
    # Connect each vertex to a few nearest neighbors
    for i in range(n_vertices):
        for j in range(i+1, min(i+4, n_vertices)):  # Connect to next 3 vertices
            edges.append([i + vertex_offset, j + vertex_offset])
    
    return edges

def plot_tpms_lattice(vertices, edges, strut_thickness, colorscale='Viridis'):
    """Create interactive 3D plot for TPMS lattice"""
    if len(vertices) == 0 or len(edges) == 0:
        # Create empty plot
        fig = go.Figure()
        fig.add_annotation(
            text="No structure generated. Try adjusting parameters.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    x_lines = []
    y_lines = []
    z_lines = []
    
    for edge in edges:
        if edge[0] < len(vertices) and edge[1] < len(vertices):
            start, end = edge
            x_lines.extend([vertices[start, 0], vertices[end, 0], None])
            y_lines.extend([vertices[start, 1], vertices[end, 1], None])
            z_lines.extend([vertices[start, 2], vertices[end, 2], None])
    
    # Create colors based on position
    colors = np.zeros(len(x_lines))
    idx = 0
    for edge in edges:
        if edge[0] < len(vertices) and edge[1] < len(vertices):
            start, end = edge
            pos = vertices[start] + vertices[end]
            colors[idx:idx+3] = np.sum(pos) % 10
            idx += 3
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(
            color=colors,
            width=strut_thickness,
            colorscale=colorscale
        )
    )])
    
    # Add vertex points
    if len(vertices) < 500:  # Only show vertices for smaller structures
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='Nodes'
        ))
    
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

# Streamlit app
st.set_page_config(layout="wide", page_title="TPMS Lattice Structure Visualizer")
st.title("TPMS-Based Lattice Structure Generator")

# Create two columns
col1, col2 = st.columns([1, 3])

# Control panel
with col1:
    st.header("TPMS Settings")
    
    with st.expander("TPMS Type", expanded=True):
        tpms_type = st.selectbox(
            "Select TPMS Surface",
            ["Gyroid", "Schwarz P", "Schwarz D", "Neovius", "Fischer-Koch", "Lidinoid"]
        )
        
        st.write("""
        **TPMS Types:**
        - **Gyroid**: Smooth, interconnected channels
        - **Schwarz P**: Cubic symmetry, simple
        - **Schwarz D**: Diamond-like structure
        - **Neovius**: Complex cubic symmetry
        - **Fischer-Koch**: Twisted channels
        - **Lidinoid**: Complex interconnected structure
        """)
    
    with st.expander("Structure Parameters", expanded=True):
        method = st.radio(
            "Generation Method",
            ["Beam Approximation", "Skeletal Structure"]
        )
        
        unit_cells = st.selectbox(
            "Unit Cells",
            [1, 2, 3],
            format_func=lambda x: f"{x}x{x}x{x}"
        )
        
        if method == "Beam Approximation":
            beams_per_cell = st.slider(
                "Beams per Cell",
                min_value=8,
                max_value=50,
                value=20
            )
        else:
            thickness_param = st.slider(
                "Thickness Parameter",
                min_value=-1.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
            
            connection_threshold = st.slider(
                "Connection Threshold",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
    
    with st.expander("Visualization", expanded=True):
        strut_thickness = st.slider(
            "Strut Thickness",
            min_value=1,
            max_value=10,
            value=3
        )
        
        colorscale = st.selectbox(
            "Color Scheme",
            ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Rainbow']
        )

# Generate structure based on method
with col2:
    try:
        if method == "Beam Approximation":
            vertices, edges = create_tpms_beam_approximation(
                tpms_type, unit_cells, beams_per_cell
            )
        else:
            vertices, edges = create_tpms_skeletal_structure(
                tpms_type, 
                grid_size=20 + unit_cells*10,
                domain_size=unit_cells * 2 * np.pi,
                thickness_param=thickness_param,
                connection_threshold=connection_threshold
            )
        
        # Display structure info
        st.write(f"**Structure Info:** {len(vertices)} nodes, {len(edges)} beams")
        
        # Plot the structure
        fig = plot_tpms_lattice(vertices, edges, strut_thickness, colorscale)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating structure: {str(e)}")
        st.write("Try adjusting the parameters or selecting a different method.")

    with st.expander("TPMS Theory & Applications", expanded=False):
        st.markdown("""
        ### Triply Periodic Minimal Surfaces (TPMS)
        
        TPMS are mathematical surfaces that:
        - Repeat infinitely in all three spatial directions
        - Have zero mean curvature (minimal surfaces)
        - Divide space into two interwoven regions
        
        ### Applications:
        - **Lightweight structures**: High strength-to-weight ratio
        - **Heat exchangers**: Large surface area for heat transfer
        - **Fluid mixing**: Complex flow patterns
        - **Biomimetic structures**: Similar to natural foams and bones
        - **Metamaterials**: Unique mechanical properties
        
        ### Implementation Notes:
        - **Beam Approximation**: Faster, creates beam-like structures
        - **Skeletal Structure**: More accurate to TPMS geometry
        - Adjust thickness/threshold parameters to control density
        - Higher unit cell counts create more complex patterns
        """)

# Footer
st.markdown("---")
st.markdown("TPMS-based lattice structures for advanced engineering applications")
