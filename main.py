import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from skimage import measure
import plotly.figure_factory as ff

# TPMS Mathematical Functions
def gyroid_surface(x, y, z, t=0.0):
    """Gyroid TPMS equation"""
    return np.sin(x)*np.cos(y) + np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x) - t

def schwarz_p_surface(x, y, z, t=0.0):
    """Schwarz P TPMS equation"""
    return np.cos(x) + np.cos(y) + np.cos(z) - t

def schwarz_d_surface(x, y, z, t=0.0):
    """Schwarz D (Diamond) TPMS equation"""
    return (np.sin(x)*np.sin(y)*np.sin(z) + 
            np.sin(x)*np.cos(y)*np.cos(z) + 
            np.cos(x)*np.sin(y)*np.cos(z) + 
            np.cos(x)*np.cos(y)*np.sin(z)) - t

def neovius_surface(x, y, z, t=0.0):
    """Neovius TPMS equation"""
    return (3*(np.cos(x) + np.cos(y) + np.cos(z)) + 
            4*np.cos(x)*np.cos(y)*np.cos(z)) - t

def primitive_surface(x, y, z, t=0.0):
    """Primitive (Schwarz P) surface"""
    return np.cos(x) + np.cos(y) + np.cos(z) - t

def fischer_koch_surface(x, y, z, t=0.0):
    """Fischer-Koch surface"""
    return (np.cos(2*x)*np.sin(y)*np.cos(z) + 
            np.cos(x)*np.cos(2*y)*np.sin(z) + 
            np.sin(x)*np.cos(y)*np.cos(2*z)) - t

# =============================================================================
# FEATURE 1: SHEET-BASED TPMS (Actual Surfaces using Marching Cubes)
# =============================================================================

def create_sheet_based_tpms(tpms_type, resolution=64, domain_size=2*np.pi, thickness=0.0):
    """Create actual triangulated surfaces using marching cubes algorithm"""
    
    tpms_functions = {
        'Gyroid': gyroid_surface,
        'Schwarz P': schwarz_p_surface,
        'Schwarz D': schwarz_d_surface,
        'Neovius': neovius_surface,
        'Primitive': primitive_surface,
        'Fischer-Koch': fischer_koch_surface
    }
    
    tpms_func = tpms_functions[tpms_type]
    
    # Create 3D grid
    x = np.linspace(0, domain_size, resolution)
    y = np.linspace(0, domain_size, resolution)
    z = np.linspace(0, domain_size, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Evaluate TPMS function on the grid
    values = tpms_func(X, Y, Z, thickness)
    
    # Extract surface using marching cubes
    try:
        vertices, faces, normals, _ = measure.marching_cubes(
            values, 
            level=0.0,  # Extract zero-level surface
            spacing=(domain_size/resolution, domain_size/resolution, domain_size/resolution)
        )
        
        # Compute face centers for coloring
        face_centers = vertices[faces].mean(axis=1)
        colors = np.sqrt(face_centers[:, 0]**2 + face_centers[:, 1]**2 + face_centers[:, 2]**2)
        
        return vertices, faces, normals, colors
        
    except Exception as e:
        st.error(f"Error in marching cubes: {e}")
        return None, None, None, None

def plot_sheet_surface(vertices, faces, colors, surface_name="TPMS Surface"):
    """Plot triangulated surface mesh"""
    if vertices is None or faces is None:
        return None
    
    # Create mesh3d plot
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1], 
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=colors,
            colorscale='Viridis',
            opacity=0.7,
            name=surface_name,
            showscale=True
        )
    ])
    
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title=f'{surface_name} - Sheet-Based Visualization',
        margin=dict(l=0, r=0, t=40, b=0),
        height=700
    )
    
    return fig

# =============================================================================
# FEATURE 2: HYBRID STRUCTURES (Combine Multiple TPMS Types)
# =============================================================================

def create_hybrid_tpms(tpms1_type, tpms2_type, blend_function, resolution=64, 
                      domain_size=2*np.pi, thickness1=0.0, thickness2=0.0):
    """Create hybrid structure by combining two TPMS types"""
    
    tpms_functions = {
        'Gyroid': gyroid_surface,
        'Schwarz P': schwarz_p_surface,
        'Schwarz D': schwarz_d_surface,
        'Neovius': neovius_surface,
        'Primitive': primitive_surface,
        'Fischer-Koch': fischer_koch_surface
    }
    
    tpms1_func = tpms_functions[tpms1_type]
    tpms2_func = tpms_functions[tpms2_type]
    
    # Create 3D grid
    x = np.linspace(0, domain_size, resolution)
    y = np.linspace(0, domain_size, resolution)
    z = np.linspace(0, domain_size, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Evaluate both TPMS functions
    values1 = tpms1_func(X, Y, Z, thickness1)
    values2 = tpms2_func(X, Y, Z, thickness2)
    
    # Combine based on blend function
    if blend_function == "Union (OR)":
        # Union: min of both surfaces
        combined_values = np.minimum(np.abs(values1), np.abs(values2)) * np.sign(values1)
    elif blend_function == "Intersection (AND)":
        # Intersection: max of both surfaces  
        combined_values = np.maximum(np.abs(values1), np.abs(values2)) * np.sign(values1)
    elif blend_function == "Weighted Average":
        # Weighted combination
        weight = 0.5
        combined_values = weight * values1 + (1 - weight) * values2
    elif blend_function == "Spatial Transition":
        # Smooth spatial transition between the two
        transition = np.sin(X) * np.cos(Y)  # Spatial modulation
        weight = (transition + 1) / 2  # Normalize to [0,1]
        combined_values = weight * values1 + (1 - weight) * values2
    elif blend_function == "XOR (Exclusive)":
        # XOR operation - where only one surface exists
        mask1 = np.abs(values1) < 0.1
        mask2 = np.abs(values2) < 0.1
        xor_mask = mask1 ^ mask2  # XOR operation
        combined_values = np.where(xor_mask, 
                                 np.minimum(np.abs(values1), np.abs(values2)), 
                                 np.maximum(np.abs(values1), np.abs(values2)))
    
    # Extract surface using marching cubes
    try:
        vertices, faces, normals, _ = measure.marching_cubes(
            combined_values,
            level=0.0,
            spacing=(domain_size/resolution, domain_size/resolution, domain_size/resolution)
        )
        
        # Color by blend ratio for visualization
        face_centers = vertices[faces].mean(axis=1)
        val1_at_faces = np.array([
            tpms1_func(fc[0], fc[1], fc[2], thickness1) for fc in face_centers
        ])
        val2_at_faces = np.array([
            tpms2_func(fc[0], fc[1], fc[2], thickness2) for fc in face_centers  
        ])
        
        # Color represents which TPMS dominates
        colors = np.abs(val1_at_faces) / (np.abs(val1_at_faces) + np.abs(val2_at_faces) + 1e-10)
        
        return vertices, faces, colors, f"{tpms1_type} + {tpms2_type}"
        
    except Exception as e:
        st.error(f"Error creating hybrid structure: {e}")
        return None, None, None, None

# =============================================================================  
# FEATURE 3: GRADED STRUCTURES (Spatially Varying Thickness/Density)
# =============================================================================

def create_graded_tpms(tpms_type, grading_function, resolution=64, 
                      domain_size=2*np.pi, base_thickness=0.0, grading_intensity=1.0):
    """Create graded TPMS with spatially varying thickness/density"""
    
    tpms_functions = {
        'Gyroid': gyroid_surface,
        'Schwarz P': schwarz_p_surface,
        'Schwarz D': schwarz_d_surface,
        'Neovius': neovius_surface,
        'Primitive': primitive_surface,
        'Fischer-Koch': fischer_koch_surface
    }
    
    tpms_func = tpms_functions[tpms_type]
    
    # Create 3D grid
    x = np.linspace(0, domain_size, resolution)
    y = np.linspace(0, domain_size, resolution) 
    z = np.linspace(0, domain_size, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create grading function
    center_x, center_y, center_z = domain_size/2, domain_size/2, domain_size/2
    
    if grading_function == "Radial (Center to Edge)":
        # Distance from center
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
        max_distance = np.sqrt(3) * domain_size/2
        grading = distance / max_distance
    elif grading_function == "Linear X":
        # Linear gradient along X
        grading = X / domain_size
    elif grading_function == "Linear Y":
        # Linear gradient along Y
        grading = Y / domain_size
    elif grading_function == "Linear Z":
        # Linear gradient along Z
        grading = Z / domain_size
    elif grading_function == "Sinusoidal X":
        # Sinusoidal variation along X
        grading = (np.sin(2 * np.pi * X / domain_size) + 1) / 2
    elif grading_function == "Spherical Shells":
        # Concentric spherical shells
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
        grading = np.abs(np.sin(4 * np.pi * distance / domain_size))
    elif grading_function == "Cubic Shells":
        # Cubic shells from center
        distance = np.maximum(np.maximum(np.abs(X - center_x), np.abs(Y - center_y)), 
                            np.abs(Z - center_z))
        max_distance = domain_size/2
        grading = distance / max_distance
    
    # Apply grading to thickness
    graded_thickness = base_thickness + grading_intensity * grading
    
    # Evaluate TPMS with graded thickness
    values = tpms_func(X, Y, Z, graded_thickness)
    
    # Extract surface using marching cubes
    try:
        vertices, faces, normals, _ = measure.marching_cubes(
            values,
            level=0.0,
            spacing=(domain_size/resolution, domain_size/resolution, domain_size/resolution)
        )
        
        # Color by grading value for visualization
        face_centers = vertices[faces].mean(axis=1)
        
        # Interpolate grading values at face centers
        if grading_function == "Radial (Center to Edge)":
            face_distances = np.sqrt((face_centers[:, 0] - center_x)**2 + 
                                   (face_centers[:, 1] - center_y)**2 + 
                                   (face_centers[:, 2] - center_z)**2)
            max_distance = np.sqrt(3) * domain_size/2
            colors = face_distances / max_distance
        else:
            # For other grading functions, use a simpler coloring
            colors = face_centers[:, 0] / domain_size  # X-position based coloring
        
        return vertices, faces, colors, grading_function
        
    except Exception as e:
        st.error(f"Error creating graded structure: {e}")
        return None, None, None, None

# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="Advanced TPMS Features", layout="wide", page_icon="🏗️")
    
    st.title("🏗️ Advanced TPMS Features")
    st.markdown("*Sheet-Based Surfaces • Hybrid Structures • Graded Properties*")
    
    # Sidebar for feature selection
    with st.sidebar:
        st.header("Feature Selection")
        
        feature_type = st.selectbox(
            "Select Feature Type",
            ["Sheet-Based TPMS", "Hybrid Structures", "Graded Structures"],
            help="Choose which advanced feature to explore"
        )
        
        st.header("Parameters")
        
        # Common parameters
        resolution = st.slider("Resolution", 32, 128, 64, step=8, 
                             help="Higher resolution = more detail, slower computation")
        domain_periods = st.selectbox("Domain Periods", [1, 2, 3], index=1)
        domain_size = domain_periods * 2 * np.pi
        
        # Feature-specific parameters
        if feature_type == "Sheet-Based TPMS":
            st.subheader("🔧 Surface Parameters")
            tpms_type = st.selectbox(
                "TPMS Type", 
                ["Gyroid", "Schwarz P", "Schwarz D", "Neovius", "Primitive", "Fischer-Koch"]
            )
            thickness = st.slider("Thickness", -1.0, 1.0, 0.0, 0.1)
            
        elif feature_type == "Hybrid Structures":
            st.subheader("🔧 Hybrid Parameters")
            col1, col2 = st.columns(2)
            with col1:
                tpms1_type = st.selectbox("First TPMS", 
                    ["Gyroid", "Schwarz P", "Schwarz D", "Neovius"])
                thickness1 = st.slider("Thickness 1", -1.0, 1.0, 0.0, 0.1)
            with col2:
                tpms2_type = st.selectbox("Second TPMS", 
                    ["Schwarz P", "Gyroid", "Schwarz D", "Neovius"])
                thickness2 = st.slider("Thickness 2", -1.0, 1.0, 0.0, 0.1)
            
            blend_function = st.selectbox(
                "Blend Function",
                ["Union (OR)", "Intersection (AND)", "Weighted Average", 
                 "Spatial Transition", "XOR (Exclusive)"],
                help="How to combine the two TPMS structures"
            )
            
        else:  # Graded Structures
            st.subheader("Grading Parameters")
            tpms_type = st.selectbox(
                "TPMS Type",
                ["Gyroid", "Schwarz P", "Schwarz D", "Neovius"]
            )
            grading_function = st.selectbox(
                "Grading Function",
                ["Radial (Center to Edge)", "Linear X", "Linear Y", "Linear Z", 
                 "Sinusoidal X", "Spherical Shells", "Cubic Shells"],
                help="How thickness/density varies spatially"
            )
            base_thickness = st.slider("Base Thickness", -1.0, 1.0, 0.0, 0.1)
            grading_intensity = st.slider("Grading Intensity", 0.0, 2.0, 1.0, 0.1,
                                        help="Strength of the grading effect")
        
        # Generate button
        generate_btn = st.button("Generate Structure", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if generate_btn:
            with st.spinner(f"Generating {feature_type.lower()}..."):
                try:
                    if feature_type == "Sheet-Based TPMS":
                        vertices, faces, colors, name = create_sheet_based_tpms(
                            tpms_type, resolution, domain_size, thickness
                        ), None, None, tpms_type
                        vertices, faces, normals, colors = create_sheet_based_tpms(
                            tpms_type, resolution, domain_size, thickness
                        )
                        name = f"{tpms_type} Sheet Surface"
                        
                    elif feature_type == "Hybrid Structures":
                        vertices, faces, colors, name = create_hybrid_tpms(
                            tpms1_type, tpms2_type, blend_function, resolution,
                            domain_size, thickness1, thickness2
                        )
                        
                    else:  # Graded Structures
                        vertices, faces, colors, grading_name = create_graded_tpms(
                            tpms_type, grading_function, resolution, domain_size,
                            base_thickness, grading_intensity
                        )
                        name = f"{tpms_type} with {grading_name} Grading"
                    
                    if vertices is not None:
                        fig = plot_sheet_surface(vertices, faces, colors, name)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Store results for analysis
                            st.session_state['vertices'] = vertices
                            st.session_state['faces'] = faces
                            st.session_state['colors'] = colors
                            st.session_state['structure_name'] = name
                        else:
                            st.error("Failed to create visualization")
                    else:
                        st.error("Failed to generate structure")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Try reducing resolution or adjusting parameters")
        else:
            st.info("Configure parameters and click 'Generate Structure' to create advanced TPMS features")
    
    with col2:
        st.subheader("Structure Analysis")
        
        if 'vertices' in st.session_state:
            vertices = st.session_state['vertices']
            faces = st.session_state['faces']
            colors = st.session_state['colors']
            
            # Basic metrics
            st.metric("Vertices", f"{len(vertices):,}")
            st.metric("Faces", f"{len(faces):,}")
            
            # Surface area estimation
            if faces is not None and len(faces) > 0:
                # Calculate triangle areas
                v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
                v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
                areas = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
                total_area = np.sum(areas)
                st.metric("Surface Area", f"{total_area:.2f}")
            
            # Bounding box
            if len(vertices) > 0:
                bbox = np.ptp(vertices, axis=0)
                st.metric("Bounding Box", f"{bbox[0]:.1f} × {bbox[1]:.1f} × {bbox[2]:.1f}")
        
        # Feature descriptions
        st.subheader("Feature Info")
        
        if feature_type == "Sheet-Based TPMS":
            st.markdown("""
            **Sheet-Based TPMS** creates actual triangulated surfaces using the marching cubes algorithm:
            -  True 3D surfaces (not just points)
            -  Proper mesh topology
            -  Exportable to CAD/FEA
            -  Accurate surface area calculations
            -  Normal vectors for analysis
            """)
            
        elif feature_type == "Hybrid Structures":
            st.markdown("""
            **Hybrid Structures** combine multiple TPMS types:
            - **Union**: Overlapping regions merge
            - **Intersection**: Only common regions remain  
            - **Weighted Average**: Smooth blending
            - **Spatial Transition**: Location-based mixing
            - **XOR**: Exclusive regions only
            """)
            
        else:  # Graded Structures
            st.markdown("""
            **Graded Structures** vary properties spatially:
            - **Radial**: Density changes from center outward
            - **Linear**: Gradient along coordinate axes
            - **Sinusoidal**: Periodic thickness variation
            - **Shells**: Concentric density patterns
            
            Applications: Bone scaffolds, heat exchangers, impact absorption
            """)

if __name__ == "__main__":
    main()
