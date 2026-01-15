import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------- 1. PHYSICS ENGINE ----------

def compute_collision(x1, x2, v1, v2, m1, m2, r1, r2, restitution):
    """
    Checks and resolves collision between two spheres.
    Returns: (v1_new, v2_new, has_collided)
    """
    dist_vec = x1 - x2
    dist = np.linalg.norm(dist_vec)
    
    # Check if touching (and moving towards each other)
    if dist <= (r1 + r2):
        # Normal vector (direction of impact)
        n = dist_vec / dist
        
        # Relative velocity
        v_rel = v1 - v2
        
        # Velocity along the normal (impact speed)
        vel_along_normal = np.dot(v_rel, n)
        
        # If moving away, do not resolve (prevents sticking)
        if vel_along_normal > 0:
            return v1, v2, False

        # Impulse scalar (J)
        # J = -(1 + e) * (v_rel . n) / (1/m1 + 1/m2)
        j = -(1 + restitution) * vel_along_normal
        j /= (1 / m1 + 1 / m2)
        
        # Apply impulse
        impulse = j * n
        v1_new = v1 + impulse / m1
        v2_new = v2 - impulse / m2
        
        return v1_new, v2_new, True
        
    return v1, v2, False

# ---------- 2. STREAMLIT APP ----------

def run():
    st.header("2D Elastic/Inelastic Collisions")
    
    # --- Sidebar Controls ---
    st.sidebar.subheader("Simulation Settings")
    
    # Speed Control: Changes dt, not sleep!
    time_scale = st.sidebar.slider("Time Scale (Speed)", 0.1, 2.0, 1.0, 0.1, 
                                   help="< 1.0 is Slow Motion, > 1.0 is Fast Forward")
    
    restitution = st.sidebar.slider("Restitution (e)", 0.0, 1.0, 1.0, 
                                    help="1.0 = Perfectly Elastic (Superball), 0.0 = Sticky Clay")

    st.sidebar.subheader("🔵 Blue Particle (m1)")
    m1 = st.sidebar.number_input("Mass m1", 0.1, 10.0, 2.0)
    v1_init = st.sidebar.slider("Velocity X (m1)", -5.0, 5.0, 3.0)
    
    st.sidebar.subheader("🔴 Red Particle (m2)")
    m2 = st.sidebar.number_input("Mass m2", 0.1, 10.0, 2.0)
    v2_init = st.sidebar.slider("Velocity X (m2)", -5.0, 5.0, -2.0)
    offset = st.sidebar.slider("Impact Parameter (Offset Y)", 0.0, 1.0, 0.2, 
                               help="Shifts m2 up/down to create glancing collisions")

    # --- Session State for Animation ---
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
        st.session_state.x1 = np.array([-4.0, 0.0])
        st.session_state.x2 = np.array([4.0, offset])
        st.session_state.v1 = np.array([v1_init, 0.0])
        st.session_state.v2 = np.array([v2_init, 0.0])
        st.session_state.energy_history = []

    # --- Control Buttons ---
    col1, col2, col3 = st.columns(3)
    
    if col1.button("▶️ Start"):
        st.session_state.sim_running = True
        
    if col2.button("⏸️ Stop"):
        st.session_state.sim_running = False
        
    if col3.button("Pw Reset"):
        st.session_state.sim_running = False
        st.session_state.x1 = np.array([-4.0, 0.0])
        st.session_state.x2 = np.array([4.0, offset])
        st.session_state.v1 = np.array([v1_init, 0.0])
        st.session_state.v2 = np.array([v2_init, 0.0])
        st.session_state.energy_history = []
        st.rerun()

    # --- Plot Placeholders ---
    # We use two columns: Animation on left, Energy Graph on right
    plot_col, stats_col = st.columns([2, 1])
    
    with plot_col:
        plot_placeholder = st.empty()
    with stats_col:
        energy_placeholder = st.empty()

    # --- Simulation Loop ---
    # CONSTANTS
    R1, R2 = 0.4, 0.4  # Radii
    BASE_DT = 0.03     # Base physics step
    
    if st.session_state.sim_running:
        # Run a small batch of frames
        for _ in range(200): # Limit max frames per click to prevent infinite loops hanging browser
            if not st.session_state.sim_running:
                break
                
            # 1. Physics Step (Scaled by User Speed)
            dt = BASE_DT * time_scale
            
            # Update Positions
            st.session_state.x1 += st.session_state.v1 * dt
            st.session_state.x2 += st.session_state.v2 * dt
            
            # Detect & Resolve Collision
            v1_new, v2_new, hit = compute_collision(
                st.session_state.x1, st.session_state.x2,
                st.session_state.v1, st.session_state.v2,
                m1, m2, R1, R2, restitution
            )
            
            st.session_state.v1 = v1_new
            st.session_state.v2 = v2_new
            
            # Calculate Energy
            E_kin = 0.5 * m1 * np.linalg.norm(st.session_state.v1)**2 + \
                    0.5 * m2 * np.linalg.norm(st.session_state.v2)**2
            st.session_state.energy_history.append(E_kin)
            
            # 2. Rendering (The Animation)
            fig, ax = plt.subplots(figsize=(5, 3)) # Small size for speed
            ax.set_xlim(-6, 6)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Draw Particles
            c1 = plt.Circle(st.session_state.x1, R1, color='royalblue', alpha=0.9)
            c2 = plt.Circle(st.session_state.x2, R2, color='crimson', alpha=0.9)
            ax.add_patch(c1)
            ax.add_patch(c2)
            
            # Draw Velocity Vectors (Arrows)
            ax.arrow(st.session_state.x1[0], st.session_state.x1[1], 
                     st.session_state.v1[0]*0.5, st.session_state.v1[1]*0.5, 
                     head_width=0.2, color='royalblue')
            ax.arrow(st.session_state.x2[0], st.session_state.x2[1], 
                     st.session_state.v2[0]*0.5, st.session_state.v2[1]*0.5, 
                     head_width=0.2, color='crimson')

            plot_placeholder.pyplot(fig)
            plt.close(fig) # Crucial: Close memory leak
            

            # Smooth Frame Rate Control
            # We always sleep a tiny bit to let the UI breathe, 
            # but the SPEED is controlled by 'dt' above.
            time.sleep(0.01) 
            
    else:
        # Static View (When stopped/loaded)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        c1 = plt.Circle(st.session_state.x1, R1, color='royalblue')
        c2 = plt.Circle(st.session_state.x2, R2, color='crimson')
        ax.add_patch(c1)
        ax.add_patch(c2)
        
        plot_placeholder.pyplot(fig)