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

#-- simulation and rendering --
def simulate_collision(
    m1, m2, v1_init, v2_init, offset,
    restitution, dt=0.02, steps=200
):
    x1 = np.array([-4.0, 0.0])
    x2 = np.array([4.0, offset])

    v1 = np.array([v1_init, 0.0])
    v2 = np.array([v2_init, 0.0])

    R1 = R2 = 0.4

    history = []

    for _ in range(steps):
        x1 = x1 + v1 * dt
        x2 = x2 + v2 * dt

        v1, v2, _ = compute_collision(
            x1, x2, v1, v2,
            m1, m2, R1, R2, restitution
        )

        history.append({
            "x1": x1.copy(),
            "x2": x2.copy(),
            "v1": v1.copy(),
            "v2": v2.copy()
        })

    return history

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
    
    history = simulate_collision(
        m1, m2,
        v1_init, v2_init,
        offset, restitution
    )

    step = st.slider(
        "Simulation step",
        0, len(history) - 1,
        0
    )

    state = history[step]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Particles
    ax.add_patch(plt.Circle(state["x1"], 0.4, color="royalblue"))
    ax.add_patch(plt.Circle(state["x2"], 0.4, color="crimson"))

    # Velocity vectors
    ax.arrow(
        state["x1"][0], state["x1"][1],
        state["v1"][0] * 0.5, state["v1"][1] * 0.5,
        head_width=0.2, color="royalblue"
    )

    ax.arrow(
        state["x2"][0], state["x2"][1],
        state["v2"][0] * 0.5, state["v2"][1] * 0.5,
        head_width=0.2, color="crimson"
    )


    st.pyplot(fig)