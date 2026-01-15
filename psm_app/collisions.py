import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


# ---------- COLLISION PHYSICS ----------

def elastic_collision(v1, v2, x1, x2, m1, m2, restitution=1.0):
    n = x2 - x1
    n = n / np.linalg.norm(n)

    v1n = np.dot(v1, n)
    v2n = np.dot(v2, n)

    v1t = v1 - v1n * n
    v2t = v2 - v2n * n

    v1n_new = ((m1 - restitution*m2) * v1n + (1 + restitution) * m2 * v2n) / (m1 + m2)
    v2n_new = ((m2 - restitution*m1) * v2n + (1 + restitution) * m1 * v1n) / (m1 + m2)

    v1_new = v1t + v1n_new * n
    v2_new = v2t + v2n_new * n

    return v1_new, v2_new


# ---------- STREAMLIT APP ----------

def run():
    placeholder = st.empty() # to fix state leak
    st.header("Collision of Two Material Points – Numeric Task 8")

    st.markdown("""
    This application visualizes elastic and inelastic collisions
    of two material points with configurable masses and velocities.
    """)

    # ---- SIDEBAR PARAMETERS ----
    st.sidebar.subheader("Physical parameters")

    m1 = st.sidebar.slider("Mass m1", 0.5, 5.0, 1.0)
    m2 = st.sidebar.slider("Mass m2", 0.5, 20.0, 5.0)

    v1x = st.sidebar.slider("Initial v1x", -5.0, 5.0, 2.0)
    v2x = st.sidebar.slider("Initial v2x", -5.0, 5.0, -1.0)

    restitution = st.sidebar.slider("Restitution (elasticity)", 0.0, 1.0, 1.0)
    speed = st.sidebar.slider("Simulation speed", 0.005, 0.05, 0.02)

    # ---- INITIAL STATE ----
    x1 = np.array([-3.0, 0.0])
    x2 = np.array([3.0, 0.0])

    v1 = np.array([v1x, 0.0])
    v2 = np.array([v2x, 0.0])

    r1 = r2 = 0.3
    dt = 0.02

    # ---- PLOT SETUP ----
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)

    placeholder = st.empty()
    collided = False

    # ---- ANIMATION LOOP ----
    for _ in range(300):
        x1 += v1 * dt
        x2 += v2 * dt

        dist = np.linalg.norm(x2 - x1)

        if dist <= r1 + r2 and not collided:
            v1, v2 = elastic_collision(v1, v2, x1, x2, m1, m2, restitution)
            collided = True

        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 2)

        # Draw particles
        ax.add_patch(plt.Circle(x1, r1, color="blue"))
        ax.add_patch(plt.Circle(x2, r2, color="red"))

        # Labels
        ax.text(x1[0], x1[1] + 0.5, "m1", ha="center")
        ax.text(x2[0], x2[1] + 0.5, "m2", ha="center")

        ax.set_title("Particle collision simulation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        placeholder.pyplot(fig)
        time.sleep(speed/10)
