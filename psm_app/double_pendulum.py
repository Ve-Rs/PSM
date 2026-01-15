import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


# ---------- SIMPLE PENDULUM ----------

def simulate_simple(theta0, omega0, g, L, dt, t_max, gamma=0.1):
    n = int(t_max / dt)

    theta = np.zeros(n)
    omega = np.zeros(n)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(1, n):
        omega[i] = omega[i-1] - (g / L) * np.sin(theta[i-1]) * dt - gamma * omega[i-1] * dt
        theta[i] = theta[i-1] + omega[i] * dt

    x = L * np.sin(theta)
    y = -L * np.cos(theta)

    return x, y


# ---------- DOUBLE PENDULUM ----------

def simulate_double(theta1_init, theta2_init, g, L1, L2, m1, m2, dt, t_max, gamma=0.1):
    """
    Simulates a double pendulum with added Damping (gamma).
    Higher gamma = more friction (the pendulum stops faster).
    """
    n_steps = int(t_max / dt)
    
    theta1 = np.zeros(n_steps); theta2 = np.zeros(n_steps)
    omega1 = np.zeros(n_steps); omega2 = np.zeros(n_steps)

    theta1[0] = theta1_init; theta2[0] = theta2_init
    omega1[0] = 0.0; omega2[0] = 0.0

    for i in range(1, n_steps):
        t1_old, t2_old = theta1[i-1], theta2[i-1]
        w1_old, w2_old = omega1[i-1], omega2[i-1]
        
        delta = t2_old - t1_old
        sin_delta, cos_delta = np.sin(delta), np.cos(delta)
        
        # --- EQUATION 1: Top Mass ---
        den1 = (m1 + m2) * L1 - m2 * L1 * cos_delta**2
        
        num1 = (m2 * L1 * w1_old**2 * sin_delta * cos_delta # Centrifugal force
                + m2 * g * np.sin(t2_old) * cos_delta # Gravity on bottom mass
                + m2 * L2 * w2_old**2 * sin_delta # Bottom mass motion
                - (m1 + m2) * g * np.sin(t1_old)) # Gravity on top mass
        
        # Add Damping to the acceleration: -gamma * velocity
        alpha1 = (num1 / den1) - (gamma * w1_old)

        # --- EQUATION 2: Bottom Mass ---
        den2 = (L2 / L1) * den1
        
        num2 = (-m2 * L2 * w2_old**2 * sin_delta * cos_delta # Centrifugal force
                + (m1 + m2) * (g * np.sin(t1_old) * cos_delta # Gravity on top mass
                - L1 * w1_old**2 * sin_delta # Top mass motion
                - g * np.sin(t2_old))) # Gravity on bottom mass
        
        # Add Damping to the acceleration: -gamma * velocity
        alpha2 = (num2 / den2) - (gamma * w2_old)

        # --- UPDATE STATE (Euler Step) ---
        omega1[i] = w1_old + alpha1 * dt
        omega2[i] = w2_old + alpha2 * dt
        theta1[i] = t1_old + omega1[i] * dt
        theta2[i] = t2_old + omega2[i] * dt

    # Convert to X/Y for plotting
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2


# ---------- STREAMLIT APP ----------

def run():
    placeholder = st.empty() # to fix state leak
    st.header("Pendulum Experiments – Motion Trace (Task 7)")

    st.markdown(
        """
        This simulation presents experiments with pendulum motion
        in real space. The trajectory (trace) of the pendulum mass
        is drawn in the horizontal plane.
        """
    )

    mode = st.sidebar.radio(
        "Pendulum type",
        ["Simple pendulum", "Double pendulum"]
    )

    g = st.sidebar.slider("Gravity g", 1.0, 20.0, 9.81)
    dt = st.sidebar.slider("Time step Δt", 0.01, 0.05, 0.01)
    t_max = st.sidebar.slider("Simulation time", 5.0, 30.0, 15.0)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    placeholder = st.empty()

    if mode == "Simple pendulum":
        st.subheader("Simple pendulum")

        L = st.sidebar.slider("Length L", 0.5, 3.0, 1.5)
        theta0 = np.radians(st.sidebar.slider("Initial angle (deg)", 5, 170, 60))
        omega0 = 0.0

        x, y = simulate_simple(theta0, omega0, g, L, dt, t_max)

        for i in range(10, len(x), 3):
            ax.clear()
            ax.plot(x[:i], y[:i], color="blue")
            ax.plot([0, x[i]], [0, y[i]], color="black")
            ax.scatter(x[i], y[i], color="red")
            ax.set_aspect("equal")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_title("Simple pendulum – trajectory")
            placeholder.pyplot(fig)
            time.sleep(0.02)

    else:
        st.subheader("Double pendulum (complex system)")

        L1 = st.sidebar.slider("Length L1", 0.5, 2.5, 1.2)
        L2 = st.sidebar.slider("Length L2", 0.5, 2.5, 1.2)
        m1 = st.sidebar.slider("Mass m1", 0.5, 3.0, 1.0)
        m2 = st.sidebar.slider("Mass m2", 0.5, 3.0, 1.0)

        theta1 = np.radians(st.sidebar.slider("θ1 initial (deg)", 5, 170, 90))
        theta2 = np.radians(st.sidebar.slider("θ2 initial (deg)", 5, 170, 120))

        x1, y1, x2, y2 = simulate_double(theta1, theta2, g, L1, L2, m1, m2, dt, t_max)

        for i in range(10, len(x2), 3):
            ax.clear()
            # Trace of second mass
            ax.plot(x2[:i], y2[:i], color="purple", linewidth=1)

            # Rods
            ax.plot([0, x1[i]], [0, y1[i]], color="black", linewidth=2)      # first rod
            ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color="black", linewidth=2)  # second rod

            # Masses
            ax.scatter(x1[i], y1[i], color="blue", s=30, label="Mass 1")
            ax.scatter(x2[i], y2[i], color="red", s=20, label="Mass 2")
            ax.set_aspect("equal")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_title("Double pendulum – chaotic trace")
            placeholder.pyplot(fig)
            time.sleep(0.02)
