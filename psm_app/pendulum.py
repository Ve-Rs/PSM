import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time as timeImport


def simulate_pendulum(theta0, omega0, g, L, gamma, dt, t_max):
    """
    Simulate a mathematical pendulum using the Euler method.
    Returns arrays of theta and omega.
    """
    n_steps = int(t_max / dt)

    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    time = np.zeros(n_steps)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(1, n_steps):
        omega[i] = omega[i-1] - (g / L) * np.sin(theta[i-1]) * dt - gamma * omega[i-1] * dt
        theta[i] = theta[i-1] + omega[i] * dt
        time[i] = time[i-1] + dt

    return time, theta, omega


def run():
    
    placeholder = st.empty() # to fix state leak

    st.header("Mathematical Pendulum – Phase Space")

    st.markdown(
        """
        This simulation shows the phase space trajectory (θ, ω) of a
        mathematical pendulum with optional damping.
        """
    )

    # Sidebar parameters
    st.sidebar.subheader("Pendulum parameters")

    theta0_deg = st.sidebar.slider(
        "Initial angle θ₀ (degrees)",
        1, 179, 57
    )
    theta0 = np.radians(theta0_deg)

    omega0 = st.sidebar.slider(
        "Initial angular velocity ω₀",
        -2.0, 2.0, 0.0
    )

    g = st.sidebar.slider(
        "Gravitational acceleration g",
        1.0, 20.0, 9.81
    )

    L = st.sidebar.slider(
        "Pendulum length L",
        0.5, 5.0, 1.0
    )

    gamma = st.sidebar.slider(
        "Damping coefficient γ",
        0.0, 2.0, 0.0
    )

    dt = st.sidebar.slider(
        "Time step Δt",
        0.001, 0.05, 0.01
    )

    t_max = st.sidebar.slider(
        "Simulation time",
        5.0, 50.0, 20.0
    )

    # Run simulation
    time, theta, omega = simulate_pendulum(
        theta0, omega0, g, L, gamma, dt, t_max
    )

    # Optional time plots — show immediately (move before animation loop)
    with st.expander("Show time evolution"):
        fig2, ax2 = plt.subplots()
        ax2.plot(time, theta, label="θ(t)")
        ax2.plot(time, omega, label="ω(t)")
        ax2.legend()
        ax2.set_xlabel("Time")
        st.pyplot(fig2)

    placeholder = st.empty()

    # Setup outside the loop
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2) # Create an empty line object
    ax.set_xlabel("θ (angle)")
    ax.set_ylabel("ω (angular velocity)")
    ax.set_title("Phase Space Diagram")
    ax.set_xlim(-np.pi, np.pi)
    # set y limist based on omega range
    ax.set_ylim(min(omega)*1.1, max(omega)*1.1)

    for i in range(10, len(theta), 5):
        # Update only the data, no need to clear the axes!
        line.set_data(theta[:i], omega[:i])
        placeholder.pyplot(fig)
        timeImport.sleep(0.01)
