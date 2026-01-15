import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# =========================
# PHYSICS MODEL
# =========================

def simulate_pendulum(theta0, omega0, g, L, gamma, dt, t_max):
    """
    Simulate a mathematical pendulum using the Euler method.
    Returns time, theta, omega arrays.
    """
    n = int(t_max / dt)

    theta = np.zeros(n)
    omega = np.zeros(n)
    time = np.zeros(n)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(1, n):
        omega[i] = omega[i-1] - (g / L) * np.sin(theta[i-1]) * dt - gamma * omega[i-1] * dt
        theta[i] = theta[i-1] + omega[i] * dt
        time[i] = time[i-1] + dt

    return time, theta, omega


# =========================
# STREAMLIT APP
# =========================

def run():
    st.header("Mathematical Pendulum – Phase Space (θ, ω)")

    st.markdown(
        """
        This simulation presents the phase-space trajectory of a mathematical
        pendulum. The system evolution is explored using a time-step slider.
        """
    )

    # ---------- Sidebar Controls ----------
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

    # ---------- Run Simulation Once ----------
    time, theta, omega = simulate_pendulum(
        theta0, omega0, g, L, gamma, dt, t_max
    )

    # ---------- Time-Step Slider ----------
    i = st.slider(
        "Time step",
        0,
        len(time) - 1,
        0
    )

    # ---------- Phase Space Plot ----------
    fig, ax = plt.subplots()
    ax.plot(theta[:i], omega[:i], color="blue", linewidth=2)
    ax.scatter(theta[i], omega[i], color="red", s=40)

    ax.set_xlabel("θ (angle)")
    ax.set_ylabel("ω (angular velocity)")
    ax.set_title(f"Phase Space at t = {time[i]:.2f} s")

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(min(omega) * 1.1, max(omega) * 1.1)
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

    # ---------- Optional Time Evolution ----------
    with st.expander("Show time evolution θ(t) and ω(t)"):
        fig2, ax2 = plt.subplots()
        ax2.plot(time, theta, label="θ(t)")
        ax2.plot(time, omega, label="ω(t)")
        ax2.set_xlabel("Time")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig2)


if __name__ == "__main__":
    run()
