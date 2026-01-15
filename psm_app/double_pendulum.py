import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =========================
# SIMULATION FUNCTIONS
# =========================

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


def simulate_double(theta1_init, theta2_init, g, L1, L2, m1, m2, dt, t_max, gamma=0.1):
    n = int(t_max / dt)

    theta1 = np.zeros(n)
    theta2 = np.zeros(n)
    omega1 = np.zeros(n)
    omega2 = np.zeros(n)

    theta1[0] = theta1_init
    theta2[0] = theta2_init

    for i in range(1, n):
        t1, t2 = theta1[i-1], theta2[i-1]
        w1, w2 = omega1[i-1], omega2[i-1]
        delta = t2 - t1

        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        a1 = (
            m2 * L1 * w1**2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(t2) * np.cos(delta)
            + m2 * L2 * w2**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(t1)
        ) / den1 - gamma * w1

        den2 = (L2 / L1) * den1
        a2 = (
            -m2 * L2 * w2**2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * (
                g * np.sin(t1) * np.cos(delta)
                - L1 * w1**2 * np.sin(delta)
                - g * np.sin(t2)
            )
        ) / den2 - gamma * w2

        omega1[i] = w1 + a1 * dt
        omega2[i] = w2 + a2 * dt
        theta1[i] = t1 + omega1[i] * dt
        theta2[i] = t2 + omega2[i] * dt

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2


# =========================
# STREAMLIT APP
# =========================

def run():
    st.header("Pendulum Experiments – Motion Trace")

    mode = st.sidebar.radio(
        "Pendulum type",
        ["Simple pendulum", "Double pendulum"]
    )

    g = st.sidebar.slider("Gravity g", 1.0, 20.0, 9.81)
    dt = st.sidebar.slider("Time step Δt", 0.01, 0.05, 0.02)
    t_max = st.sidebar.slider("Simulation time", 5.0, 30.0, 15.0)
    frame_skip = st.sidebar.slider("Animation speed", 1, 10, 3)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    plot_area = st.empty()

    if mode == "Simple pendulum":
        L = st.sidebar.slider("Length L", 0.5, 3.0, 1.5)
        theta0 = np.radians(st.sidebar.slider("Initial angle (deg)", 5, 170, 60))

        x, y = simulate_simple(theta0, 0.0, g, L, dt, t_max)

        trace, = ax.plot([], [], "b-")
        rod, = ax.plot([], [], "k-")
        mass = ax.scatter([], [], c="red")

        for i in range(0, len(x), frame_skip):
            trace.set_data(x[:i], y[:i])
            rod.set_data([0, x[i]], [0, y[i]])
            mass.set_offsets([[x[i], y[i]]])
            ax.set_title("Simple pendulum")
            plot_area.pyplot(fig)

    else:
        L1 = st.sidebar.slider("Length L1", 0.5, 2.5, 1.2)
        L2 = st.sidebar.slider("Length L2", 0.5, 2.5, 1.2)
        m1 = st.sidebar.slider("Mass m1", 0.5, 3.0, 1.0)
        m2 = st.sidebar.slider("Mass m2", 0.5, 3.0, 1.0)

        t1 = np.radians(st.sidebar.slider("θ1 initial (deg)", 5, 170, 90))
        t2 = np.radians(st.sidebar.slider("θ2 initial (deg)", 5, 170, 120))

        x1, y1, x2, y2 = simulate_double(t1, t2, g, L1, L2, m1, m2, dt, t_max)

        trace, = ax.plot([], [], "purple", lw=1)
        rod1, = ax.plot([], [], "k-", lw=2)
        rod2, = ax.plot([], [], "k-", lw=2)
        m1_dot = ax.scatter([], [], c="blue", s=30)
        m2_dot = ax.scatter([], [], c="red", s=20)

        for i in range(0, len(x2), frame_skip):
            trace.set_data(x2[:i], y2[:i])
            rod1.set_data([0, x1[i]], [0, y1[i]])
            rod2.set_data([x1[i], x2[i]], [y1[i], y2[i]])
            m1_dot.set_offsets([[x1[i], y1[i]]])
            m2_dot.set_offsets([[x2[i], y2[i]]])
            ax.set_title("Double pendulum (chaotic motion)")
            plot_area.pyplot(fig)


if __name__ == "__main__":
    run()
