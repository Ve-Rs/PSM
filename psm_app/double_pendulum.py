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
    st.header("Pendulum Experiments – Motion Trace (Slider Controlled)")

    mode = st.sidebar.radio(
        "Pendulum type",
        ["Simple pendulum", "Double pendulum"]
    )

    g = st.sidebar.slider("Gravity g", 1.0, 20.0, 9.81)
    dt = st.sidebar.slider("Time step Δt", 0.01, 0.05, 0.02)
    t_max = st.sidebar.slider("Simulation time", 5.0, 30.0, 15.0)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if mode == "Simple pendulum":
        st.subheader("Simple pendulum")

        L = st.sidebar.slider("Length L", 0.5, 3.0, 1.5)
        theta0 = np.radians(st.sidebar.slider("Initial angle (deg)", 5, 170, 60))

        x, y = simulate_simple(theta0, 0.0, g, L, dt, t_max)

        i = st.slider(
            "Time step",
            0,
            len(x) - 1,
            0
        )

        ax.plot(x[:i], y[:i], color="blue", label="Trajectory")
        ax.plot([0, x[i]], [0, y[i]], color="black", linewidth=2)
        ax.scatter(x[i], y[i], color="red", s=40)
        ax.set_title(f"t = {i * dt:.2f} s")
        ax.legend()

        st.pyplot(fig)

    else:
        st.subheader("Double pendulum")

        L1 = st.sidebar.slider("Length L1", 0.5, 2.5, 1.2)
        L2 = st.sidebar.slider("Length L2", 0.5, 2.5, 1.2)
        m1 = st.sidebar.slider("Mass m1", 0.5, 3.0, 1.0)
        m2 = st.sidebar.slider("Mass m2", 0.5, 3.0, 1.0)

        t1 = np.radians(st.sidebar.slider("θ1 initial (deg)", 5, 170, 90))
        t2 = np.radians(st.sidebar.slider("θ2 initial (deg)", 5, 170, 120))

        x1, y1, x2, y2 = simulate_double(t1, t2, g, L1, L2, m1, m2, dt, t_max)

        i = st.slider(
            "Time step",
            0,
            len(x2) - 1,
            0
        )

        ax.plot(x2[:i], y2[:i], color="purple", linewidth=1, label="Trace (mass 2)")
        ax.plot([0, x1[i]], [0, y1[i]], color="black", linewidth=2)
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color="black", linewidth=2)

        ax.scatter(x1[i], y1[i], color="blue", s=30, label="Mass 1")
        ax.scatter(x2[i], y2[i], color="red", s=20, label="Mass 2")

        ax.set_title(f"t = {i * dt:.2f} s")
        ax.legend()

        st.pyplot(fig)


if __name__ == "__main__":
    run()
