import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. BARNSLEY FERN
# =====================================================

def barnsley_fern(n_points):
    x, y = 0.0, 0.0
    xs, ys = [], []

    for _ in range(n_points):
        r = np.random.rand()
        if r < 0.01:
            x, y = 0, 0.16 * y
        elif r < 0.86:
            x, y = 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6
        elif r < 0.93:
            x, y = 0.20 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6
        else:
            x, y = -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44

        xs.append(x)
        ys.append(y)

    return xs, ys

# =====================================================
# 2. MANDELBROT SET
# =====================================================

def mandelbrot(width, height, max_iter, zoom, cx, cy):
    x = np.linspace(-2.5/zoom + cx, 1.5/zoom + cx, width)
    y = np.linspace(-1.5/zoom + cy, 1.5/zoom + cy, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.zeros(C.shape)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] = i

    return M

# =====================================================
# 3. CUSTOM FRACTAL (JULIA SET)
# =====================================================

def julia_set(width, height, c, max_iter):
    x = np.linspace(-1.5, 1.5, width)
    y = np.linspace(-1.5, 1.5, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    M = np.zeros(Z.shape)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 3 + c
        M[mask] = i

    return M

# =====================================================
# STREAMLIT UI
# =====================================================

def run():
    st.header("Fractals – Numeric Task 10")

    fractal_type = st.sidebar.radio(
        "Choose fractal",
        ["Barnsley Fern", "Mandelbrot Set", "Custom Julia Set"]
    )

    if fractal_type == "Barnsley Fern":
        st.subheader("Barnsley Fern (IFS)")

        n = st.sidebar.slider("Number of points", 1_000, 100_000, 30_000)

        xs, ys = barnsley_fern(n)

        fig, ax = plt.subplots(figsize=(5, 7))
        ax.scatter(xs, ys, s=0.1, color="green")
        ax.set_axis_off()
        st.pyplot(fig)

    elif fractal_type == "Mandelbrot Set":
        st.subheader("Mandelbrot Fractal")

        max_iter = st.sidebar.slider("Max iterations", 5, 500, 200)
        zoom = st.sidebar.slider("Zoom", 1.0, 10.0, 1.0)
        cx = st.sidebar.slider("Center X", -1.0, 1.0, -0.5)
        cy = st.sidebar.slider("Center Y", -1.0, 1.0, 0.0)

        M = mandelbrot(600, 600, max_iter, zoom, cx, cy)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(M, cmap="inferno")
        ax.set_axis_off()
        st.pyplot(fig)

    else:
        st.subheader("Custom Julia Set")

        re = st.sidebar.slider("Re(c)", -1.0, 1.0, -0.7)
        im = st.sidebar.slider("Im(c)", -1.0, 1.0, 0.27015)
        max_iter = st.sidebar.slider("Max iterations", 5, 500, 200)

        J = julia_set(600, 600, complex(re, im), max_iter)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(J, cmap="plasma")
        ax.set_axis_off()
        st.pyplot(fig)
