import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def get_triangle_neighbors(r, c, rows, cols):
    neighbors = set()
    is_up = (r + c) % 2 == 0

    # --- EDGE NEIGHBORS ---
    if is_up:
        edge_offsets = [
            (r, c - 1),   # left
            (r, c + 1),   # right
            (r + 1, c),   # bottom
        ]
    else:
        edge_offsets = [
            (r, c - 1),   # left
            (r, c + 1),   # right
            (r - 1, c),   # top
        ]

    # --- VERTEX (TIP) NEIGHBORS ---
    vertex_offsets = [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r,     c - 2),
        (r,     c + 2),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
        (r + 2, c),
    ]

    for nr, nc in edge_offsets + vertex_offsets:
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.add((nr, nc))

    return list(neighbors)


def apply_triangular_rules(grid, birth_cond, survive_cond):
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    
    for r in range(rows):
        for c in range(cols):
            coords = get_triangle_neighbors(r, c, rows, cols)
            # Count how many neighbors are alive (1)
            neighbor_count = sum(grid[nr, nc] for nr, nc in coords)
            
            if grid[r, c] == 1:
                if neighbor_count in survive_cond:
                    new_grid[r, c] = 1
            else:
                if neighbor_count in birth_cond:
                    new_grid[r, c] = 1
    return new_grid

def draw_triangular_grid(grid):
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    # Height of an equilateral triangle with side 1
    h = np.sqrt(3) / 2
    
    for r in range(rows):
        for c in range(cols):
            # Calculate vertices based on row and column
            # Horizontal shift is 0.5 per column
            x_offset = c * 0.5
            y_offset = r * h
            
            if (r + c) % 2 == 0: # Pointing Up
                nodes = [[x_offset, y_offset], [x_offset + 1, y_offset], [x_offset + 0.5, y_offset + h]]
            else: # Pointing Down
                nodes = [[x_offset, y_offset + h], [x_offset + 1, y_offset + h], [x_offset + 0.5, y_offset]]
            
            color = 'forestgreen' if grid[r, c] == 1 else 'white'
            poly = Polygon(nodes, facecolor=color, edgecolor='silver', linewidth=0.5)
            ax.add_patch(poly)

    ax.set_xlim(-0.5, cols * 0.5 + 0.5)
    ax.set_ylim(-0.5, rows * h + 0.5)
    ax.axis('off')
    return fig
def run():
    st.header("Triangular Life: Branching Timelines")
    st.info("Going 'Back' will delete the current future, allowing you to change rules and start a new path.")

    # --- 1. SESSION STATE ---
    if 'tri_history' not in st.session_state:
        # Initial random grid
        st.session_state.tri_grid = (np.random.rand(12, 16) < 0.3).astype(int)
        st.session_state.tri_history = [st.session_state.tri_grid]
        st.session_state.tri_step = 0

    # --- 2. SIDEBAR ---
    st.sidebar.subheader("🧬 Triangular Rules")
    birth_opts = st.sidebar.multiselect("Birth (B)", range(13), default=[2, 3])
    survive_opts = st.sidebar.multiselect("Survival (S)", range(13), default=[1,2 ])

    if st.sidebar.button("Reset Simulation"):
        st.session_state.tri_grid = (np.random.rand(12, 16) < 0.3).astype(int)
        st.session_state.tri_history = [st.session_state.tri_grid]
        st.session_state.tri_step = 0
        st.rerun()

    # --- 3. BRANCHING NAVIGATION ---
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        # PREVIOUS: Go back AND delete the future
        if st.button("⬅️ Back (Delete Future)") and st.session_state.tri_step > 0:
            st.session_state.tri_step -= 1
            # The Magic Slice: Keep only up to the NEW current step
            st.session_state.tri_history = st.session_state.tri_history[:st.session_state.tri_step + 1]
            st.rerun()
            
    with c2:
        # NEXT: Always calculate based on CURRENT rules
        if st.button("➡️ Next (New Future)"):
            current_state = st.session_state.tri_history[st.session_state.tri_step]
            next_gen = apply_triangular_rules(current_state, birth_opts, survive_opts)
            
            # Append the new state to the end of our history
            st.session_state.tri_history.append(next_gen)
            st.session_state.tri_step += 1
            st.rerun()
            
    with c3:
        st.write(f"Generation: **{st.session_state.tri_step}**")

    # --- 4. RENDER ---
    # We always render the LAST item in the history because 
    # the "Back" button deletes everything after it.
    current_grid = st.session_state.tri_history[st.session_state.tri_step]
    fig = draw_triangular_grid(current_grid)
    st.pyplot(fig)