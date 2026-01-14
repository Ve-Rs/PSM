import streamlit as st

# Import simulation modules
import pendulum
import double_pendulum
import life
import fractals
import collisions


st.set_page_config(
    page_title="PSM Numerical Simulations",
    layout="wide"
)

st.title("PSM Numerical Simulations")
st.markdown(
    """
    This application presents several non-trivial numerical simulations
    implemented in Python for the PSM course.
    """
)

# Sidebar navigation
simulation = st.sidebar.radio(
    "Choose a simulation",
    [
        "Pendulum (phase space)",
        "Double pendulum",
        "Game of Life",
        "Fractals",
        "Collisions"
    ]
)

# Dispatch
if simulation == "Pendulum (phase space)":
    pendulum.run()

elif simulation == "Double pendulum":
    double_pendulum.run()

elif simulation == "Game of Life":
    life.run()

elif simulation == "Fractals":
    fractals.run()

elif simulation == "Collisions":
    collisions.run()
