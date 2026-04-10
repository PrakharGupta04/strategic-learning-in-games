import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("⚔️ Strategic Game Simulation (Attacker vs Defender)")

# -----------------------------
# Payoff Matrix
# -----------------------------
payoff = np.array([
    [3, -1, 2],
    [-1, 4, -2],
    [2, -3, 5]
])

strategies = ["A1", "A2", "A3"]

# -----------------------------
# Initialize
# -----------------------------
if "attacker" not in st.session_state:
    st.session_state.attacker = np.array([1/3, 1/3, 1/3])
    st.session_state.history = []

# -----------------------------
# UI Controls
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    steps = st.slider("Simulation steps", 10, 500, 100)

with col2:
    speed = st.slider("Speed (ms)", 10, 200, 50)

start = st.button("▶️ Start Simulation")
reset = st.button("🔄 Reset")

# -----------------------------
# Reset
# -----------------------------
if reset:
    st.session_state.attacker = np.array([1/3, 1/3, 1/3])
    st.session_state.history = []

# -----------------------------
# Best Response Function
# -----------------------------
def best_response(p):
    expected = payoff.T @ p
    br = np.zeros(3)
    br[np.argmax(expected)] = 1
    return br

# -----------------------------
# Simulation
# -----------------------------
if start:
    chart = st.empty()
    payoff_text = st.empty()

    for i in range(steps):

        p = st.session_state.attacker

        # Fictitious play update
        br = best_response(p)
        new_p = (i * p + br) / (i + 1)

        st.session_state.attacker = new_p
        st.session_state.history.append(new_p.copy())

        history = np.array(st.session_state.history)

        # Plot
        fig, ax = plt.subplots()
        for j in range(3):
            ax.plot(history[:, j], label=strategies[j])
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("Strategy Evolution (Fictitious Play)")
        chart.pyplot(fig)

        # Payoff
        value = new_p @ payoff @ new_p
        payoff_text.write(f"💰 Current Expected Payoff: {value:.4f}")

        time.sleep(speed / 1000)

# -----------------------------
# Final Output
# -----------------------------
if st.session_state.history:
    final = st.session_state.attacker
    st.success(f"Final Strategy: {np.round(final, 3)}")