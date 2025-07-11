import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import io

st.title("Demo: Calculate + Export buttons on top")

# Inputs
input_val = st.number_input("Enter a number", value=10)

# Place buttons in a single row with columns
calc_col, export_csv_col, export_png_col = st.columns([2, 1, 1])

with calc_col:
    calculate_clicked = st.button("Calculate")

# Initialize session state vars if not exist
if "calc_done" not in st.session_state:
    st.session_state.calc_done = False
if "export_csv" not in st.session_state:
    st.session_state.export_csv = None
if "export_png" not in st.session_state:
    st.session_state.export_png = None

if calculate_clicked:
    # Do some calculation
    df = pd.DataFrame({
        "x": range(input_val),
        "y": np.random.randn(input_val)
    })
    fig = go.Figure()
    fig.add_scatter(x=df["x"], y=df["y"], mode="lines+markers")

    # Prepare CSV bytes
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    # Prepare PNG bytes from figure
    png_bytes = fig.to_image(format="png")

    # Save to session_state
    st.session_state.calc_done = True
    st.session_state.export_csv = csv_bytes
    st.session_state.export_png = png_bytes

    st.write("Calculation done! Here's the plot:")
    st.plotly_chart(fig)

# Export buttons always visible but disabled if no calculation done
with export_csv_col:
    st.download_button(
        label="Export CSV",
        data=st.session_state.export_csv,
        file_name="data.csv",
        mime="text/csv",
        disabled=not st.session_state.calc_done
    )

with export_png_col:
    st.download_button(
        label="Export PNG",
        data=st.session_state.export_png,
        file_name="plot.png",
        mime="image/png",
        disabled=not st.session_state.calc_done
    )
