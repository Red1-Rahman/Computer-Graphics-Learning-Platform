import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Line Drawing Algorithms", layout="wide")

st.title("Line Drawing Algorithms")
st.markdown("Visualize **DDA** and **Bresenham** line drawing algorithms step by step.")

# ── Input Section ──────────────────────────────────────────────────────────────
st.header("Input Points")
col1, col2, col3, col4 = st.columns(4)
with col1:
    x1 = st.number_input("x₁ (start)", value=0, step=1)
with col2:
    y1 = st.number_input("y₁ (start)", value=0, step=1)
with col3:
    x2 = st.number_input("x₂ (end)", value=6, step=1)
with col4:
    y2 = st.number_input("y₂ (end)", value=4, step=1)

x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

st.divider()

# ── Algorithm Selection ────────────────────────────────────────────────────────
algo_col1, algo_col2 = st.columns(2)
show_dda = algo_col1.checkbox("DDA Algorithm", value=True)
show_bres = algo_col2.checkbox("Bresenham Algorithm", value=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# DDA Algorithm
# ══════════════════════════════════════════════════════════════════════════════
def run_dda(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return None, None, [(x1, y1, x1, y1)]

    slope_str = None
    slope_note = ""

    if dx == 0:
        slope_str = "undefined (vertical line)"
        slope_val = None
    else:
        slope_val = dy / dx
        slope_str = f"{dy}/{dx} = {slope_val:.4f}"

    if abs(dx) >= abs(dy):
        slope_note = f"|slope| ≤ 1  →  step along X axis  (steps = {steps})"
    else:
        slope_note = f"|slope| > 1  →  step along Y axis  (steps = {steps})"

    x_inc = dx / steps
    y_inc = dy / steps

    rows = []
    cx, cy = float(x1), float(y1)
    for i in range(steps + 1):
        rows.append({
            "Step (i)": i,
            "x (exact)": round(cx, 4),
            "y (exact)": round(cy, 4),
            "x (rounded)": round(cx),
            "y (rounded)": round(cy),
        })
        cx += x_inc
        cy += y_inc

    return slope_str, slope_note, rows


# ══════════════════════════════════════════════════════════════════════════════
# Bresenham Algorithm
# ══════════════════════════════════════════════════════════════════════════════
def run_bresenham(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1

    orig_dx = x2 - x1
    orig_dy = y2 - y1

    slope_str = None
    slope_note = ""

    if orig_dx == 0:
        slope_str = "undefined (vertical line)"
    else:
        slope_val = orig_dy / orig_dx
        slope_str = f"{orig_dy}/{orig_dx} = {slope_val:.4f}"

    rows = []

    # ── Case: |slope| ≤ 1  (drive along X) ───────────────────────────────────
    if dx >= dy:
        slope_note = f"|slope| ≤ 1  →  drive along X axis"
        P = 2 * dy - dx
        cx, cy = x1, y1

        total_steps = dx
        for i in range(total_steps):
            nx = cx + sx
            if P < 0:
                ny = cy
                new_P = P + 2 * dy
            else:
                ny = cy + sy
                new_P = P + 2 * dy - 2 * dx

            rows.append({
                "i": i,
                "Pᵢ": P,
                "xᵢ": cx,
                "yᵢ": cy,
                "x(i+1)": nx,
                "y(i+1)": ny,
                "Decision": "P < 0  →  y unchanged" if P < 0 else "P ≥ 0  →  y incremented",
            })
            cx, cy, P = nx, ny, new_P

    # ── Case: |slope| > 1  (drive along Y) ───────────────────────────────────
    else:
        slope_note = f"|slope| > 1  →  drive along Y axis"
        P = 2 * dx - dy
        cx, cy = x1, y1

        total_steps = dy
        for i in range(total_steps):
            ny = cy + sy
            if P < 0:
                nx = cx
                new_P = P + 2 * dx
            else:
                nx = cx + sx
                new_P = P + 2 * dx - 2 * dy

            rows.append({
                "i": i,
                "Pᵢ": P,
                "xᵢ": cx,
                "yᵢ": cy,
                "x(i+1)": nx,
                "y(i+1)": ny,
                "Decision": "P < 0  →  x unchanged" if P < 0 else "P ≥ 0  →  x incremented",
            })
            cx, cy, P = nx, ny, new_P

    return slope_str, slope_note, rows


# ══════════════════════════════════════════════════════════════════════════════
# Render DDA
# ══════════════════════════════════════════════════════════════════════════════
if show_dda:
    st.header("DDA Algorithm")

    if x1 == x2 and y1 == y2:
        st.warning("Start and end points are the same. Nothing to draw.")
    else:
        slope_str, slope_note, dda_rows = run_dda(x1, y1, x2, y2)

        # Slope info
        st.subheader("Slope")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("m = Δy / Δx", slope_str)
        with info_col2:
            st.info(slope_note)

        # Formulas used
        dx_val = x2 - x1
        dy_val = y2 - y1
        steps_val = max(abs(dx_val), abs(dy_val))
        with st.expander("Formulas used"):
            st.markdown(f"""
| Symbol | Value |
|--------|-------|
| Δx | {dx_val} |
| Δy | {dy_val} |
| steps | max(|Δx|, |Δy|) = {steps_val} |
| x increment | Δx / steps = {dx_val}/{steps_val} = {dx_val/steps_val:.4f} |
| y increment | Δy / steps = {dy_val}/{steps_val} = {dy_val/steps_val:.4f} |
""")

        # Iteration table
        st.subheader("Iteration Table")
        df_dda = pd.DataFrame(dda_rows)
        st.dataframe(df_dda, use_container_width=True, hide_index=True)

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Render Bresenham
# ══════════════════════════════════════════════════════════════════════════════
if show_bres:
    st.header("Bresenham Algorithm")

    if x1 == x2 and y1 == y2:
        st.warning("Start and end points are the same. Nothing to draw.")
    else:
        slope_str, slope_note, bres_rows = run_bresenham(x1, y1, x2, y2)

        # Slope info
        st.subheader("Slope")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("m = Δy / Δx", slope_str)
        with info_col2:
            st.info(slope_note)

        # Initial decision parameter
        dx_abs = abs(x2 - x1)
        dy_abs = abs(y2 - y1)

        with st.expander("Initial Decision Parameter (P₀)"):
            if dx_abs >= dy_abs:
                st.markdown(f"""
**|slope| ≤ 1 case** (drive along X)

$$P_0 = 2 \\cdot \\Delta y - \\Delta x = 2 \\times {dy_abs} - {dx_abs} = {2*dy_abs - dx_abs}$$

- If $P_i < 0$: &nbsp; $P_{{i+1}} = P_i + 2\\Delta y$
- If $P_i \\geq 0$: &nbsp; $P_{{i+1}} = P_i + 2\\Delta y - 2\\Delta x$
""")
            else:
                st.markdown(f"""
**|slope| > 1 case** (drive along Y)

$$P_0 = 2 \\cdot \\Delta x - \\Delta y = 2 \\times {dx_abs} - {dy_abs} = {2*dx_abs - dy_abs}$$

- If $P_i < 0$: &nbsp; $P_{{i+1}} = P_i + 2\\Delta x$
- If $P_i \\geq 0$: &nbsp; $P_{{i+1}} = P_i + 2\\Delta x - 2\\Delta y$
""")

        # Decision table
        st.subheader("Decision Parameter Table")
        if bres_rows:
            df_bres = pd.DataFrame(bres_rows)
            # Rename columns for display
            df_bres = df_bres.rename(columns={
                "i": "i",
                "Pᵢ": "Pᵢ (decision)",
                "xᵢ": "xᵢ",
                "yᵢ": "yᵢ",
                "x(i+1)": "x(i+1)",
                "y(i+1)": "y(i+1)",
                "Decision": "Decision",
            })
            st.dataframe(df_bres, use_container_width=True, hide_index=True)
        else:
            st.info("No steps to display.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9rem;'>"
    "Developed by <a href='https://redwan-rahman.netlify.app/' target='_blank'>Redwan Rahman</a>"
    " and <a href='https://claude.ai/' target='_blank'>Claude</a>"
    "</div>",
    unsafe_allow_html=True,
)
