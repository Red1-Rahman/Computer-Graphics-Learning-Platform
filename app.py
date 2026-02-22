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
algo_col1, algo_col2, algo_col3 = st.columns(3)
show_dda  = algo_col1.checkbox("DDA Algorithm",            value=True)
show_bres = algo_col2.checkbox("Bresenham Algorithm",       value=True)
show_sym  = algo_col3.checkbox("8-Way Symmetry (Bresenham)", value=False)

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

# ══════════════════════════════════════════════════════════════════════════════
# 8-Way Symmetry helpers
# ══════════════════════════════════════════════════════════════════════════════

# Zone table: label, slope range, transform expression, condition description
ZONE_TABLE = [
    {"Zone": 0, "Slope Range": "0  to  1",   "Transform (x_plot, y_plot)": "(x₀ + xᵢ,  y₀ + yᵢ)",  "Direction": "dx ≥ 0, dy ≥ 0, dx ≥ dy"},
    {"Zone": 1, "Slope Range": "1  to  +∞",  "Transform (x_plot, y_plot)": "(x₀ + yᵢ,  y₀ + xᵢ)",  "Direction": "dx ≥ 0, dy ≥ 0, dy > dx"},
    {"Zone": 2, "Slope Range": "−∞ to −1",   "Transform (x_plot, y_plot)": "(x₀ − yᵢ,  y₀ + xᵢ)",  "Direction": "dx < 0, dy ≥ 0, dy ≥ |dx|"},
    {"Zone": 3, "Slope Range": "−1 to  0",   "Transform (x_plot, y_plot)": "(x₀ − xᵢ,  y₀ + yᵢ)",  "Direction": "dx < 0, dy ≥ 0, |dx| > dy"},
    {"Zone": 4, "Slope Range": "0  to  1",   "Transform (x_plot, y_plot)": "(x₀ − xᵢ,  y₀ − yᵢ)",  "Direction": "dx ≤ 0, dy ≤ 0, |dx| ≥ |dy|"},
    {"Zone": 5, "Slope Range": "1  to  +∞",  "Transform (x_plot, y_plot)": "(x₀ − yᵢ,  y₀ − xᵢ)",  "Direction": "dx ≤ 0, dy ≤ 0, |dy| > |dx|"},
    {"Zone": 6, "Slope Range": "−∞ to −1",   "Transform (x_plot, y_plot)": "(x₀ + yᵢ,  y₀ − xᵢ)",  "Direction": "dx ≥ 0, dy < 0, |dy| > dx"},
    {"Zone": 7, "Slope Range": "−1 to  0",   "Transform (x_plot, y_plot)": "(x₀ + xᵢ,  y₀ − yᵢ)",  "Direction": "dx ≥ 0, dy < 0, dx ≥ |dy|"},
]

def detect_zone(dx, dy):
    """Return which of the 8 zones the vector (dx, dy) belongs to."""
    if dx >= 0 and dy >= 0:
        return 0 if dx >= dy else 1
    elif dx < 0 and dy >= 0:
        return 2 if dy >= -dx else 3
    elif dx <= 0 and dy <= 0:
        return 4 if -dx >= -dy else 5
    else:  # dx > 0, dy < 0
        return 7 if dx >= -dy else 6


def to_zone0(zone, dx, dy):
    """Transform (dx, dy) into the equivalent Zone-0 deltas (always positive, |dx0| ≥ |dy0|)."""
    if zone == 0: return  dx,  dy
    if zone == 1: return  dy,  dx
    if zone == 2: return  dy, -dx
    if zone == 3: return -dx,  dy
    if zone == 4: return -dx, -dy
    if zone == 5: return -dy, -dx
    if zone == 6: return -dy,  dx
    if zone == 7: return  dx, -dy


def inv_transform(zone, x0, y0, xi, yi):
    """Map a Zone-0 incremental point (xi, yi) back to actual pixel (x, y)."""
    if zone == 0: return x0 + xi,  y0 + yi
    if zone == 1: return x0 + yi,  y0 + xi
    if zone == 2: return x0 - yi,  y0 + xi
    if zone == 3: return x0 - xi,  y0 + yi
    if zone == 4: return x0 - xi,  y0 - yi
    if zone == 5: return x0 - yi,  y0 - xi
    if zone == 6: return x0 + yi,  y0 - xi
    if zone == 7: return x0 + xi,  y0 - yi


def bresenham_zone0(dx0, dy0):
    """
    Run Bresenham in Zone 0 (dx0 >= dy0 >= 0).
    Returns list of (xi, yi) incremental points starting from (0,0).
    """
    rows = []
    xi, yi = 0, 0
    P = 2 * dy0 - dx0
    rows.append({"Step": 0, "xᵢ (z0)": xi, "yᵢ (z0)": yi})
    for _ in range(dx0):
        xi += 1
        if P < 0:
            P = P + 2 * dy0
        else:
            yi += 1
            P = P + 2 * dy0 - 2 * dx0
        rows.append({"Step": _ + 1, "xᵢ (z0)": xi, "yᵢ (z0)": yi})
    return rows


def run_8way_symmetry(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    zone = detect_zone(dx, dy)
    dx0, dy0 = to_zone0(zone, dx, dy)
    z0_points = bresenham_zone0(dx0, dy0)

    rows = []
    for p in z0_points:
        xi, yi = p["xᵢ (z0)"], p["yᵢ (z0)"]
        ax, ay = inv_transform(zone, x1, y1, xi, yi)
        rows.append({
            "Step": p["Step"],
            "xᵢ (zone 0)": xi,
            "yᵢ (zone 0)": yi,
            "x (actual)": ax,
            "y (actual)": ay,
        })
    return zone, dx0, dy0, rows


# ══════════════════════════════════════════════════════════════════════════════
# Render 8-Way Symmetry
# ══════════════════════════════════════════════════════════════════════════════
if show_sym:
    st.header("8-Way Symmetry — Bresenham")

    if x1 == x2 and y1 == y2:
        st.warning("Start and end points are the same. Nothing to draw.")
    else:
        # ── Zone reference table ───────────────────────────────────────────────
        st.subheader("Zone Reference Table")
        st.markdown(
            "Each zone maps a general line direction into **Zone 0** "
            "(slope 0–1, first octant) so a single Bresenham kernel handles all cases."
        )
        df_zones = pd.DataFrame(ZONE_TABLE)
        st.dataframe(df_zones, use_container_width=True, hide_index=True)

        st.divider()

        dx_in = x2 - x1
        dy_in = y2 - y1

        zone, dx0, dy0, sym_rows = run_8way_symmetry(x1, y1, x2, y2)

        # ── Zone detection ─────────────────────────────────────────────────────
        st.subheader("Zone Detection")
        zc1, zc2, zc3 = st.columns(3)
        zc1.metric("Δx", dx_in)
        zc2.metric("Δy", dy_in)
        zc3.metric("Detected Zone", f"Zone {zone}")

        zone_info = ZONE_TABLE[zone]
        st.info(
            f"**Zone {zone}** — Slope range: {zone_info['Slope Range']}   |   "
            f"Condition: {zone_info['Direction']}   |   "
            f"Transform: {zone_info['Transform (x_plot, y_plot)']}"
        )

        # ── Zone-0 equivalent line ─────────────────────────────────────────────
        with st.expander("Zone-0 Transformation Details"):
            st.markdown(f"""
| Property | Original | Zone-0 Equivalent |
|----------|----------|-------------------|
| Start point | ({x1}, {y1}) | (0, 0) |
| End point | ({x2}, {y2}) | ({dx0}, {dy0}) |
| Δx | {dx_in} | {dx0} |
| Δy | {dy_in} | {dy0} |
| Initial P₀ | — | 2·Δy − Δx = 2×{dy0} − {dx0} = **{2*dy0 - dx0}** |
""")

        # ── Computed points ────────────────────────────────────────────────────
        st.subheader("Computed Points (Zone 0 → Actual)")
        st.markdown(
            f"Bresenham runs in **Zone 0** on the transformed line `(0,0)→({dx0},{dy0})`, "
            f"then each point is mapped back to the original zone via the inverse transform: "
            f"**{ZONE_TABLE[zone]['Transform (x_plot, y_plot)']}**"
        )
        df_sym = pd.DataFrame(sym_rows)
        st.dataframe(df_sym, use_container_width=True, hide_index=True)

    st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9rem;'>"
    "Developed by <a href='https://redwan-rahman.netlify.app/' target='_blank'>Redwan Rahman</a>"
    " and <a href='https://claude.ai/' target='_blank'>Claude</a>"
    "</div>",
    unsafe_allow_html=True,
)
