import streamlit as st
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

st.set_page_config(page_title="Computer Graphics Algorithms", layout="wide")
st.title("Computer Graphics Algorithms")


# ══════════════════════════════════════════════════════════════════════════════
# Pixel Grid Visualizer
# ══════════════════════════════════════════════════════════════════════════════

def draw_pixel_grid(pixels, x1, y1, x2, y2, title="", ideal_line=True):
    """
    Draw a pixel-grid visualization.
    pixels : list of (x, y) integer tuples (the lit pixels)
    """
    if not pixels:
        return None

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    pad = 1
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    # Cell size in figure units; scale figure to grid size
    cols = xmax - xmin + 1
    rows_g = ymax - ymin + 1
    cell = 0.42
    fig_w = max(3.5, min(cols * cell + 1.2, 7))
    fig_h = max(2.5, min(rows_g * cell + 1.2, 6))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # ── grid lines ────────────────────────────────────────────────────────────
    for gx in range(xmin, xmax + 2):
        ax.axvline(gx - 0.5, color="#2a2d35", linewidth=0.6, zorder=1)
    for gy in range(ymin, ymax + 2):
        ax.axhline(gy - 0.5, color="#2a2d35", linewidth=0.6, zorder=1)

    # ── ideal continuous line ─────────────────────────────────────────────────
    if ideal_line:
        ax.plot([x1, x2], [y1, y2],
                color="#ffffff", linewidth=1, linestyle="--",
                alpha=0.35, zorder=2, label="Ideal line")

    # ── filled pixels ─────────────────────────────────────────────────────────
    pixel_set = set(pixels)
    for idx, (px, py) in enumerate(pixels):
        is_start = (px == x1 and py == y1)
        is_end   = (px == x2 and py == y2)
        if is_start:
            facecolor, edgecolor, label = "#2ecc71", "#27ae60", "Start"
        elif is_end:
            facecolor, edgecolor, label = "#e74c3c", "#c0392b", "End"
        else:
            facecolor, edgecolor, label = "#3498db", "#2980b9", None

        rect = mpatches.FancyBboxPatch(
            (px - 0.46, py - 0.46), 0.92, 0.92,
            boxstyle="round,pad=0.04",
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=1.2, zorder=3
        )
        ax.add_patch(rect)

        # coordinate label inside cell
        ax.text(px, py - 0.02, f"({px},{py})",
                ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold",
                zorder=4,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

        # step index (small, top-right of cell)
        ax.text(px + 0.35, py + 0.34, str(idx),
                ha="right", va="top",
                fontsize=5.5, color="#ecf0f1", alpha=0.8, zorder=4)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", edgecolor="#27ae60", label=f"Start ({x1},{y1})"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="#c0392b", label=f"End ({x2},{y2})"),
        mpatches.Patch(facecolor="#3498db", edgecolor="#2980b9", label="Pixel"),
    ]
    if ideal_line:
        legend_elements.append(
            plt.Line2D([0], [0], color="white", linewidth=1,
                       linestyle="--", alpha=0.5, label="Ideal line")
        )
    ax.legend(handles=legend_elements, loc="upper left",
              fontsize=7, framealpha=0.3,
              labelcolor="white", facecolor="#1a1d23")

    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("y", color="#aaaaaa", fontsize=9)
    ax.set_title(title, color="#dddddd", fontsize=10, pad=8)
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # integer ticks only
    ax.set_xticks(range(xmin, xmax + 1))
    ax.set_yticks(range(ymin, ymax + 1))

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Circle Grid Visualizer
# ══════════════════════════════════════════════════════════════════════════════

def draw_circle_grid(pixels, cx, cy, r, title=""):
    if not pixels:
        return None
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    pad = 1
    xmin = min(xs + [cx]) - pad
    xmax = max(xs + [cx]) + pad
    ymin = min(ys + [cy]) - pad
    ymax = max(ys + [cy]) + pad
    cols_n = xmax - xmin + 1
    rows_n = ymax - ymin + 1
    cell   = 0.42
    fig_w  = max(4, min(cols_n * cell + 1.2, 8))
    fig_h  = max(4, min(rows_n * cell + 1.2, 8))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for gx in range(xmin, xmax + 2):
        ax.axvline(gx - 0.5, color="#2a2d35", linewidth=0.6, zorder=1)
    for gy in range(ymin, ymax + 2):
        ax.axhline(gy - 0.5, color="#2a2d35", linewidth=0.6, zorder=1)
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
            color="#ffffff", linewidth=1, linestyle="--", alpha=0.3,
            zorder=2, label="Ideal circle")
    ax.plot(cx, cy, marker="+", color="#f1c40f", markersize=8,
            markeredgewidth=1.5, zorder=5)
    for idx, (px, py) in enumerate(pixels):
        rect = mpatches.FancyBboxPatch(
            (px - 0.46, py - 0.46), 0.92, 0.92,
            boxstyle="round,pad=0.04",
            facecolor="#9b59b6", edgecolor="#8e44ad", linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(px, py - 0.02, f"({px},{py})", ha="center", va="center",
                fontsize=5.8, color="white", fontweight="bold", zorder=4,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
        ax.text(px + 0.35, py + 0.34, str(idx + 1), ha="right", va="top",
                fontsize=5, color="#ecf0f1", alpha=0.8, zorder=4)
    legend_elements = [
        mpatches.Patch(facecolor="#9b59b6", edgecolor="#8e44ad", label="Pixel"),
        plt.Line2D([0], [0], color="#f1c40f", marker="+", markersize=8,
                   linewidth=0, label=f"Center ({cx},{cy})"),
        plt.Line2D([0], [0], color="white", linewidth=1,
                   linestyle="--", alpha=0.5, label="Ideal circle"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              framealpha=0.3, labelcolor="white", facecolor="#1a1d23")
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("y", color="#aaaaaa", fontsize=9)
    ax.set_title(title, color="#dddddd", fontsize=10, pad=8)
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_xticks(range(xmin, xmax + 1))
    ax.set_yticks(range(ymin, ymax + 1))
    plt.tight_layout()
    return fig

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
# Shared UI helper
# ══════════════════════════════════════════════════════════════════════════════

def section_header(title, en_url, bn_url):
    btn_base = (
        "display:inline-flex;align-items:center;gap:6px;"
        "padding:5px 12px;border-radius:6px;font-size:0.78rem;"
        "font-weight:600;text-decoration:none;margin-left:8px;"
    )
    active_style   = btn_base + "background:#ff0000;color:#ffffff;"
    disabled_style = btn_base + "background:#444;color:#888;cursor:not-allowed;pointer-events:none;"
    yt_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" height="13" width="13" '
        'viewBox="0 0 24 24" fill="currentColor" style="flex-shrink:0">'
        '<path d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.5 3.5 12 3.5 12 3.5s-7.5 0-9.4.6'
        'A3 3 0 0 0 .5 6.2 31.3 31.3 0 0 0 0 12a31.3 31.3 0 0 0 .5 5.8 3 3 0 0 0 2.1 2.1'
        'c1.9.6 9.4.6 9.4.6s7.5 0 9.4-.6a3 3 0 0 0 2.1-2.1A31.3 31.3 0 0 0 24 12'
        'a31.3 31.3 0 0 0-.5-5.8zM9.7 15.5V8.5l6.3 3.5-6.3 3.5z"/>'
        '</svg>'
    )
    def make_btn(url, label):
        if url and url.lower() not in ("tba", "to be added"):
            return f'<a href="{url}" target="_blank" style="{active_style}">{yt_icon} {label}</a>'
        return f'<span style="{disabled_style}">{yt_icon} {label} — Soon</span>'
    html = (
        "<div style='display:flex;align-items:center;justify-content:space-between;"
        "margin-bottom:0.6rem'>"
        f"<h2 style='margin:0;padding:0'>{title}</h2>"
        f"<div style='display:flex;align-items:center'>"
        f"{make_btn(en_url, 'English')}{make_btn(bn_url, 'Bangla')}"
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 8-Way Symmetry helpers
# ══════════════════════════════════════════════════════════════════════════════

ZONE_TABLE = [
    {"Zone": 0, "Slope Range": "0  to  1",  "Transform (x_plot, y_plot)": "(x₀ + xᵢ,  y₀ + yᵢ)", "Direction": "dx ≥ 0, dy ≥ 0, dx ≥ dy"},
    {"Zone": 1, "Slope Range": "1  to  +∞", "Transform (x_plot, y_plot)": "(x₀ + yᵢ,  y₀ + xᵢ)", "Direction": "dx ≥ 0, dy ≥ 0, dy > dx"},
    {"Zone": 2, "Slope Range": "−∞ to −1",  "Transform (x_plot, y_plot)": "(x₀ − yᵢ,  y₀ + xᵢ)", "Direction": "dx < 0, dy ≥ 0, dy ≥ |dx|"},
    {"Zone": 3, "Slope Range": "−1 to  0",  "Transform (x_plot, y_plot)": "(x₀ − xᵢ,  y₀ + yᵢ)", "Direction": "dx < 0, dy ≥ 0, |dx| > dy"},
    {"Zone": 4, "Slope Range": "0  to  1",  "Transform (x_plot, y_plot)": "(x₀ − xᵢ,  y₀ − yᵢ)", "Direction": "dx ≤ 0, dy ≤ 0, |dx| ≥ |dy|"},
    {"Zone": 5, "Slope Range": "1  to  +∞", "Transform (x_plot, y_plot)": "(x₀ − yᵢ,  y₀ − xᵢ)", "Direction": "dx ≤ 0, dy ≤ 0, |dy| > |dx|"},
    {"Zone": 6, "Slope Range": "−∞ to −1",  "Transform (x_plot, y_plot)": "(x₀ + yᵢ,  y₀ − xᵢ)", "Direction": "dx ≥ 0, dy < 0, |dy| > dx"},
    {"Zone": 7, "Slope Range": "−1 to  0",  "Transform (x_plot, y_plot)": "(x₀ + xᵢ,  y₀ − yᵢ)", "Direction": "dx ≥ 0, dy < 0, dx ≥ |dy|"},
]

def detect_zone(dx, dy):
    if dx >= 0 and dy >= 0: return 0 if dx >= dy else 1
    elif dx < 0 and dy >= 0: return 2 if dy >= -dx else 3
    elif dx <= 0 and dy <= 0: return 4 if -dx >= -dy else 5
    else: return 7 if dx >= -dy else 6

def to_zone0(zone, dx, dy):
    if zone == 0: return  dx,  dy
    if zone == 1: return  dy,  dx
    if zone == 2: return  dy, -dx
    if zone == 3: return -dx,  dy
    if zone == 4: return -dx, -dy
    if zone == 5: return -dy, -dx
    if zone == 6: return -dy,  dx
    if zone == 7: return  dx, -dy

def inv_transform(zone, x0, y0, xi, yi):
    if zone == 0: return x0 + xi, y0 + yi
    if zone == 1: return x0 + yi, y0 + xi
    if zone == 2: return x0 - yi, y0 + xi
    if zone == 3: return x0 - xi, y0 + yi
    if zone == 4: return x0 - xi, y0 - yi
    if zone == 5: return x0 - yi, y0 - xi
    if zone == 6: return x0 + yi, y0 - xi
    if zone == 7: return x0 + xi, y0 - yi

def bresenham_zone0(dx0, dy0):
    rows = []
    xi, yi = 0, 0
    P = 2 * dy0 - dx0
    rows.append({"Step": 0, "xᵢ (z0)": xi, "yᵢ (z0)": yi})
    for k in range(dx0):
        xi += 1
        if P < 0:
            P = P + 2 * dy0
        else:
            yi += 1
            P = P + 2 * dy0 - 2 * dx0
        rows.append({"Step": k + 1, "xᵢ (z0)": xi, "yᵢ (z0)": yi})
    return rows

def run_8way_symmetry(x1, y1, x2, y2):
    dx = x2 - x1; dy = y2 - y1
    zone = detect_zone(dx, dy)
    dx0, dy0 = to_zone0(zone, dx, dy)
    z0_pts = bresenham_zone0(dx0, dy0)
    rows = []
    for p in z0_pts:
        xi, yi = p["xᵢ (z0)"], p["yᵢ (z0)"]
        ax, ay = inv_transform(zone, x1, y1, xi, yi)
        rows.append({"Step": p["Step"], "xᵢ (zone 0)": xi, "yᵢ (zone 0)": yi,
                     "x (actual)": ax, "y (actual)": ay})
    return zone, dx0, dy0, rows


# ══════════════════════════════════════════════════════════════════════════════
# Midpoint Circle Algorithm
# ══════════════════════════════════════════════════════════════════════════════

def eight_points(cx, cy, x, y):
    pts = set()
    for sx, sy in [(x,y),(-x,y),(x,-y),(-x,-y),(y,x),(-y,x),(y,-x),(-y,-x)]:
        pts.add((cx + sx, cy + sy))
    return sorted(pts)

def run_midpoint_circle(cx, cy, r):
    r_float = float(r)
    is_int  = isinstance(r, int) or (isinstance(r, float) and r.is_integer())
    if is_int:
        P = 1 - int(r_float)
        p0_str = f"1 − r = 1 − {int(r_float)} = {P}"
    else:
        P = round(5/4 - r_float, 6)
        p0_str = f"5/4 − r = 1.25 − {r_float} = {P}"

    x, y = 0, int(round(r_float))
    rows = []
    all_pixels = []

    while x <= y:
        pts = eight_points(cx, cy, x, y)
        all_pixels.extend(pts)
        x_old, y_old, P_old = x, y, P
        x += 1
        if P_old < 0:
            y_new = y_old
            new_P = P_old + 2 * x_old + 3
            dec   = "P < 0  →  y unchanged,  P+1 = P + 2xₖ + 3"
        else:
            y    -= 1
            y_new = y
            new_P = P_old + 2 * x_old + 5 - 2 * y_old
            dec   = "P ≥ 0  →  y decremented,  P+1 = P + 2xₖ + 5 − 2yₖ"
        pts_str = ", ".join(f"({px},{py})" for px, py in pts)
        rows.append({"k": len(rows), "Pₖ": P_old, "xₖ": x_old, "yₖ": y_old,
                     "x(k+1)": x, "y(k+1)": y_new, "Pₖ₊₁": new_P,
                     "Decision": dec, "8 pixels": pts_str})
        P = new_P

    seen = set(); unique_pixels = []
    for pt in all_pixels:
        if pt not in seen:
            seen.add(pt); unique_pixels.append(pt)
    return p0_str, rows, unique_pixels


# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

tab_line, tab_circle, tab_2d, tab_3d = st.tabs([
    "╱  Line Drawing",
    "◯  Circle Drawing",
    "⊡  2D Transformation",
    "⬢  3D Transformation",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Line Drawing
# ─────────────────────────────────────────────────────────────────────────────
with tab_line:
    st.header("Input Points")
    col1, col2, col3, col4 = st.columns(4)
    with col1: x1 = st.number_input("x₁ (start)", value=1, step=1, key="lx1")
    with col2: y1 = st.number_input("y₁ (start)", value=1, step=1, key="ly1")
    with col3: x2 = st.number_input("x₂ (end)",   value=2, step=1, key="lx2")
    with col4: y2 = st.number_input("y₂ (end)",   value=7, step=1, key="ly2")
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    st.divider()

    algo_col1, algo_col2, algo_col3 = st.columns(3)
    show_dda  = algo_col1.checkbox("DDA Algorithm",             value=True)
    show_bres = algo_col2.checkbox("Bresenham Algorithm",        value=True)
    show_sym  = algo_col3.checkbox("8-Way Symmetry (Bresenham)", value=False)

    st.divider()

    # ── DDA ──────────────────────────────────────────────────────────────────
    if show_dda:
        section_header("DDA Algorithm",
                        "https://youtu.be/W5P8GlaEOSI?si=iOWG155vbS8MGnFp",
                        "https://youtu.be/0eQv0MQBu7Q?si=D-__o4SUTA5_T9BQ")
        if x1 == x2 and y1 == y2:
            st.warning("Start and end points are the same.")
        else:
            slope_str, slope_note, dda_rows = run_dda(x1, y1, x2, y2)
            st.subheader("Slope")
            sc1, sc2 = st.columns(2)
            with sc1: st.metric("m = Δy / Δx", slope_str)
            with sc2: st.info(slope_note)
            dx_val = x2 - x1; dy_val = y2 - y1
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
            st.subheader("Iteration Table")
            st.dataframe(pd.DataFrame(dda_rows), width='stretch', hide_index=True)
        st.divider()

    # ── Bresenham ─────────────────────────────────────────────────────────────
    if show_bres:
        section_header("Bresenham Algorithm",
                        "https://youtu.be/RGB-wlatStc?si=h5-m7di5ixKXk8KA",
                        "https://youtu.be/BY_iG7CZBf8?si=3Or-mSdWSHo8ATwr")
        if x1 == x2 and y1 == y2:
            st.warning("Start and end points are the same.")
        else:
            slope_str, slope_note, bres_rows = run_bresenham(x1, y1, x2, y2)
            st.subheader("Slope")
            bc1, bc2 = st.columns(2)
            with bc1: st.metric("m = Δy / Δx", slope_str)
            with bc2: st.info(slope_note)
            dx_abs = abs(x2 - x1); dy_abs = abs(y2 - y1)
            with st.expander("Initial Decision Parameter (P₀)"):
                if dx_abs >= dy_abs:
                    st.markdown(f"""
**|slope| ≤ 1 case** (drive along X)

$$P_0 = 2\\Delta y - \\Delta x = 2\\times{dy_abs} - {dx_abs} = {2*dy_abs - dx_abs}$$

- $P_i < 0$: $P_{{i+1}} = P_i + 2\\Delta y$
- $P_i \\geq 0$: $P_{{i+1}} = P_i + 2\\Delta y - 2\\Delta x$
""")
                else:
                    st.markdown(f"""
**|slope| > 1 case** (drive along Y)

$$P_0 = 2\\Delta x - \\Delta y = 2\\times{dx_abs} - {dy_abs} = {2*dx_abs - dy_abs}$$

- $P_i < 0$: $P_{{i+1}} = P_i + 2\\Delta x$
- $P_i \\geq 0$: $P_{{i+1}} = P_i + 2\\Delta x - 2\\Delta y$
""")
            st.subheader("Decision Parameter Table")
            if bres_rows:
                df_b = pd.DataFrame(bres_rows).rename(columns={"Pᵢ": "Pᵢ (decision)"})
                st.dataframe(df_b, width='stretch', hide_index=True)
            else:
                st.info("No steps to display.")
        st.divider()

    # ── 8-Way Symmetry ────────────────────────────────────────────────────────
    if show_sym:
        section_header("8-Way Symmetry — Bresenham",
                        "tba",
                        "https://youtu.be/x0Mto5Sp9Dc?si=iUObnGfJVx5DSByF")
        if x1 == x2 and y1 == y2:
            st.warning("Start and end points are the same.")
        else:
            st.subheader("Zone Reference Table")
            st.markdown("Each zone maps into **Zone 0** so one Bresenham kernel handles all directions.")
            st.dataframe(pd.DataFrame(ZONE_TABLE), width='stretch', hide_index=True)
            st.divider()
            dx_in = x2 - x1; dy_in = y2 - y1
            zone, dx0, dy0, sym_rows = run_8way_symmetry(x1, y1, x2, y2)
            st.subheader("Zone Detection")
            zc1, zc2, zc3 = st.columns(3)
            zc1.metric("Δx", dx_in); zc2.metric("Δy", dy_in)
            zc3.metric("Detected Zone", f"Zone {zone}")
            zi = ZONE_TABLE[zone]
            st.info(f"**Zone {zone}** — {zi['Slope Range']} | {zi['Direction']} | {zi['Transform (x_plot, y_plot)']}")
            with st.expander("Zone-0 Transformation Details"):
                st.markdown(f"""
| Property | Original | Zone-0 Equivalent |
|----------|----------|-------------------|
| Start | ({x1},{y1}) | (0,0) |
| End | ({x2},{y2}) | ({dx0},{dy0}) |
| Δx | {dx_in} | {dx0} |
| Δy | {dy_in} | {dy0} |
| P₀ | — | 2·Δy − Δx = **{2*dy0 - dx0}** |
""")
            st.subheader("Computed Points (Zone 0 → Actual)")
            st.dataframe(pd.DataFrame(sym_rows), width='stretch', hide_index=True)
        st.divider()

    # ── Combined grid ─────────────────────────────────────────────────────────
    if (show_dda or show_bres or show_sym) and not (x1 == x2 and y1 == y2):
        st.header("Grid Visualization")
        grid_items = []
        if show_dda:
            _, _, dda_rows_g = run_dda(x1, y1, x2, y2)
            grid_items.append(("DDA", [(r["x (rounded)"], r["y (rounded)"]) for r in dda_rows_g]))
        if show_bres:
            _, _, br_g = run_bresenham(x1, y1, x2, y2)
            grid_items.append(("Bresenham",
                                [(br_g[0]["xᵢ"], br_g[0]["yᵢ"])] +
                                [(r["x(i+1)"], r["y(i+1)"]) for r in br_g]))
        if show_sym:
            zone_g, _, _, sr_g = run_8way_symmetry(x1, y1, x2, y2)
            grid_items.append((f"8-Way Sym. Zone {zone_g}",
                                [(r["x (actual)"], r["y (actual)"]) for r in sr_g]))
        if grid_items:
            gcols = st.columns(len(grid_items))
            for col, (title, pixels) in zip(gcols, grid_items):
                with col:
                    st.markdown(f"**{title}**")
                    fig = draw_pixel_grid(pixels, x1, y1, x2, y2,
                                          title=f"({x1},{y1}) → ({x2},{y2})")
                    if fig:
                        st.pyplot(fig, width='stretch')
                        plt.close(fig)
        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Circle Drawing
# ─────────────────────────────────────────────────────────────────────────────
with tab_circle:
    section_header("Midpoint Circle Algorithm", "tba", "https://youtu.be/9JprrOeScvU?si=EWIdEU-sJaRqpDvD")

    st.markdown(
        "<p><span style='color: red; font-weight: bold;'>Prerequisite:</span> This algorithm uses "
        "<span style='color: #3498db; font-weight: bold;'>8-way symmetry</span> "
        "to efficiently compute all 8 octant pixels from one set of coordinates.</p>",
        unsafe_allow_html=True
    )

    st.header("Input")
    cc1, cc2, cc3 = st.columns(3)
    with cc1: cx = st.number_input("Center x", value=0, step=1, key="cx")
    with cc2: cy = st.number_input("Center y", value=0, step=1, key="cy")
    with cc3: r  = st.number_input("Radius r", value=5.0, min_value=1.0, step=0.5, format="%.2f", key="cr")
    cx, cy = int(cx), int(cy)
    r = int(r) if float(r) == int(r) else float(r)

    st.divider()

    p0_str, circle_rows, circle_pixels = run_midpoint_circle(cx, cy, r)

    # ── Initial decision parameter ───────────────────────────────────────────
    st.subheader("Initial Decision Parameter (P₀)")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.metric("P₀", p0_str)
    with pc2:
        r_is_int = isinstance(r, int) or (isinstance(r, float) and float(r).is_integer())
        if r_is_int:
            st.info("**r is integer** → $P_0 = 1 - r$")
        else:
            st.info("**r is non-integer** → $P_0 = \\frac{5}{4} - r$")

    with st.expander("Recurrence formulas"):
        st.markdown(r"""
Starting point: $(x_0, y_0) = (0,\ r)$

At each step $x_k \to x_{k+1} = x_k + 1$:

| Condition | $y_{k+1}$ | $P_{k+1}$ |
|-----------|-----------|-----------|
| $P_k < 0$ | $y_k$ (unchanged) | $P_k + 2x_k + 3$ |
| $P_k \geq 0$ | $y_k - 1$ (decremented) | $P_k + 2x_k + 5 - 2y_k$ |

Loop continues while $x \leq y$.  
8-way symmetry generates all 8 octant pixels from each $(x_k, y_k)$.
""")

    # ── Decision table ───────────────────────────────────────────────────────
    st.subheader("Decision Parameter Table")
    st.dataframe(pd.DataFrame(circle_rows), width='stretch', hide_index=True)

    st.divider()

    # ── Grid ─────────────────────────────────────────────────────────────────
    st.subheader("Grid Visualization")
    fig_c = draw_circle_grid(circle_pixels, cx, cy, float(r),
                             title=f"Midpoint Circle  center=({cx},{cy})  r={r}")
    if fig_c:
        st.pyplot(fig_c, width='stretch')
        plt.close(fig_c)

    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — 2D Transformation
# ─────────────────────────────────────────────────────────────────────────────
with tab_2d:
    st.markdown(
        "<div style='text-align:center;padding:4rem 0 2rem'>"
        "<div style='font-size:4rem;margin-bottom:1rem'>⊡</div>"
        "<h2 style='margin:0'>2D Transformation</h2>"
        "<p style='color:gray;margin-top:0.5rem'>Translation · Rotation · Scaling · Shearing · Reflection</p>"
        "<span style='display:inline-block;margin-top:1.5rem;padding:6px 20px;"
        "border-radius:20px;background:#2a2a2a;color:#888;font-size:0.85rem'>Coming soon</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — 3D Transformation
# ─────────────────────────────────────────────────────────────────────────────
CUBE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 72 72"
     fill="none" stroke="#aaaaaa" stroke-width="2" stroke-linejoin="round">
  <!-- front face -->
  <polygon points="36,42 14,30 14,12 36,24" fill="#1a1d23" stroke="#aaaaaa"/>
  <polygon points="36,42 58,30 58,12 36,24" fill="#141618" stroke="#aaaaaa"/>
  <polygon points="36,6  14,12 36,24 58,12"  fill="#22262e" stroke="#aaaaaa"/>
  <!-- 12 edges -->
  <!-- bottom face edges (hidden but drawn for wireframe) -->
  <line x1="14" y1="30" x2="36" y2="42"/>
  <line x1="58" y1="30" x2="36" y2="42"/>
  <line x1="14" y1="30" x2="36" y2="18"/>
  <line x1="58" y1="30" x2="36" y2="18"/>
</svg>
"""

with tab_3d:
    st.markdown(
        f"<div style='text-align:center;padding:4rem 0 2rem'>"
        "<div style='font-size:4rem;margin-bottom:1rem'>⬢</div>"
        "<h2 style='margin:0'>3D Transformation</h2>"
        "<p style='color:gray;margin-top:0.5rem'>Translation · Rotation · Scaling · Projection · Homogeneous Coordinates</p>"
        "<span style='display:inline-block;margin-top:1.5rem;padding:6px 20px;"
        "border-radius:20px;background:#2a2a2a;color:#888;font-size:0.85rem'>Coming soon</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.9rem'>"
    "Developed by <a href='https://redwan-rahman.netlify.app/' target='_blank'>Redwan Rahman</a>"
    " and <a href='https://claude.ai/' target='_blank'>Claude</a>"
    "</div>",
    unsafe_allow_html=True,
)
