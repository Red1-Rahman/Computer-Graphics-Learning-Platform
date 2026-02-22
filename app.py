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
# 2D Rotation
# ══════════════════════════════════════════════════════════════════════════════

POINT_LABELS = list("ABCDEFGHIJ")


def run_2d_rotation(points, theta_deg, clockwise=False):
    """Rotate a list of (x,y) points by theta_deg degrees."""
    theta = math.radians(theta_deg)
    if clockwise:
        theta = -theta
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    if clockwise:
        xp_header = "x' = x·cosθ + y·sinθ"
        yp_header = "y' = −x·sinθ + y·cosθ"
    else:
        xp_header = "x' = x·cosθ − y·sinθ"
        yp_header = "y' = x·sinθ + y·cosθ"

    rows = []
    new_points = []
    for i, (x, y) in enumerate(points):
        xp = x * cos_t - y * sin_t
        yp = x * sin_t + y * cos_t
        rows.append({
            "Point": POINT_LABELS[i],
            "x": x,
            "y": y,
            "cos θ": round(cos_t, 6),
            "sin θ": round(sin_t, 6),
            xp_header: round(xp, 4),
            yp_header: round(yp, 4),
        })
        new_points.append((round(xp, 4), round(yp, 4)))
    return rows, new_points


def draw_2d_rotation(points, new_points, theta_deg, clockwise=False):
    """Visualise original and rotated points on a dark-themed plot."""
    if not points:
        return None
    all_x = [p[0] for p in points] + [p[0] for p in new_points]
    all_y = [p[1] for p in points] + [p[1] for p in new_points]
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 4)
    pad  = span * 0.30
    cx_c = (max(all_x) + min(all_x)) / 2
    cy_c = (max(all_y) + min(all_y)) / 2
    half = span / 2 + pad
    xmin, xmax = cx_c - half, cx_c + half
    ymin, ymax = cy_c - half, cy_c + half

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Axes
    ax.axhline(0, color="#555", linewidth=0.9, zorder=1)
    ax.axvline(0, color="#555", linewidth=0.9, zorder=1)
    ax.grid(True, color="#1e2030", linewidth=0.6, zorder=0)

    # Connect original polygon
    if len(points) > 1:
        ox = [p[0] for p in points] + [points[0][0]]
        oy = [p[1] for p in points] + [points[0][1]]
        ax.plot(ox, oy, color="#3498db", linewidth=1.4,
                alpha=0.55, linestyle="--", zorder=2)
        nx_ = [p[0] for p in new_points] + [new_points[0][0]]
        ny_ = [p[1] for p in new_points] + [new_points[0][1]]
        ax.plot(nx_, ny_, color="#2ecc71", linewidth=1.4,
                alpha=0.55, linestyle="--", zorder=2)

    # Rotation arrows
    for (ox2, oy2), (nx2, ny2) in zip(points, new_points):
        ax.annotate("",
            xy=(nx2, ny2), xytext=(ox2, oy2),
            arrowprops=dict(arrowstyle="->", color="#f39c12",
                            lw=1.1, alpha=0.55),
            zorder=3)

    # Original points
    for i, (px, py) in enumerate(points):
        ax.scatter(px, py, color="#3498db", s=80, zorder=5,
                   edgecolors="#2980b9", linewidths=1.2)
        ax.text(px + 0.06 * span, py + 0.06 * span,
                f"{POINT_LABELS[i]}({px}, {py})",
                color="#7fb3d3", fontsize=8, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8,
                                            foreground="#0e1117")])

    # Rotated points
    for i, (px, py) in enumerate(new_points):
        ax.scatter(px, py, color="#2ecc71", s=80, zorder=5,
                   edgecolors="#27ae60", linewidths=1.2)
        ax.text(px + 0.06 * span, py + 0.06 * span,
                f"{POINT_LABELS[i]}'({px}, {py})",
                color="#82e0aa", fontsize=8, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8,
                                            foreground="#0e1117")])

    direction_str = "CW" if clockwise else "CCW"
    legend_elements = [
        mpatches.Patch(facecolor="#3498db", edgecolor="#2980b9",
                       label="Original points"),
        mpatches.Patch(facecolor="#2ecc71", edgecolor="#27ae60",
                       label=f"Rotated points ({direction_str} {theta_deg}\u00b0)"),
        plt.Line2D([0], [0], color="#f39c12", linewidth=1.5,
                   marker=">", markersize=6, label="Rotation path"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              framealpha=0.3, labelcolor="white", facecolor="#1a1d23")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("y", color="#aaaaaa", fontsize=9)
    direction_label = "Clockwise" if clockwise else "Counter-Clockwise"
    ax.set_title(f"2D Rotation  \u2014  \u03b8 = {theta_deg}\u00b0  ({direction_label})",
                 color="#dddddd", fontsize=10, pad=8)
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2D Translation
# ══════════════════════════════════════════════════════════════════════════════

def run_2d_translation(points, tx, ty):
    """Translate a list of (x, y) points by (tx, ty)."""
    rows = []
    new_points = []
    for i, (x, y) in enumerate(points):
        xp = x + tx
        yp = y + ty
        rows.append({
            "Point": POINT_LABELS[i],
            "x": x,
            "y": y,
            "Tx": tx,
            "Ty": ty,
            "x' = x + Tx": round(xp, 4),
            "y' = y + Ty": round(yp, 4),
        })
        new_points.append((round(xp, 4), round(yp, 4)))
    return rows, new_points


def run_2d_translation_circle(cx, cy, r, tx, ty):
    """Translate a circle centre by (tx, ty); radius is unchanged."""
    new_cx = round(cx + tx, 4)
    new_cy = round(cy + ty, 4)
    rows = [
        {"Attribute": "Center x",
         "Original": cx,
         "Formula": f"x' = x + Tx = {cx} + ({tx})",
         "New": new_cx},
        {"Attribute": "Center y",
         "Original": cy,
         "Formula": f"y' = y + Ty = {cy} + ({ty})",
         "New": new_cy},
        {"Attribute": "Radius r",
         "Original": r,
         "Formula": "r  (unchanged)",
         "New": r},
    ]
    return rows, new_cx, new_cy


def draw_2d_translation(points, new_points, tx, ty,
                        is_circle=False, radius=None):
    """Visualise original and translated shape on a dark-themed plot."""
    if not points:
        return None
    all_x = [p[0] for p in points] + [p[0] for p in new_points]
    all_y = [p[1] for p in points] + [p[1] for p in new_points]
    if is_circle and radius is not None:
        all_x += [points[0][0] - radius, points[0][0] + radius,
                  new_points[0][0] - radius, new_points[0][0] + radius]
        all_y += [points[0][1] - radius, points[0][1] + radius,
                  new_points[0][1] - radius, new_points[0][1] + radius]
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 4)
    pad  = span * 0.35
    cx_c = (max(all_x) + min(all_x)) / 2
    cy_c = (max(all_y) + min(all_y)) / 2
    half = span / 2 + pad
    xmin_p, xmax_p = cx_c - half, cx_c + half
    ymin_p, ymax_p = cy_c - half, cy_c + half

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.axhline(0, color="#555", linewidth=0.9, zorder=1)
    ax.axvline(0, color="#555", linewidth=0.9, zorder=1)
    ax.grid(True, color="#1e2030", linewidth=0.6, zorder=0)

    if is_circle and radius is not None:
        theta_arr = np.linspace(0, 2 * np.pi, 360)
        ox_c, oy_c   = points[0]
        nx_c, ny_c   = new_points[0]
        # Original circle
        ax.plot(ox_c + radius * np.cos(theta_arr),
                oy_c + radius * np.sin(theta_arr),
                color="#3498db", linewidth=1.8, alpha=0.7,
                linestyle="--", zorder=2)
        # Translated circle
        ax.plot(nx_c + radius * np.cos(theta_arr),
                ny_c + radius * np.sin(theta_arr),
                color="#2ecc71", linewidth=1.8, alpha=0.7,
                linestyle="--", zorder=2)
        # Centre points
        ax.scatter(ox_c, oy_c, color="#3498db", s=80, zorder=5,
                   edgecolors="#2980b9", linewidths=1.2)
        ax.scatter(nx_c, ny_c, color="#2ecc71", s=80, zorder=5,
                   edgecolors="#27ae60", linewidths=1.2)
        ax.text(ox_c + 0.06 * span, oy_c + 0.06 * span,
                f"C({ox_c}, {oy_c})",
                color="#7fb3d3", fontsize=8, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8,
                                            foreground="#0e1117")])
        ax.text(nx_c + 0.06 * span, ny_c + 0.06 * span,
                f"C'({nx_c}, {ny_c})",
                color="#82e0aa", fontsize=8, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8,
                                            foreground="#0e1117")])
        # Arrow from old centre to new centre
        ax.annotate("", xy=(nx_c, ny_c), xytext=(ox_c, oy_c),
                    arrowprops=dict(arrowstyle="->", color="#f39c12",
                                   lw=1.5, alpha=0.8),
                    zorder=3)
    else:
        if len(points) > 1:
            ox_poly = [p[0] for p in points] + [points[0][0]]
            oy_poly = [p[1] for p in points] + [points[0][1]]
            ax.plot(ox_poly, oy_poly, color="#3498db", linewidth=1.4,
                    alpha=0.55, linestyle="--", zorder=2)
            nx_poly = [p[0] for p in new_points] + [new_points[0][0]]
            ny_poly = [p[1] for p in new_points] + [new_points[0][1]]
            ax.plot(nx_poly, ny_poly, color="#2ecc71", linewidth=1.4,
                    alpha=0.55, linestyle="--", zorder=2)
        for (ox2, oy2), (nx2, ny2) in zip(points, new_points):
            ax.annotate("", xy=(nx2, ny2), xytext=(ox2, oy2),
                        arrowprops=dict(arrowstyle="->", color="#f39c12",
                                        lw=1.1, alpha=0.55),
                        zorder=3)
        for i, (px, py) in enumerate(points):
            ax.scatter(px, py, color="#3498db", s=80, zorder=5,
                       edgecolors="#2980b9", linewidths=1.2)
            ax.text(px + 0.06 * span, py + 0.06 * span,
                    f"{POINT_LABELS[i]}({px}, {py})",
                    color="#7fb3d3", fontsize=8, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.8,
                                                foreground="#0e1117")])
        for i, (px, py) in enumerate(new_points):
            ax.scatter(px, py, color="#2ecc71", s=80, zorder=5,
                       edgecolors="#27ae60", linewidths=1.2)
            ax.text(px + 0.06 * span, py + 0.06 * span,
                    f"{POINT_LABELS[i]}'({px}, {py})",
                    color="#82e0aa", fontsize=8, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.8,
                                                foreground="#0e1117")])

    legend_elements = [
        mpatches.Patch(facecolor="#3498db", edgecolor="#2980b9",
                       label="Original"),
        mpatches.Patch(facecolor="#2ecc71", edgecolor="#27ae60",
                       label=f"Translated  (Tx={tx}, Ty={ty})"),
        plt.Line2D([0], [0], color="#f39c12", linewidth=1.5,
                   marker=">", markersize=6, label="Translation direction"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              framealpha=0.3, labelcolor="white", facecolor="#1a1d23")
    ax.set_xlim(xmin_p, xmax_p)
    ax.set_ylim(ymin_p, ymax_p)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("y", color="#aaaaaa", fontsize=9)
    ax.set_title(f"2D Translation  \u2014  T = ({tx}, {ty})",
                 color="#dddddd", fontsize=10, pad=8)
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3D Rotation
# ══════════════════════════════════════════════════════════════════════════════

def run_3d_rotation(x, y, z, theta_deg, axis, clockwise=False):
    """Rotate point (x,y,z) by theta_deg about the given axis."""
    theta = math.radians(theta_deg)
    if clockwise:
        theta = -theta
    c = round(math.cos(theta), 6)
    s = round(math.sin(theta), 6)

    if axis == "X":
        xp, yp, zp = x, y * c - z * s, y * s + z * c
        rows = [
            {"Component": "x'", "Formula": "x' = x",
             "Substitution": f"{x}",
             "Result": round(xp, 4)},
            {"Component": "y'", "Formula": "y' = y·cosθ − z·sinθ",
             "Substitution": f"({y})×({c}) − ({z})×({s})",
             "Result": round(yp, 4)},
            {"Component": "z'", "Formula": "z' = y·sinθ + z·cosθ",
             "Substitution": f"({y})×({s}) + ({z})×({c})",
             "Result": round(zp, 4)},
        ]
    elif axis == "Y":
        xp, yp, zp = x * c + z * s, y, z * c - x * s
        rows = [
            {"Component": "x'", "Formula": "x' = x·cosθ + z·sinθ",
             "Substitution": f"({x})×({c}) + ({z})×({s})",
             "Result": round(xp, 4)},
            {"Component": "y'", "Formula": "y' = y",
             "Substitution": f"{y}",
             "Result": round(yp, 4)},
            {"Component": "z'", "Formula": "z' = z·cosθ − x·sinθ",
             "Substitution": f"({z})×({c}) − ({x})×({s})",
             "Result": round(zp, 4)},
        ]
    else:  # Z
        xp, yp, zp = x * c - y * s, x * s + y * c, z
        rows = [
            {"Component": "x'", "Formula": "x' = x·cosθ − y·sinθ",
             "Substitution": f"({x})×({c}) − ({y})×({s})",
             "Result": round(xp, 4)},
            {"Component": "y'", "Formula": "y' = x·sinθ + y·cosθ",
             "Substitution": f"({x})×({s}) + ({y})×({c})",
             "Result": round(yp, 4)},
            {"Component": "z'", "Formula": "z' = z",
             "Substitution": f"{z}",
             "Result": round(zp, 4)},
        ]
    return round(xp, 4), round(yp, 4), round(zp, 4), rows, c, s


def run_3d_find_angle(x, y, z, xp, yp, zp, axis):
    """Recover the rotation angle from original and rotated coordinates."""
    if axis == "X":
        theta = math.atan2(y * zp - z * yp, y * yp + z * zp)
    elif axis == "Y":
        theta = math.atan2(z * xp - x * zp, x * xp + z * zp)
    else:  # Z
        theta = math.atan2(x * yp - y * xp, x * xp + y * yp)
    return round(math.degrees(theta), 4)


def draw_3d_rotation(x, y, z, xp, yp, zp, theta_deg, axis, clockwise=False):
    """3-D plot showing original point, rotated point and rotation arc."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    theta_rad = math.radians(theta_deg)
    if clockwise:
        theta_rad = -theta_rad
    t_vals = np.linspace(0, theta_rad, 80)
    c_v, s_v = np.cos(t_vals), np.sin(t_vals)

    if axis == "X":
        arc_x = np.full(80, float(x))
        arc_y = y * c_v - z * s_v
        arc_z = y * s_v + z * c_v
    elif axis == "Y":
        arc_x = x * c_v + z * s_v
        arc_y = np.full(80, float(y))
        arc_z = z * c_v - x * s_v
    else:
        arc_x = x * c_v - y * s_v
        arc_y = x * s_v + y * c_v
        arc_z = np.full(80, float(z))

    fig = plt.figure(figsize=(7, 6))
    fig.patch.set_facecolor("#0e1117")
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.set_facecolor("#0e1117")

    ax_len = max(abs(x), abs(y), abs(z), abs(xp), abs(yp), abs(zp), 1.5) * 1.35

    # Coordinate axes
    ax3.quiver(0, 0, 0, ax_len, 0, 0, color="#e74c3c", linewidth=1.2,
               arrow_length_ratio=0.08)
    ax3.quiver(0, 0, 0, 0, ax_len, 0, color="#2ecc71", linewidth=1.2,
               arrow_length_ratio=0.08)
    ax3.quiver(0, 0, 0, 0, 0, ax_len, color="#3498db", linewidth=1.2,
               arrow_length_ratio=0.08)
    ax3.text(ax_len * 1.08, 0, 0, "X", color="#e74c3c", fontsize=9, fontweight="bold")
    ax3.text(0, ax_len * 1.08, 0, "Y", color="#2ecc71", fontsize=9, fontweight="bold")
    ax3.text(0, 0, ax_len * 1.08, "Z", color="#3498db", fontsize=9, fontweight="bold")

    # Dashed radii to origin
    ax3.plot([0, x],  [0, y],  [0, z],  color="#3498db", linewidth=1,
             alpha=0.4, linestyle="--")
    ax3.plot([0, xp], [0, yp], [0, zp], color="#2ecc71", linewidth=1,
             alpha=0.4, linestyle="--")

    # Rotation arc
    ax3.plot(arc_x, arc_y, arc_z, color="#f39c12", linewidth=2, alpha=0.8, zorder=4)

    # Points
    ax3.scatter([x],  [y],  [z],  color="#3498db", s=90,
                edgecolors="#2980b9", linewidths=1.2, zorder=5)
    ax3.scatter([xp], [yp], [zp], color="#2ecc71", s=90,
                edgecolors="#27ae60", linewidths=1.2, zorder=5)
    ax3.text(x,  y,  z,  f"  A({x},{y},{z})",
             color="#7fb3d3", fontsize=8, fontweight="bold")
    ax3.text(xp, yp, zp, f"  A'({xp},{yp},{zp})",
             color="#82e0aa", fontsize=8, fontweight="bold")

    direction_label = "Clockwise" if clockwise else "Counter-Clockwise"
    ax3.set_title(
        f"3D Rotation — {axis}-axis  \u03b8={theta_deg}\u00b0  ({direction_label})",
        color="#dddddd", fontsize=10, pad=10)
    ax3.set_xlabel("X", color="#aaaaaa", fontsize=8)
    ax3.set_ylabel("Y", color="#aaaaaa", fontsize=8)
    ax3.set_zlabel("Z", color="#aaaaaa", fontsize=8)
    ax3.tick_params(colors="#666666", labelsize=6)
    for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#2a2d35")
    ax3.xaxis.line.set_color("#333333")
    ax3.yaxis.line.set_color("#333333")
    ax3.zaxis.line.set_color("#333333")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3D Translation
# ══════════════════════════════════════════════════════════════════════════════

def run_3d_translation(points, tx, ty, tz):
    """Translate a list of (x,y,z) points by vector (tx,ty,tz)."""
    rows = []
    new_points = []
    for i, (x, y, z) in enumerate(points):
        xp = round(x + tx, 4)
        yp = round(y + ty, 4)
        zp = round(z + tz, 4)
        lbl = POINT_LABELS[i]
        rows.append({"Point": lbl,
                     "x": x,  "y": y,  "z": z,
                     "Tx": tx, "Ty": ty, "Tz": tz,
                     "x' = x+Tx": xp,
                     "y' = y+Ty": yp,
                     "z' = z+Tz": zp})
        new_points.append((xp, yp, zp))
    return new_points, rows


def draw_3d_translation(points, new_points, tx, ty, tz):
    """3-D plot showing original and translated points with shift arrows."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    all_coords = [c for pt in points + new_points for c in pt]
    ax_len = max(max(abs(v) for v in all_coords), 1.5) * 1.35

    fig = plt.figure(figsize=(7, 6))
    fig.patch.set_facecolor("#0e1117")
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.set_facecolor("#0e1117")

    # Coordinate axes
    ax3.quiver(0, 0, 0, ax_len, 0, 0, color="#e74c3c", linewidth=1.2,
               arrow_length_ratio=0.08)
    ax3.quiver(0, 0, 0, 0, ax_len, 0, color="#2ecc71", linewidth=1.2,
               arrow_length_ratio=0.08)
    ax3.quiver(0, 0, 0, 0, 0, ax_len, color="#3498db", linewidth=1.2,
               arrow_length_ratio=0.08)
    ax3.text(ax_len * 1.08, 0, 0, "X", color="#e74c3c",
             fontsize=9, fontweight="bold")
    ax3.text(0, ax_len * 1.08, 0, "Y", color="#2ecc71",
             fontsize=9, fontweight="bold")
    ax3.text(0, 0, ax_len * 1.08, "Z", color="#3498db",
             fontsize=9, fontweight="bold")

    for i, ((x, y, z), (xp, yp, zp)) in enumerate(zip(points, new_points)):
        lbl = POINT_LABELS[i]
        # Dashed lines from origin
        ax3.plot([0, x],  [0, y],  [0, z],  color="#3498db",
                 linewidth=1, alpha=0.35, linestyle="--")
        ax3.plot([0, xp], [0, yp], [0, zp], color="#2ecc71",
                 linewidth=1, alpha=0.35, linestyle="--")
        # Translation arrow
        ax3.quiver(x, y, z, xp - x, yp - y, zp - z,
                   color="#f39c12", linewidth=1.8, arrow_length_ratio=0.12,
                   alpha=0.9, zorder=4)
        # Original point
        ax3.scatter([x], [y], [z], color="#3498db", s=80,
                    edgecolors="#2980b9", linewidths=1.2, zorder=5)
        ax3.text(x, y, z, f"  {lbl}({x},{y},{z})",
                 color="#7fb3d3", fontsize=8, fontweight="bold")
        # Translated point
        ax3.scatter([xp], [yp], [zp], color="#2ecc71", s=80,
                    edgecolors="#27ae60", linewidths=1.2, zorder=5)
        ax3.text(xp, yp, zp, f"  {lbl}'({xp},{yp},{zp})",
                 color="#82e0aa", fontsize=8, fontweight="bold")

    ax3.set_title(
        f"3D Translation  \u2014  T = ({tx}, {ty}, {tz})",
        color="#dddddd", fontsize=10, pad=10)
    ax3.set_xlabel("X", color="#aaaaaa", fontsize=8)
    ax3.set_ylabel("Y", color="#aaaaaa", fontsize=8)
    ax3.set_zlabel("Z", color="#aaaaaa", fontsize=8)
    ax3.tick_params(colors="#666666", labelsize=6)
    for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#2a2d35")
    ax3.xaxis.line.set_color("#333333")
    ax3.yaxis.line.set_color("#333333")
    ax3.zaxis.line.set_color("#333333")
    plt.tight_layout()
    return fig


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
    subtab_rot2d, subtab_trans2d = st.tabs([
        "↻  2D Rotation",
        "↔  2D Translation",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB A — 2D Rotation
    # ══════════════════════════════════════════════════════════════════════════
    with subtab_rot2d:
        section_header("2D Rotation", "tba", "tba")

        # ── Direction toggle ──────────────────────────────────────────────────
        direction = st.radio(
            "Rotation direction",
            options=["Counter-Clockwise (CCW)  ↺", "Clockwise (CW)  ↻"],
            horizontal=True,
            key="rot2d_dir",
        )
        clockwise_2d = direction.startswith("Clockwise")

        # ── Rotation angle ─────────────────────────────────────────────────────
        theta_deg = st.number_input(
            "Rotation angle θ (degrees)",
            value=45.0, min_value=0.0, max_value=360.0,
            step=1.0, format="%.2f", key="rot2d_theta",
        )

        st.divider()

        # ── Number of points ──────────────────────────────────────────────────
        st.subheader("Input Points")
        n_pts = st.slider("Number of points", min_value=1, max_value=10,
                          value=3, key="rot2d_npts")

        points_2d = []
        cols_per_row = min(n_pts, 5)
        for row_start in range(0, n_pts, cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(row_cols):
                idx = row_start + j
                if idx >= n_pts:
                    break
                label = POINT_LABELS[idx]
                with col:
                    st.markdown(f"**Point {label}**")
                    sub = st.columns(2)
                    px = sub[0].number_input(f"x", value=float(idx + 1),
                                             step=1.0, key=f"rot2d_x{idx}",
                                             label_visibility="visible")
                    py = sub[1].number_input(f"y", value=float(idx + 1),
                                             step=1.0, key=f"rot2d_y{idx}",
                                             label_visibility="visible")
                    points_2d.append((float(px), float(py)))

        st.divider()

        # ── Formula / Derivation ───────────────────────────────────────────────
        with st.expander("📐  Formula & Derivation", expanded=True):
            st.markdown(
                "<style>.katex-display{text-align:left!important;margin:0.3rem 0!important;}"
                ".katex-display>.katex{text-align:left!important;}</style>",
                unsafe_allow_html=True,
            )
            st.markdown("**Radial-distance approach**")
            st.markdown(r"For a point $(x, y)$ at radial distance $r$ from the origin and initial angle $\phi$:")
            st.latex(r"r = \sqrt{x^2 + y^2}")
            st.latex(r"\cos\phi = \frac{x}{r} \qquad \sin\phi = \frac{y}{r}")
            st.latex(r"x = r\cos\phi \qquad y = r\sin\phi")
            st.markdown(r"After rotating by $\theta$ the new angle is $\phi + \theta$:")
            st.latex(r"x' = r\cos(\phi+\theta) \qquad y' = r\sin(\phi+\theta)")
            st.markdown("Expanding with the angle-addition identities:")
            st.latex(r"\cos(\phi+\theta) = \cos\phi\cos\theta - \sin\phi\sin\theta")
            st.latex(r"\sin(\phi+\theta) = \sin\phi\cos\theta + \cos\phi\sin\theta")
            st.markdown(r"Substituting $x = r\cos\phi$ and $y = r\sin\phi$:")
            st.latex(r"x' = r\cos\phi\cos\theta - r\sin\phi\sin\theta = x\cos\theta - y\sin\theta")
            st.latex(r"y' = r\sin\phi\cos\theta + r\cos\phi\sin\theta = x\sin\theta + y\cos\theta")
            st.success(
                "**Final formulas:**\n\n"
                r"$x' = x\cos\theta - y\sin\theta$" + "\n\n" +
                r"$y' = x\sin\theta + y\cos\theta$"
            )
            st.info(
                r"**Clockwise rotation** uses $-\theta$, giving:" + "\n\n" +
                r"$x' = x\cos\theta + y\sin\theta$" + "\n\n" +
                r"$y' = -x\sin\theta + y\cos\theta$"
            )

        # ── Computation ────────────────────────────────────────────────────────
        rot_rows, new_pts_2d = run_2d_rotation(points_2d, theta_deg, clockwise_2d)

        # ── Results table ──────────────────────────────────────────────────────
        st.subheader("Rotation Results")
        theta_r = math.radians(theta_deg)
        eff_theta = -theta_r if clockwise_2d else theta_r
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("θ (degrees)", f"{theta_deg}°")
        rc2.metric("cos θ", f"{math.cos(eff_theta):.6f}")
        rc3.metric("sin θ", f"{math.sin(eff_theta):.6f}")

        st.dataframe(pd.DataFrame(rot_rows), hide_index=True, width='stretch')

        st.divider()

        # ── Visualization ──────────────────────────────────────────────────────
        st.subheader("Visualization")
        fig_2d = draw_2d_rotation(points_2d, new_pts_2d, theta_deg, clockwise_2d)
        if fig_2d:
            st.pyplot(fig_2d, width='stretch')
            plt.close(fig_2d)

        st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB B — 2D Translation
    # ══════════════════════════════════════════════════════════════════════════
    with subtab_trans2d:
        section_header("2D Translation", "tba", "tba")

        # ── Shape selector ────────────────────────────────────────────────────
        trans_shape = st.radio(
            "Shape type",
            options=["Polygon / Points", "Circle"],
            horizontal=True,
            key="trans2d_shape",
        )

        st.divider()

        # ── Formula expander ──────────────────────────────────────────────────
        with st.expander("📐  Formula", expanded=True):
            st.markdown(
                "<style>.katex-display{text-align:left!important;margin:0.3rem 0!important;}"
                ".katex-display>.katex{text-align:left!important;}</style>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "A **translation** shifts every point by the translation "
                r"vector $(T_x,\, T_y)$:"
            )
            st.latex(r"x' = x + T_x")
            st.latex(r"y' = y + T_y")
            st.markdown("**Matrix form:**")
            st.latex(
                r"\begin{bmatrix} x' \\ y' \end{bmatrix} = "
                r"\begin{bmatrix} T_x \\ T_y \end{bmatrix} + "
                r"\begin{bmatrix} x \\ y \end{bmatrix}"
            )
            st.info(
                r"The radius $r$ of a circle is **unchanged** by translation — "
                "only its centre moves."
            )

        st.divider()

        # ── Translation vector ─────────────────────────────────────────────────
        st.subheader("Translation Vector  (Tx, Ty)")
        tv1, tv2 = st.columns(2)
        with tv1:
            tx_val = st.number_input(
                "Tx  (shift along X-axis)", value=3.0,
                step=1.0, format="%.2f", key="trans2d_tx",
            )
        with tv2:
            ty_val = st.number_input(
                "Ty  (shift along Y-axis)", value=2.0,
                step=1.0, format="%.2f", key="trans2d_ty",
            )
        tx_val, ty_val = float(tx_val), float(ty_val)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # CIRCLE branch
        # ══════════════════════════════════════════════════════════════════════
        if trans_shape == "Circle":
            st.subheader("Circle Input")
            cti1, cti2, cti3 = st.columns(3)
            with cti1:
                t_cx = st.number_input("Center x", value=2.0, step=1.0,
                                       format="%.2f", key="trans2d_cx")
            with cti2:
                t_cy = st.number_input("Center y", value=3.0, step=1.0,
                                       format="%.2f", key="trans2d_cy")
            with cti3:
                t_r  = st.number_input("Radius r", value=4.0, min_value=0.1,
                                       step=0.5, format="%.2f", key="trans2d_r")
            t_cx, t_cy, t_r = float(t_cx), float(t_cy), float(t_r)

            st.divider()

            # Compute
            circ_rows, new_cx, new_cy = run_2d_translation_circle(
                t_cx, t_cy, t_r, tx_val, ty_val
            )

            # Metrics
            st.subheader("Translation Results")
            tm1, tm2 = st.columns(2)
            with tm1:
                st.markdown("**Original circle**")
                st.markdown(
                    f"Centre = `({t_cx}, {t_cy})` &nbsp;&nbsp; Radius = `{t_r}`"
                )
            with tm2:
                st.markdown("**Translated circle**")
                st.markdown(
                    f"Centre = `({new_cx}, {new_cy})` &nbsp;&nbsp; "
                    f"Radius = `{t_r}`  (unchanged)"
                )

            st.dataframe(pd.DataFrame(circ_rows), hide_index=True,
                         width='stretch')

            st.divider()

            # Visualisation
            st.subheader("Visualization")
            fig_tc = draw_2d_translation(
                [(t_cx, t_cy)], [(new_cx, new_cy)],
                tx_val, ty_val, is_circle=True, radius=t_r,
            )
            if fig_tc:
                st.pyplot(fig_tc, width='stretch')
                plt.close(fig_tc)

        # ══════════════════════════════════════════════════════════════════════
        # POLYGON / POINTS branch
        # ══════════════════════════════════════════════════════════════════════
        else:
            st.subheader("Input Points")
            n_trans = st.slider("Number of points", min_value=1, max_value=10,
                                value=3, key="trans2d_npts")

            trans_pts = []
            cols_per_row_t = min(n_trans, 5)
            for row_start in range(0, n_trans, cols_per_row_t):
                row_cols = st.columns(cols_per_row_t)
                for j, col in enumerate(row_cols):
                    idx = row_start + j
                    if idx >= n_trans:
                        break
                    label = POINT_LABELS[idx]
                    with col:
                        st.markdown(f"**Point {label}**")
                        sub = st.columns(2)
                        tp_x = sub[0].number_input(
                            "x", value=float(idx + 1), step=1.0,
                            key=f"trans2d_x{idx}",
                            label_visibility="visible",
                        )
                        tp_y = sub[1].number_input(
                            "y", value=float(idx + 1), step=1.0,
                            key=f"trans2d_y{idx}",
                            label_visibility="visible",
                        )
                        trans_pts.append((float(tp_x), float(tp_y)))

            st.divider()

            # Compute
            trans_rows, new_trans_pts = run_2d_translation(
                trans_pts, tx_val, ty_val
            )

            # Metrics
            st.subheader("Translation Results")
            tpm1, tpm2, tpm3 = st.columns(3)
            tpm1.metric("Tx", tx_val)
            tpm2.metric("Ty", ty_val)
            tpm3.metric("Points translated", n_trans)

            st.dataframe(pd.DataFrame(trans_rows), hide_index=True,
                         width='stretch')

            st.divider()

            # Visualisation
            st.subheader("Visualization")
            fig_tp = draw_2d_translation(
                trans_pts, new_trans_pts, tx_val, ty_val,
            )
            if fig_tp:
                st.pyplot(fig_tp, width='stretch')
                plt.close(fig_tp)

        st.divider()


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
    subtab_rot3d, subtab_trans3d = st.tabs([
        "↻  3D Rotation",
        "↔  3D Translation",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB A — 3D Rotation
    # ══════════════════════════════════════════════════════════════════════════
    with subtab_rot3d:
        section_header("3D Rotation", "tba", "tba")

        # ── Mode ─────────────────────────────────────────────────────────────
        mode_3d = st.radio(
            "Mode",
            options=["Find New Coordinates", "Find Rotation Angle"],
            horizontal=True,
            key="rot3d_mode",
        )

        # ── Axis ─────────────────────────────────────────────────────────────
        axis_3d = st.radio(
            "Rotation axis",
            options=["X", "Y", "Z"],
            horizontal=True,
            key="rot3d_axis",
        )

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # MODE 1 — Find New Coordinates
        # ══════════════════════════════════════════════════════════════════════
        if mode_3d == "Find New Coordinates":

            # Direction
            dir_3d = st.radio(
                "Rotation direction",
                options=["Counter-Clockwise (CCW)  ↺", "Clockwise (CW)  ↻"],
                horizontal=True,
                key="rot3d_dir",
            )
            cw_3d = dir_3d.startswith("Clockwise")

            # Angle
            theta_3d = st.number_input(
                "Rotation angle θ (degrees)",
                value=45.0, min_value=0.0, max_value=360.0,
                step=1.0, format="%.2f", key="rot3d_theta",
            )

            # Point input
            st.subheader("Input Point A(x, y, z)")
            pc1, pc2, pc3 = st.columns(3)
            pt3_x = pc1.number_input("x", value=1.0, step=1.0, key="rot3d_x")
            pt3_y = pc2.number_input("y", value=2.0, step=1.0, key="rot3d_y")
            pt3_z = pc3.number_input("z", value=3.0, step=1.0, key="rot3d_z")
            pt3_x, pt3_y, pt3_z = float(pt3_x), float(pt3_y), float(pt3_z)

            st.divider()

            # Formula expander
            with st.expander("📐  Rotation Formulas", expanded=True):
                st.markdown(
                    "<style>.katex-display{text-align:left!important;margin:0.3rem 0!important;}"
                    ".katex-display>.katex{text-align:left!important;}</style>",
                    unsafe_allow_html=True,
                )
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    st.markdown("**X-axis rotation**")
                    st.latex(r"x' = x")
                    st.latex(r"y' = y\cos\theta - z\sin\theta")
                    st.latex(r"z' = y\sin\theta + z\cos\theta")
                with fc2:
                    st.markdown("**Y-axis rotation**")
                    st.latex(r"x' = x\cos\theta + z\sin\theta")
                    st.latex(r"y' = y")
                    st.latex(r"z' = z\cos\theta - x\sin\theta")
                with fc3:
                    st.markdown("**Z-axis rotation**")
                    st.latex(r"x' = x\cos\theta - y\sin\theta")
                    st.latex(r"y' = x\sin\theta + y\cos\theta")
                    st.latex(r"z' = z")

            # Compute
            xp_3d, yp_3d, zp_3d, rot3_rows, cos_t3, sin_t3 = run_3d_rotation(
                pt3_x, pt3_y, pt3_z, theta_3d, axis_3d, cw_3d
            )

            # Result metrics
            st.subheader("Rotation Results")
            eff_t3 = -math.radians(theta_3d) if cw_3d else math.radians(theta_3d)
            rm1, rm2, rm3, rm4, rm5 = st.columns(5)
            rm1.metric("θ", f"{theta_3d}°")
            rm2.metric("Axis", axis_3d)
            rm3.metric("cos θ", f"{math.cos(eff_t3):.6f}")
            rm4.metric("sin θ", f"{math.sin(eff_t3):.6f}")
            rm5.metric("Direction", "CW" if cw_3d else "CCW")

            coord_c1, coord_c2 = st.columns(2)
            with coord_c1:
                st.markdown("**Original point**")
                st.markdown(f"`A = ({pt3_x}, {pt3_y}, {pt3_z})`")
            with coord_c2:
                st.markdown("**Rotated point**")
                st.markdown(f"`A' = ({xp_3d}, {yp_3d}, {zp_3d})`")

            st.dataframe(pd.DataFrame(rot3_rows), hide_index=True, width='stretch')

            st.divider()

            # Visualization
            st.subheader("Visualization")
            fig_3d = draw_3d_rotation(
                pt3_x, pt3_y, pt3_z, xp_3d, yp_3d, zp_3d,
                theta_3d, axis_3d, cw_3d
            )
            if fig_3d:
                st.pyplot(fig_3d, width='stretch')
                plt.close(fig_3d)

        # ══════════════════════════════════════════════════════════════════════
        # MODE 2 — Find Rotation Angle
        # ══════════════════════════════════════════════════════════════════════
        else:
            st.subheader("Input Coordinates")
            orig_col, new_col = st.columns(2)

            with orig_col:
                st.markdown("**Original point A(x, y, z)**")
                fa1, fa2, fa3 = st.columns(3)
                fa_x = fa1.number_input("x",  value=1.0, step=1.0, key="fa_x")
                fa_y = fa2.number_input("y",  value=2.0, step=1.0, key="fa_y")
                fa_z = fa3.number_input("z",  value=3.0, step=1.0, key="fa_z")

            with new_col:
                st.markdown("**Rotated point A'(x', y', z')**")
                fb1, fb2, fb3 = st.columns(3)
                fb_x = fb1.number_input("x'", value=1.0, step=1.0, key="fb_x")
                fb_y = fb2.number_input("y'", value=-3.0, step=1.0, key="fb_y")
                fb_z = fb3.number_input("z'", value=2.0, step=1.0, key="fb_z")

            fa_x, fa_y, fa_z = float(fa_x), float(fa_y), float(fa_z)
            fb_x, fb_y, fb_z = float(fb_x), float(fb_y), float(fb_z)

            st.divider()

            # Formula expander
            with st.expander("📐  How angle is recovered", expanded=True):
                st.markdown(
                    "<style>.katex-display{text-align:left!important;margin:0.3rem 0!important;}"
                    ".katex-display>.katex{text-align:left!important;}</style>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    r"Using the dot and cross product of the two vectors in the rotation plane:"
                )
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    st.markdown("**X-axis** (rotate in YZ plane)")
                    st.latex(r"\theta = \mathrm{atan2}(y z' - z y',\; y y' + z z')")
                with fc2:
                    st.markdown("**Y-axis** (rotate in ZX plane)")
                    st.latex(r"\theta = \mathrm{atan2}(z x' - x z',\; x x' + z z')")
                with fc3:
                    st.markdown("**Z-axis** (rotate in XY plane)")
                    st.latex(r"\theta = \mathrm{atan2}(x y' - y x',\; x x' + y y')")

            # Invariant check
            tol = 1e-6
            invariant_ok = True
            if axis_3d == "X" and abs(fa_x - fb_x) > tol:
                st.warning(f"For X-axis rotation x should stay the same, but x={fa_x} and x'={fb_x}. " +
                           "θ is computed from the Y/Z components only.")
                invariant_ok = False
            elif axis_3d == "Y" and abs(fa_y - fb_y) > tol:
                st.warning(f"For Y-axis rotation y should stay the same, but y={fa_y} and y'={fb_y}. " +
                           "θ is computed from the X/Z components only.")
                invariant_ok = False
            elif axis_3d == "Z" and abs(fa_z - fb_z) > tol:
                st.warning(f"For Z-axis rotation z should stay the same, but z={fa_z} and z'={fb_z}. " +
                           "θ is computed from the X/Y components only.")
                invariant_ok = False

            # Degenerate case
            degen = False
            if axis_3d == "X" and (fa_y == 0 and fa_z == 0):
                st.error("Point lies on the X-axis — angle is indeterminate.")
                degen = True
            elif axis_3d == "Y" and (fa_x == 0 and fa_z == 0):
                st.error("Point lies on the Y-axis — angle is indeterminate.")
                degen = True
            elif axis_3d == "Z" and (fa_x == 0 and fa_y == 0):
                st.error("Point lies on the Z-axis — angle is indeterminate.")
                degen = True

            if not degen:
                angle_found = run_3d_find_angle(
                    fa_x, fa_y, fa_z, fb_x, fb_y, fb_z, axis_3d
                )

                st.subheader("Result")
                ar1, ar2, ar3 = st.columns(3)
                ar1.metric("Rotation Axis", axis_3d)
                ar2.metric("θ (degrees)",   f"{angle_found}°")
                ar3.metric("Direction",
                           "Clockwise" if angle_found < 0 else "Counter-Clockwise")

                with st.expander("Step-by-step substitution"):
                    if axis_3d == "X":
                        st.markdown(
                            rf"$$\theta = \mathrm{{atan2}}("
                            rf"{fa_y} \times {fb_z} - {fa_z} \times {fb_y},\ "
                            rf"{fa_y} \times {fb_y} + {fa_z} \times {fb_z}) = "
                            rf"\mathrm{{atan2}}({fa_y*fb_z - fa_z*fb_y:.4f},\ "
                            rf"{fa_y*fb_y + fa_z*fb_z:.4f}) = {angle_found}°$$"
                        )
                    elif axis_3d == "Y":
                        st.markdown(
                            rf"$$\theta = \mathrm{{atan2}}("
                            rf"{fa_z} \times {fb_x} - {fa_x} \times {fb_z},\ "
                            rf"{fa_x} \times {fb_x} + {fa_z} \times {fb_z}) = "
                            rf"\mathrm{{atan2}}({fa_z*fb_x - fa_x*fb_z:.4f},\ "
                            rf"{fa_x*fb_x + fa_z*fb_z:.4f}) = {angle_found}°$$"
                        )
                    else:
                        st.markdown(
                            rf"$$\theta = \mathrm{{atan2}}("
                            rf"{fa_x} \times {fb_y} - {fa_y} \times {fb_x},\ "
                            rf"{fa_x} \times {fb_x} + {fa_y} \times {fb_y}) = "
                            rf"\mathrm{{atan2}}({fa_x*fb_y - fa_y*fb_x:.4f},\ "
                            rf"{fa_x*fb_x + fa_y*fb_y:.4f}) = {angle_found}°$$"
                        )

                # Draw with the recovered angle
                st.subheader("Visualization")
                fig_3d_fa = draw_3d_rotation(
                    fa_x, fa_y, fa_z, fb_x, fb_y, fb_z,
                    abs(angle_found), axis_3d, clockwise=(angle_found < 0)
                )
                if fig_3d_fa:
                    st.pyplot(fig_3d_fa, width='stretch')
                    plt.close(fig_3d_fa)

        st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB B — 3D Translation
    # ══════════════════════════════════════════════════════════════════════════
    with subtab_trans3d:
        section_header("3D Translation", "tba", "tba")

        # ── Formula expander ──────────────────────────────────────────────────
        with st.expander("📐  Formula", expanded=True):
            st.markdown(
                "<style>.katex-display{text-align:left!important;margin:0.3rem 0!important;}"
                ".katex-display>.katex{text-align:left!important;}</style>",
                unsafe_allow_html=True,
            )
            st.markdown(
                r"A **3D translation** shifts a point by the translation "
                r"vector $(T_x,\, T_y,\, T_z)$:"
            )
            st.latex(r"x' = x + T_x")
            st.latex(r"y' = y + T_y")
            st.latex(r"z' = z + T_z")
            st.markdown("**Matrix form:**")
            st.latex(
                r"\begin{bmatrix} x' \\ y' \\ z' \end{bmatrix} = "
                r"\begin{bmatrix} T_x \\ T_y \\ T_z \end{bmatrix} + "
                r"\begin{bmatrix} x \\ y \\ z \end{bmatrix}"
            )

        st.divider()

        # ── Number of points ──────────────────────────────────────────────────
        st.subheader("Input Points")
        n_t3 = st.slider("Number of points", min_value=1, max_value=4,
                         value=1, key="trans3d_npts")

        t3_points = []
        for idx in range(n_t3):
            lbl = POINT_LABELS[idx]
            st.markdown(f"**Point {lbl}**")
            ti1, ti2, ti3 = st.columns(3)
            with ti1:
                px = st.number_input("x", value=float(idx + 1), step=1.0,
                                     format="%.2f", key=f"trans3d_x{idx}")
            with ti2:
                py = st.number_input("y", value=float(idx + 2), step=1.0,
                                     format="%.2f", key=f"trans3d_y{idx}")
            with ti3:
                pz = st.number_input("z", value=float(idx + 3), step=1.0,
                                     format="%.2f", key=f"trans3d_z{idx}")
            t3_points.append((float(px), float(py), float(pz)))

        st.divider()

        # ── Translation vector ────────────────────────────────────────────────
        st.subheader("Translation Vector  (Tx, Ty, Tz)")
        tv1, tv2, tv3 = st.columns(3)
        with tv1:
            t3_tx = st.number_input("Tx  (shift along X)", value=1.0,
                                    step=1.0, format="%.2f", key="trans3d_tx")
        with tv2:
            t3_ty = st.number_input("Ty  (shift along Y)", value=2.0,
                                    step=1.0, format="%.2f", key="trans3d_ty")
        with tv3:
            t3_tz = st.number_input("Tz  (shift along Z)", value=3.0,
                                    step=1.0, format="%.2f", key="trans3d_tz")
        t3_tx, t3_ty, t3_tz = float(t3_tx), float(t3_ty), float(t3_tz)

        st.divider()

        # ── Compute ───────────────────────────────────────────────────────────
        new_t3_points, trans3_rows = run_3d_translation(
            t3_points, t3_tx, t3_ty, t3_tz
        )

        # ── Results ───────────────────────────────────────────────────────────
        st.subheader("Translation Results")
        tr1, tr2, tr3, tr4 = st.columns(4)
        tr1.metric("Tx", t3_tx)
        tr2.metric("Ty", t3_ty)
        tr3.metric("Tz", t3_tz)
        tr4.metric("Points translated", n_t3)

        st.dataframe(pd.DataFrame(trans3_rows), hide_index=True, width='stretch')

        st.divider()

        # ── Visualization ─────────────────────────────────────────────────────
        st.subheader("Visualization")
        fig_t3 = draw_3d_translation(
            t3_points, new_t3_points, t3_tx, t3_ty, t3_tz
        )
        if fig_t3:
            st.pyplot(fig_t3, width='stretch')
            plt.close(fig_t3)

        st.divider()


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.9rem'>"
    "Developed by <a href='https://redwan-rahman.netlify.app/' target='_blank'>Redwan Rahman</a>"
    " and <a href='https://claude.ai/' target='_blank'>Claude</a>"
    "</div>",
    unsafe_allow_html=True,
)
