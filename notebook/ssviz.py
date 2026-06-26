"""
ssviz.py - visualization helpers for the shape-servoing tutorial.
==================================================================

This module holds the plotting and animation machinery so that the notebook
cells stay short and focused on the method. You do not need to read this file
to follow the tutorial; the notebook calls these functions by name. Each public
function either draws a figure (and returns ``None``) or builds an animation
(and returns an ``IPython.display.HTML`` object to be shown as the cell output).

The only physics dependency is ``dlo.py``; everything here is matplotlib.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from IPython.display import HTML

from dlo import split_grasp

plt.rcParams["animation.embed_limit"] = 120

# --------------------------------------------------------------------------- #
#  palette
# --------------------------------------------------------------------------- #
BG="#0f1622"; GRID="#1b2636"; STEEL="#5b6b80"
ROD="#F8A23B"; SAMPLE="#F4D35E"; TARGET="#5BD6A6"
ARM_L="#5AA9F0"; ARM_R="#2FC6AE"; ERR="#FF6B6B"; INK="#D8E0EC"; MUTE="#8090A6"
L_BLUE="#2E6FB0"; L_ORANGE="#E8772E"; L_TEAL="#1AA38A"; L_GREEN="#2E9E5B"
L_RED="#D64545"; L_GOLD="#C99A1E"; PURPLE="#7E57C2"; INKL="#1b2533"
F_BLUE="#E7EFF9"; F_STEEL="#ECEEF1"; F_ORANGE="#FCEEDD"; F_GREEN="#E7F5EC"; F_PURPLE="#F0EAF8"

DOF_LABELS = ["xL", "yL", "thL", "xR", "yR", "thR"]

# --------------------------------------------------------------------------- #
#  the robot stage (dark theme)
# --------------------------------------------------------------------------- #
ARM_BASE_L = np.array([-5.9, -2.1]); ARM_BASE_R = np.array([5.9, -2.1]); _L1 = _L2 = 3.3

def _ik(base, tgt, elbow):
    base = np.asarray(base, float); tgt = np.asarray(tgt, float); d = tgt - base
    D = np.clip(float(np.hypot(d[0], d[1])), 1e-6, _L1 + _L2 - 1e-6)
    a = np.arccos(np.clip((D*D + _L1*_L1 - _L2*_L2) / (2*_L1*D), -1, 1))
    sh = np.arctan2(d[1], d[0]) + elbow*a
    return np.array([base, base + _L1*np.array([np.cos(sh), np.sin(sh)]), tgt])

def _arm(ax, base, g, c, el, z=3):
    p = _ik(base, g, el)
    ax.plot(p[:,0], p[:,1], "-", color=c, lw=7.5, solid_capstyle="round", zorder=z, alpha=.96)
    ax.scatter(p[1,0], p[1,1], s=52, color=c, edgecolors=BG, linewidths=1.6, zorder=z+1)
    ax.scatter([base[0]], [base[1]], marker="s", s=150, color=STEEL, edgecolors=c, linewidths=1.6, zorder=z)

def _grip(ax, pos, th, side, c, z=6):
    seg = np.array([[-.30,-.17],[-.30,.17],[-.30,.17],[.06,.17],[-.30,-.17],[.06,-.17]])
    fl = -1. if side > 0 else 1.; cc, ss = np.cos(th), np.sin(th)
    P = (seg*np.array([fl,1.])) @ np.array([[cc,-ss],[ss,cc]]).T + np.asarray(pos, float)
    for a, b in [(0,1),(2,3),(4,5)]:
        ax.plot([P[a,0],P[b,0]],[P[a,1],P[b,1]], color=c, lw=4.6, solid_capstyle="round", zorder=z)

def _rod(ax, n, color=ROD, lw=5., alpha=1., glow=True, z=2):
    if glow: ax.plot(n[:,0], n[:,1], color=color, lw=lw*2.8, alpha=.10*alpha, solid_capstyle="round", zorder=z-.1)
    ax.plot(n[:,0], n[:,1], color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z)

def _samp(ax, pts, color=SAMPLE, s=72, z=5):
    ax.scatter(pts[:,0], pts[:,1], s=s, color=color, edgecolors=BG, linewidths=1.4, zorder=z)

def _ghost(ax, n, pts):
    ax.plot(n[:,0], n[:,1], "--", color=TARGET, lw=2., alpha=.85, zorder=1.4)
    ax.scatter(pts[:,0], pts[:,1], s=86, facecolors="none", edgecolors=TARGET, linewidths=2., zorder=4.5)

def _errs(ax, cur, tgt, color=ERR, z=7):
    for a, b in zip(cur, tgt):
        if np.hypot(*(b-a)) > 0.05:
            ax.annotate("", xy=b, xytext=a, zorder=z, arrowprops=dict(arrowstyle="-|>", color=color, lw=2., alpha=.9))

def _stage_ax(ax, xlim=(-7.1,7.1), ylim=(-2.75,2.95)):
    ax.set_facecolor(BG); ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.axis("off"); ax.axhline(-2.1, color=GRID, lw=1.2, zorder=0)

def _scene(ax, n, lp, rp, samp, tnodes=None, tpts=None, err_to=None, x_to=None):
    _stage_ax(ax)
    if tnodes is not None: _ghost(ax, tnodes, tpts)
    _arm(ax, ARM_BASE_L, np.asarray(lp)[:2], ARM_L, -1); _arm(ax, ARM_BASE_R, np.asarray(rp)[:2], ARM_R, +1)
    _rod(ax, n); _samp(ax, samp)
    _grip(ax, np.asarray(lp)[:2], lp[2], -1, ARM_L); _grip(ax, np.asarray(rp)[:2], rp[2], +1, ARM_R)
    if err_to is not None: _errs(ax, samp, err_to)
    if x_to is not None:
        ax.scatter(x_to[:,0], x_to[:,1], marker="X", s=130, color=ERR, edgecolors=BG, linewidths=1.2, zorder=8)
        _errs(ax, samp, x_to)

def _dark_panel(ax):
    ax.set_facecolor("#131c2b")
    for sp in ax.spines.values(): sp.set_color(MUTE)
    ax.tick_params(colors=INK, labelsize=9); ax.grid(True, color=GRID); ax.set_axisbelow(True)

ANIM_DPI = 70   # render animation frames a touch coarser to keep the embedded notebook light

def _embed(anim, fig):
    fig.set_dpi(ANIM_DPI)
    html = HTML(anim.to_jshtml(default_mode="loop")); plt.close(fig); return html

# --------------------------------------------------------------------------- #
#  setup figure + block diagrams
# --------------------------------------------------------------------------- #
def show_setup(rod, left0, right0, target_grasp):
    """Labelled stage: rod, samples (state s), target (s*), and the two grippers."""
    n0 = rod.solve(left0, right0, warm_start=False); P0 = rod.sample_points(n0)
    tn = rod.solve(*target_grasp, warm_start=False); tp = rod.sample_points(tn)
    fig, ax = plt.subplots(figsize=(9.2,4.8)); fig.patch.set_facecolor(BG)
    _scene(ax, n0, left0, right0, P0, tn, tp)
    def lab(t, xy, xytext, color):
        ax.annotate(t, xy=xy, xytext=xytext, color=color, fontsize=9.5, fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.2, alpha=0.85))
    lab("elastic rod  (the deformable object)", n0[12], (0.2,2.6), ROD)
    lab(r"sampled points  $\to$  feature state $s$", P0[1], (-3.6,2.4), SAMPLE)
    lab(r"target shape  $s^{*}$", tp[2], (3.4,-1.4), TARGET)
    lab(r"left gripper pose $(x_L,y_L,\theta_L)$", left0[:2], (-4.6,1.5), ARM_L)
    lab(r"right gripper $(x_R,y_R,\theta_R)$", right0[:2], (5.0,1.6), ARM_R)
    plt.show()

def _box(ax, cx, cy, w, h, text, fc, ec, fs=10):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INKL, zorder=4)

def _arrow(ax, p0, p1, text="", off=(0,.18), color="#3a4654", fs=9, tcol=None, rad=0.):
    cs = f"arc3,rad={rad}" if rad else None
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=1.8, color=color,
                 zorder=2, shrinkA=0, shrinkB=0, connectionstyle=cs))
    if text:
        ax.text((p0[0]+p1[0])/2+off[0], (p0[1]+p1[1])/2+off[1], text, ha="center", va="bottom", fontsize=fs, color=tcol or INKL)

_WIRE = "#3a4654"; F_GREY = "#F2F4F6"

def _sumj(ax, cx, cy, r=0.30, signs=()):
    """A round summing junction with cross-hairs and +/- signs."""
    ax.add_patch(plt.Circle((cx,cy), r, facecolor="white", edgecolor=INKL, lw=1.8, zorder=3))
    ax.plot([cx-0.14,cx+0.14],[cy,cy], color=INKL, lw=1.1, zorder=4)
    ax.plot([cx,cx],[cy-0.14,cy+0.14], color=INKL, lw=1.1, zorder=4)
    for s, where in signs:
        if where == "left":   ax.text(cx-r-0.16, cy+0.18, s, fontsize=12, color=INKL, ha="center")
        if where == "top":    ax.text(cx-0.30, cy+r+0.06, s, fontsize=12, color=INKL, ha="center")
        if where == "bottom": ax.text(cx-0.30, cy-r-0.30, s, fontsize=13, color=INKL, ha="center")

def _straight(ax, p0, p1, text="", off=(0,0.16), fs=9, tcol=None, color=_WIRE):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=1.8, color=color, zorder=2, shrinkA=0, shrinkB=0))
    if text: ax.text((p0[0]+p1[0])/2+off[0], (p0[1]+p1[1])/2+off[1], text, ha="center", va="bottom", fontsize=fs, color=tcol or INKL)

def _wire(ax, pts, color=_WIRE, lw=1.8, arrow=True):
    pts = np.asarray(pts, float)
    for a, b in zip(pts[:-1], pts[1:]):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-", lw=lw, color=color, zorder=2, shrinkA=0, shrinkB=0))
    if arrow:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=14, lw=lw, color=color, zorder=2, shrinkA=0, shrinkB=0))

def _dot(ax, p, color=_WIRE): ax.plot([p[0]],[p[1]], marker="o", ms=5, color=color, zorder=5)

def online_loop_diagram():
    """Block diagram of the online control loop, with the first-difference (z^-1) Delta-s path."""
    fig, ax = plt.subplots(figsize=(13.6,4.2)); ax.set_aspect("equal"); ax.set_xlim(-0.3,14.7); ax.set_ylim(-3.25,0.9); ax.axis("off")
    _sumj(ax, 1.3, 0, signs=(("+","left"),("-","bottom")))
    ax.text(0.1, 0, r"$s^{*}$", fontsize=14, ha="center", va="center", color=L_GREEN, fontweight="bold")
    _straight(ax, (0.42,0), (1.0,0))
    _box(ax, 3.6,0, 2.2,0.95, "Controller\n"+r"$\Delta r=\alpha\,\hat J^{+} e$", F_BLUE, L_BLUE)
    _box(ax, 6.2,0, 1.5,0.95, "Robots", F_STEEL, STEEL)
    _box(ax, 8.6,0, 2.0,0.95, "Deformable\nrod (plant)", F_ORANGE, L_ORANGE)
    _box(ax, 11.1,0, 2.3,0.95, "Perception", F_GREEN, L_GREEN)
    _straight(ax, (1.6,0), (2.50,0), r"$e=s^{*}\!-\!s$")
    _straight(ax, (4.70,0), (5.45,0), r"$\Delta r$")
    _straight(ax, (6.95,0), (7.60,0), r"$r$")
    _straight(ax, (9.60,0), (10.0,0), r"$x$")
    _straight(ax, (12.25,0), (12.8,0), r"$s$")
    yfb = -1.05
    _dot(ax, (12.55,0)); _wire(ax, [(12.55,0),(12.55,yfb),(1.3,yfb),(1.3,-0.30)])
    ax.text(6.9, yfb-0.27, r"measured shape $s$", fontsize=8.5, color=STEEL, ha="center")
    yE = -2.45; estC = 5.5
    _box(ax, estC,yE, 3.6,0.95, "Jacobian estimator\n"+r"online Broyden update (init. $\hat J^{(0)}$)", F_PURPLE, PURPLE, fs=9.5)
    _wire(ax, [(3.95,yE+0.475),(3.95,-0.475)]); ax.text(3.72,-1.55, r"$\hat J$", fontsize=11, color=PURPLE)
    _dot(ax, (5.05,0)); _wire(ax, [(5.05,0),(5.05,yE+0.475)]); ax.text(5.24,-1.5, r"$\Delta r$", fontsize=9, color=PURPLE)
    _sumj(ax, 13.25,-1.5, r=0.27, signs=(("+","left"),("-","top")))
    _box(ax, 13.25,-0.62, 0.82,0.5, r"$z^{-1}$", F_GREY, STEEL, fs=11)
    _dot(ax, (12.95,0)); _wire(ax, [(12.95,0),(12.95,-1.5),(12.98,-1.5)])
    _dot(ax, (13.7,0));  _wire(ax, [(13.7,0),(13.7,-0.62),(13.66,-0.62)])
    _wire(ax, [(13.25,-0.87),(13.25,-1.23)])
    _wire(ax, [(13.25,-1.77),(13.25,yE),(estC+1.8,yE)])
    ax.text(10.6, yE+0.16, r"$\Delta s=s_k-s_{k-1}$", fontsize=9, color=PURPLE, ha="center")
    plt.tight_layout(); plt.show()

def offline_loop_diagram():
    """Block diagram of the offline initialization. The stack receives both Delta r_k and Delta s_k."""
    fig, ax = plt.subplots(figsize=(12.4,2.9)); ax.set_aspect("equal"); ax.set_xlim(-0.3,13.6); ax.set_ylim(-1.7,1.7); ax.axis("off")
    _box(ax, 2.4,0, 3.0,1.0, "Exploration\ncommand "+r"$\Delta r_k$"+"\n(one DoF at a time)", F_BLUE, L_BLUE, fs=9.5)
    _box(ax, 6.7,0, 3.0,1.0, "Robots, rod &\nperception\nmeasure "+r"$\Delta s_k$", F_GREEN, L_GREEN, fs=9.5)
    _box(ax, 11.0,0, 3.2,1.0, "Stack "+r"$\Delta R,\Delta S$"+"\nleast-squares fit\n"+r"$\hat J^{(0)}=\Delta S\,\Delta R^{+}$", F_PURPLE, PURPLE, fs=9.5)
    _straight(ax, (3.9,0), (5.2,0), r"$\Delta r_k$")
    _straight(ax, (8.2,0), (9.4,0), r"$\Delta s_k$")
    _straight(ax, (12.6,0), (13.3,0), r"$\hat J^{(0)}$", tcol=PURPLE)
    _dot(ax, (2.4,-0.5)); _wire(ax, [(2.4,-0.5),(2.4,-1.15),(11.0,-1.15),(11.0,-0.5)])
    ax.text(6.7, -1.05, r"$\Delta r_k$  (stacked into $\Delta R$)", fontsize=9, color=L_BLUE, ha="center")
    ax.add_patch(FancyArrowPatch((6.7,0.55), (2.4,0.55), arrowstyle="-|>", mutation_scale=12, lw=1.5, color=STEEL, zorder=2, connectionstyle="arc3,rad=0.45"))
    ax.text(4.55, 1.28, r"repeat for $k=1,\dots,24$", fontsize=9.5, color=STEEL, ha="center")
    plt.tight_layout(); plt.show()

# --------------------------------------------------------------------------- #
#  shape representations
# --------------------------------------------------------------------------- #
def show_representations(P):
    """Draw the three shape descriptors on the same sampled shape (P points)."""
    P = np.asarray(P, float)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a in ax:
        a.set_aspect("equal"); a.grid(True, color="0.93"); a.set_axisbelow(True); a.tick_params(labelsize=8)
    # 1) point coordinates
    ax[0].plot(*P.T, "-", color="0.7", lw=1.6, zorder=1)
    ax[0].scatter(*P.T, s=70, color=L_GOLD, edgecolors=INKL, zorder=3)
    for i, p in enumerate(P):
        ax[0].annotate(rf"$p_{i+1}$", p, textcoords="offset points", xytext=(4,7), fontsize=8, color=INKL)
    ax[0].annotate(r"$(x_3,y_3)$", P[2], textcoords="offset points", xytext=(6,-16), fontsize=9, color=L_GOLD)
    ax[0].set_title("point coordinates  "+r"$(x_i,y_i)$", color=INKL, fontweight="bold", fontsize=11)
    # 2) edge vectors
    ax[1].plot(*P.T, "-", color="0.85", lw=1.2, zorder=1)
    ax[1].scatter(*P.T, s=46, color="0.6", edgecolors=INKL, zorder=3)
    for i in range(len(P)-1):
        ax[1].annotate("", xy=P[i+1], xytext=P[i], zorder=2,
                       arrowprops=dict(arrowstyle="-|>", color=L_ORANGE, lw=2.4))
    mid = (P[0]+P[1])/2
    ax[1].annotate(r"$e_i=p_{i+1}-p_i$", mid, textcoords="offset points", xytext=(2,10), fontsize=9, color=L_ORANGE)
    ax[1].set_title("edge vectors  "+r"$e_i=p_{i+1}-p_i$", color=INKL, fontweight="bold", fontsize=11)
    # 3) curvature (turning angles)
    ax[2].plot(*P.T, "-", color="0.8", lw=1.6, zorder=1)
    ax[2].scatter(*P.T, s=46, color="0.6", edgecolors=INKL, zorder=3)
    for i in range(1, len(P)-1):
        din = np.arctan2(*(P[i]-P[i-1])[::-1]); dout = np.arctan2(*(P[i+1]-P[i])[::-1])
        a0, a1 = np.degrees(din), np.degrees(dout)
        # dashed continuation of the incoming edge
        ext = P[i] + 0.5*np.array([np.cos(din), np.sin(din)])
        ax[2].plot([P[i][0], ext[0]], [P[i][1], ext[1]], ":", color=PURPLE, lw=1.2, alpha=0.7, zorder=2)
        lo, hi = sorted([a0, a1])
        ax[2].add_patch(Arc(P[i], 0.6, 0.6, angle=0, theta1=lo, theta2=hi, color=PURPLE, lw=2.2, zorder=3))
        mang = np.radians((a0+a1)/2)
        ax[2].text(P[i][0]+0.42*np.cos(mang), P[i][1]+0.42*np.sin(mang), rf"$\kappa_{i}$", color=PURPLE, fontsize=9, ha="center")
    ax[2].set_title("curvature  "+r"$\kappa_i=\theta_{i+1}-\theta_i$", color=INKL, fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.show()

def compare_representations(U, U_moved, names, vals):
    """Left: a shape before/after a rigid motion. Right: how much each rep changes."""
    U, U_moved = np.asarray(U, float), np.asarray(U_moved, float)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axL.set_aspect("equal"); axL.grid(True, color="0.92"); axL.set_axisbelow(True)
    axL.plot(*U.T, "-o", color=L_ORANGE, lw=2.6, ms=7, label="original")
    axL.plot(*U_moved.T, "-o", color=L_BLUE, lw=2.6, ms=7, label="after rigid motion")
    axL.set_title("Same shape, moved rigidly", color=INKL, fontweight="bold"); axL.legend(fontsize=9)
    bars = axR.bar(list(names), vals, color=[L_ORANGE, L_GOLD, L_GREEN], edgecolor=INKL)
    axR.set_title(r"Change $\|\Delta s\|$ under the rigid motion", color=INKL, fontweight="bold")
    axR.grid(True, axis="y", color="0.92"); axR.set_axisbelow(True)
    for b, v in zip(bars, vals):
        axR.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.show()

def show_representation_servo(n0, nT, finals):
    """Overlay the target and the shapes the controller reaches under each representation.

    ``finals`` is a dict {name: nodes}; colours and line styles are assigned here.
    """
    styles = [(L_ORANGE, "-"), (L_BLUE, "--"), (PURPLE, ":")]
    fig, ax = plt.subplots(figsize=(8.6, 4.2)); ax.set_aspect("equal")
    ax.grid(True, color="0.93"); ax.set_axisbelow(True)
    ax.plot(*n0.T, color="0.75", lw=2., label="initial")
    ax.plot(*nT.T, color=L_GREEN, lw=3.4, label="target", alpha=0.9)
    for k, (name, n) in enumerate(finals.items()):
        c, ls = styles[k % len(styles)]
        ax.plot(*n.T, color=c, lw=2.6, ls=ls, label=f"reached with {name}")
    ax.set_title("Same target, two feature spaces: the controller matches what the features see",
                 color=INKL, fontweight="bold")
    ax.legend(fontsize=9, loc="upper center", ncol=2); plt.tight_layout(); plt.show()

def animate_feature_spaces(rod, runs, target_grasp, n_frames=22):
    """Animate the controller run under several feature representations at once.

    Top row: the rod reshaping toward the (rotated) target under each representation. Bottom row:
    each representation's own feature error (all converge), the geometric error (only point
    coordinates reach zero), and the commanded action norm. ``runs`` is a dict
    {name: (R_hist, node_hist, feat_errs)}, where ``feat_errs`` is the error in that feature space.
    """
    from matplotlib.gridspec import GridSpec
    tnodes = rod.solve(*target_grasp, warm_start=False); PT = rod.sample_points(tnodes)
    cols = [L_ORANGE, L_GOLD, L_BLUE]; data = {}
    for (name, (R_hist, node_hist, ferr)), col in zip(runs.items(), cols):
        L = len(node_hist)
        sel = np.unique(np.linspace(0, L-1, n_frames).astype(int))
        if len(sel) < n_frames: sel = np.concatenate([sel, np.full(n_frames-len(sel), sel[-1])])
        samples = [rod.sample_points(n) for n in node_hist]
        geom = np.array([np.linalg.norm(s - PT) for s in samples])
        act = np.array([0.0] + [np.linalg.norm(R_hist[i]-R_hist[i-1]) for i in range(1, L)])
        ferr = np.asarray(ferr, float); ferr = ferr / max(ferr[0], 1e-9)        # normalise to start at 1
        poses = [split_grasp(R_hist[i]) for i in range(L)]
        data[name] = dict(col=col, sel=sel, nodes=node_hist, poses=poses, samples=samples,
                          geom=geom, act=act, ferr=ferr, L=L)
    gmax = max(d["geom"].max() for d in data.values()); amax = max(d["act"].max() for d in data.values())
    Lmax = max(d["L"] for d in data.values()); names = list(data)
    fig = plt.figure(figsize=(12.8, 6.6)); fig.patch.set_facecolor(BG)
    gs = GridSpec(2, 3, height_ratios=[1.3, 1.0], hspace=0.36, wspace=0.32, figure=fig)
    axst = [fig.add_subplot(gs[0,j]) for j in range(3)]
    axf, axe, axa = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[1,2])
    def _curve(ax, key, i, name):
        d = data[name]; ax.plot(np.arange(i+1), d[key][:i+1], color=d["col"], lw=2.2, label=name)
        ax.scatter([i],[d[key][i]], color=d["col"], s=26, zorder=5, edgecolors=BG)
    def _f(k):
        for ax, name in zip(axst, names):
            d = data[name]; i = d["sel"][k]
            ax.clear(); _scene(ax, d["nodes"][i], *d["poses"][i], d["samples"][i], tnodes, PT, err_to=PT)
            ax.set_title(name, color=d["col"], fontsize=12, fontweight="bold", pad=6)
        axf.clear(); _dark_panel(axf)
        for name in names: _curve(axf, "ferr", data[name]["sel"][k], name)
        axf.set_xlim(0, Lmax); axf.set_ylim(0, 1.05); axf.set_xlabel("iteration", color=INK)
        axf.set_ylabel("feature error (rel. to start)", color=INK)
        axf.set_title("each converges in its own feature space", color=INK, fontsize=10.5, fontweight="bold")
        axf.legend(fontsize=7.5, facecolor="#131c2b", edgecolor=MUTE, labelcolor=INK, loc="upper right")
        axe.clear(); _dark_panel(axe)
        for name in names: _curve(axe, "geom", data[name]["sel"][k], name)
        axe.set_xlim(0, Lmax); axe.set_ylim(0, gmax*1.05); axe.set_xlabel("iteration", color=INK)
        axe.set_ylabel(r"geometric error $\|P-P^{*}\|$", color=INK)
        axe.set_title("but only point coords reaches the target", color=INK, fontsize=10.5, fontweight="bold")
        axa.clear(); _dark_panel(axa)
        for name in names: _curve(axa, "act", data[name]["sel"][k], name)
        axa.set_xlim(0, Lmax); axa.set_ylim(0, amax*1.05); axa.set_xlabel("iteration", color=INK)
        axa.set_ylabel(r"action norm $\|\Delta r\|$", color=INK)
        axa.set_title("commanded gripper motion per step", color=INK, fontsize=10.5, fontweight="bold")
    return _embed(animation.FuncAnimation(fig, _f, frames=n_frames, interval=120), fig)

# --------------------------------------------------------------------------- #
#  interaction Jacobian
# --------------------------------------------------------------------------- #
def animate_probe(rod, r0):
    """Move each gripper DoF in turn; show the displacement of every sampled point
    and the corresponding column of J lighting up."""
    P0 = rod.sample_points(rod.solve(*split_grasp(r0), warm_start=False))
    probe = np.array([0.55,0.55,0.33,0.55,0.55,0.33]); seqs = []
    for j in range(6):
        fr = []
        for t in np.sin(np.linspace(0, np.pi, 7)):
            r = r0.copy(); r[j] += t*probe[j]; lp, rp = split_grasp(r); n = rod.solve(lp, rp, warm_start=True)
            fr.append((n, lp, rp, rod.sample_points(n)))
        seqs.append(fr)
    frames = [(j,k) for j in range(6) for k in range(7)]
    Jref = rod.true_jacobian(*split_grasp(r0)); vmax = float(np.abs(Jref).max())
    fig, (axS, axJ) = plt.subplots(1, 2, figsize=(11.5,4.6), gridspec_kw={"width_ratios":[1.5,1]}); fig.patch.set_facecolor(BG)
    def _f(idx):
        j, k = frames[idx]; n, lp, rp, samp = seqs[j][k]
        axS.clear(); _scene(axS, n, lp, rp, samp); _samp(axS, P0, color=MUTE, s=22)
        for a, b in zip(P0, samp):
            if np.hypot(*(b-a)) > 0.02: axS.annotate("", xy=b, xytext=a, zorder=8, arrowprops=dict(arrowstyle="-|>", color=SAMPLE, lw=2.8))
        axS.set_title(f"move gripper DoF  {DOF_LABELS[j]}", color=INK, fontsize=13, fontweight="bold", pad=8)
        axJ.clear(); axJ.imshow(Jref, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        axJ.add_patch(plt.Rectangle((j-.5,-.5), 1, 10, fill=False, edgecolor="white", lw=3))
        axJ.set_xticks(range(6)); axJ.set_xticklabels(DOF_LABELS, fontsize=9); axJ.tick_params(colors=INK); axJ.set_yticks([])
        axJ.set_ylabel("feature index", color=INK); axJ.set_title("column of  J  for this DoF", color=INK, fontsize=12, fontweight="bold")
    return _embed(animation.FuncAnimation(fig, _f, frames=len(frames), interval=80), fig)

def plot_jacobian(J, S, kappa):
    """Heat-map of the estimate and its singular values."""
    J, S = np.asarray(J, float), np.asarray(S, float)
    fig, ax = plt.subplots(1, 2, figsize=(10.5,4.5)); fig.patch.set_facecolor("white")
    v = float(np.abs(J).max()); im = ax[0].imshow(J, cmap="coolwarm", vmin=-v, vmax=v)
    for (i,j), val in np.ndenumerate(J): ax[0].text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="0.12")
    ax[0].set_xticks(range(6)); ax[0].set_xticklabels(DOF_LABELS); ax[0].set_yticks(range(10)); ax[0].set_ylabel("feature index")
    ax[0].set_title(r"estimate $\hat J^{(0)}$  (column per DoF)", color=INKL, fontweight="bold"); ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[1].bar(range(1,7), S, color=L_BLUE, edgecolor=INKL)
    ax[1].set_title(rf"singular values     $\kappa$ = {kappa:.1f}", color=INKL, fontweight="bold")
    ax[1].set_xlabel("channel i"); ax[1].set_ylabel(r"$\sigma_i$"); ax[1].grid(True, axis="y", color="0.92"); ax[1].set_axisbelow(True)
    plt.tight_layout(); plt.show()

# --------------------------------------------------------------------------- #
#  closing the loop
# --------------------------------------------------------------------------- #
def preview_goal(rod, initial_grasp, target_grasp, n_steps=30):
    """Teaser: interpolate the grasp from initial to target and animate the rod."""
    from dlo import grasp_vector
    r0 = grasp_vector(*[np.asarray(x, float) for x in [initial_grasp[0], initial_grasp[1]]]) \
        if False else grasp_vector(initial_grasp[0], initial_grasp[1])
    rT = grasp_vector(target_grasp[0], target_grasp[1])
    tn = rod.solve(*target_grasp, warm_start=False); tp = rod.sample_points(tn)
    ts = np.concatenate([np.linspace(0,1,n_steps), np.full(6,1.0)]); N, Po, Sa = [], [], []
    for t in ts:
        lp, rp = split_grasp((1-t)*r0 + t*rT); n = rod.solve(lp, rp, warm_start=(t>0))
        N.append(n); Po.append((lp,rp)); Sa.append(rod.sample_points(n))
    fig, ax = plt.subplots(figsize=(8.8,4.4)); fig.patch.set_facecolor(BG)
    def _f(i):
        ax.clear(); lp, rp = Po[i]; _scene(ax, N[i], lp, rp, Sa[i], tn, tp, err_to=tp)
        ax.set_title("Goal:  reshape the rod onto the green target", color=INK, fontsize=13, fontweight="bold", pad=10)
    return _embed(animation.FuncAnimation(fig, _f, frames=len(ts), interval=90), fig)

def animate_reshape(rod, R_hist, node_hist, errs, *, target_grasp=None, desired_pts=None,
                    overlay=None, floor=None, title="reshaping the rod", n_frames=30):
    """Animate a control run: the arms reshaping the rod (left) and the error (right).

    target_grasp : (lp, rp) of a reachable target -> drawn as a green ghost.
    desired_pts  : (P,2) of an unreachable target -> drawn as red X markers.
    overlay      : (errs2, label) a second error curve (e.g. a reachable comparison).
    floor        : y-value of a dotted reference line (e.g. the unreachable component).
    """
    sel = np.unique(np.linspace(0, len(node_hist)-1, n_frames).astype(int))
    poses = [split_grasp(R_hist[i]) for i in sel]; samps = [rod.sample_points(node_hist[i]) for i in sel]
    tn = tp = None
    if target_grasp is not None:
        tn = rod.solve(*target_grasp, warm_start=False); tp = rod.sample_points(tn)
    main_col = ERR if desired_pts is not None else ROD
    xmax = max(len(errs), len(overlay[0]) if overlay else 0)
    fig, (axS, axE) = plt.subplots(1, 2, figsize=(12,4.4), gridspec_kw={"width_ratios":[1.55,1]}); fig.patch.set_facecolor(BG)
    def _f(k):
        i = sel[k]; lp, rp = poses[k]
        axS.clear()
        _scene(axS, node_hist[i], lp, rp, samps[k], tn, tp,
               err_to=(tp if (target_grasp is not None and desired_pts is None) else None),
               x_to=desired_pts)
        axS.set_title(f"{title}    (iteration {i})", color=INK, fontsize=12.5, fontweight="bold", pad=8)
        axE.clear(); _dark_panel(axE)
        if overlay is not None: axE.plot(overlay[0], color=TARGET, lw=2.2, label=overlay[1])
        axE.plot(np.arange(i+1), errs[:i+1], color=main_col, lw=2.6,
                 label=("unreachable target" if desired_pts is not None else None))
        axE.scatter([i],[errs[i]], color=main_col, s=44, zorder=5, edgecolors=BG)
        if floor is not None: axE.axhline(floor, color=ERR, ls=":", lw=1.5)
        axE.set_xlim(0, xmax); axE.set_ylim(0, max(errs[0], overlay[0][0] if overlay else 0)*1.05)
        axE.set_xlabel("iteration", color=INK); axE.set_ylabel(r"$\|s^{*}-s\|$", color=INK)
        axE.set_title("shape error", color=INK, fontsize=12, fontweight="bold")
        if overlay is not None or desired_pts is not None:
            axE.legend(fontsize=8, facecolor="#131c2b", edgecolor=MUTE, labelcolor=INK, loc="upper right")
    return _embed(animation.FuncAnimation(fig, _f, frames=len(sel), interval=110), fig)

def plot_convergence_compare(errs_on, errs_off):
    fig, ax = plt.subplots(figsize=(6.8,3.6)); fig.patch.set_facecolor("white")
    ax.plot(errs_on, color=L_ORANGE, lw=2.2, label=f"with online update   ({len(errs_on)-1} iters)")
    ax.plot(errs_off, color=L_BLUE, lw=2.2, ls="--", label=f"fixed offline estimate   ({len(errs_off)-1} iters)")
    ax.set_title("Here the online update speeds convergence appreciably", color=INKL, fontweight="bold")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\|s^{*}-s\|$"); ax.set_ylim(bottom=0)
    ax.legend(); ax.grid(True, color="0.92"); ax.set_axisbelow(True); plt.tight_layout(); plt.show()

# --------------------------------------------------------------------------- #
#  failure modes
# --------------------------------------------------------------------------- #
def plot_failure_modes(seps_taut, smin, w, kap, deltas, full, feat, seps_buck, lam, cross):
    fig, ax = plt.subplots(1, 3, figsize=(13.5,3.9)); fig.patch.set_facecolor("white")
    ax[0].plot(seps_taut, smin, "-o", color=L_BLUE, ms=3, label=r"$\sigma_{\min}$"); ax[0].plot(seps_taut, w, "-s", color=L_ORANGE, ms=3, label=r"$w$")
    axk = ax[0].twinx(); axk.plot(seps_taut, kap, "--", color="0.45"); axk.set_ylabel(r"$\kappa$", color="0.45")
    ax[0].set_title("(1) Controllability: taut state", color=INKL, fontweight="bold"); ax[0].set_xlabel("gripper separation")
    ax[0].set_ylabel(r"$\sigma_{\min},\; w$"); ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(deltas, full, "-o", color=L_ORANGE, ms=3, label=r"full state $\|\Delta s\|$"); ax[1].plot(deltas, feat, "-s", color=L_BLUE, ms=3, label="midpoint-height feature")
    ax[1].set_title("(2) Observability: a blind feature", color=INKL, fontweight="bold"); ax[1].set_xlabel(r"antisymmetric motion $\delta$"); ax[1].set_ylabel(r"$|\Delta s|$"); ax[1].legend(fontsize=8)
    ax[2].plot(seps_buck, lam, "-o", color=L_TEAL, ms=3); ax[2].axhline(0, color="0.6", lw=1); ax[2].plot(seps_buck[cross], lam[cross], "o", color=L_RED, ms=9, label="buckling onset")
    ax[2].set_title("(3) Mechanical instability: buckling", color=INKL, fontweight="bold"); ax[2].set_xlabel("separation (smaller = more compression)"); ax[2].set_ylabel(r"$\lambda_{\min}(H)$"); ax[2].invert_xaxis(); ax[2].legend(fontsize=8)
    for a in ax: a.grid(True, color="0.92"); a.set_axisbelow(True)
    plt.tight_layout(); plt.show()

def _kappa(J):
    S = np.linalg.svd(J, compute_uv=False); return float(S[0]/S[-1]) if S[-1] > 0 else np.inf

def animate_taut(rod):
    seps = np.concatenate([np.linspace(6.,7.2,22), np.full(4,7.2)]); kap = []; TN, TP = [], []
    for sep in seps:
        lp, rp = [-sep/2,1.,0.], [sep/2,1.,0.]; n = rod.solve(lp, rp, warm_start=True); TN.append(n); TP.append(rod.sample_points(n)); kap.append(_kappa(rod.true_jacobian(lp, rp)))
    kap = np.array(kap)
    fig, (axS, axK) = plt.subplots(1, 2, figsize=(12,4.2), gridspec_kw={"width_ratios":[1.6,1]}); fig.patch.set_facecolor(BG)
    def _f(i):
        sep = seps[i]; lp, rp = [-sep/2,1.,0.], [sep/2,1.,0.]
        axS.clear(); _scene(axS, TN[i], lp, rp, TP[i]); axS.set_title(f"gripper separation = {sep:.2f}", color=INK, fontsize=13, fontweight="bold", pad=8)
        axK.clear(); _dark_panel(axK); axK.plot(seps[:i+1], kap[:i+1], color=ERR, lw=2.6); axK.scatter([sep],[kap[i]], color=ERR, s=44, zorder=5, edgecolors=BG)
        axK.set_xlim(6.,7.25); axK.set_ylim(0, kap.max()*1.08); axK.set_xlabel("separation", color=INK); axK.set_ylabel(r"condition number $\kappa$", color=ERR)
        axK.set_title("the loop loses controllability", color=INK, fontsize=12, fontweight="bold")
    return _embed(animation.FuncAnimation(fig, _f, frames=len(seps), interval=110), fig)

def animate_buckle(rod):
    seps = np.linspace(6.4,4.7,26); lam = np.array([rod.straight_min_eig(sep, y=1.) for sep in seps]); BN, BP = [], []
    rod._warm = None
    for sep in seps:
        lp, rp = [-sep/2,1.,0.], [sep/2,1.,0.]; n = rod.solve(lp, rp, warm_start=True); BN.append(n); BP.append(rod.sample_points(n))
    fig, (axS, axL) = plt.subplots(1, 2, figsize=(12,4.2), gridspec_kw={"width_ratios":[1.6,1]}); fig.patch.set_facecolor(BG)
    def _f(i):
        sep = seps[i]; lp, rp = [-sep/2,1.,0.], [sep/2,1.,0.]
        axS.clear(); _scene(axS, BN[i], lp, rp, BP[i])
        st = "stable" if lam[i] > 0 else "BUCKLED"; col = TARGET if lam[i] > 0 else ERR
        axS.set_title(f"separation = {sep:.2f}   ({st})", color=col, fontsize=13, fontweight="bold", pad=8)
        axL.clear(); _dark_panel(axL); axL.plot(seps[:i+1], lam[:i+1], color=TARGET, lw=2.6); axL.axhline(0, color=MUTE, lw=1)
        axL.scatter([sep],[lam[i]], color=ERR, s=46, zorder=5, edgecolors=BG)
        axL.set_xlim(seps[0]+.05, seps[-1]-.05); axL.set_ylim(lam.min()-.3, lam.max()+.3)
        axL.set_xlabel("separation (decreasing)", color=INK); axL.set_ylabel(r"$\lambda_{\min}(H_{\rm energy})$", color=INK)
        axL.set_title("energy Hessian eigenvalue", color=INK, fontsize=12, fontweight="bold")
    return _embed(animation.FuncAnimation(fig, _f, frames=len(seps), interval=120), fig)

def show_observability(rod_sym):
    fig, ax = plt.subplots(figsize=(7.4,3.2)); ax.set_aspect("equal"); ax.set_facecolor("white")
    base = rod_sym.solve([-3.,1.,0.],[3.,1.,0.], warm_start=False); ax.plot(*base.T, color="0.6", lw=2.2, label=r"$\delta=0$")
    for d, c in [(0.4,L_ORANGE),(0.8,L_BLUE)]:
        ax.plot(*rod_sym.solve([-3.,1.+d,0.],[3.,1.-d,0.], warm_start=False).T, lw=2.4, color=c, label=rf"$\delta={d}$")
    ax.axhline(1.0, color=L_GREEN, ls="--", lw=1.6); ax.scatter([0.],[1.], s=95, color=L_GREEN, edgecolors=INKL, zorder=6, label="midpoint feature (fixed)")
    ax.set_title("Observability: the rod deforms, a symmetric feature does not", color=INKL, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, color="0.92"); ax.set_axisbelow(True); plt.tight_layout(); plt.show()
