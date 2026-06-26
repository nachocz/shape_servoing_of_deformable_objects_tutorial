"""
Shared plotting helpers for the shape-servoing lab.
===================================================

Thin wrappers around matplotlib so every exercise produces a consistent,
readable figure with the same colour scheme as the accompanying notes:

    rod      -> orange      sampled state s   -> yellow
    target   -> green       left/right grip   -> blue / teal

You are encouraged to read these helpers, but you never need to edit them.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Colour scheme (matches the LaTeX notes).
ROD     = "#F88A35"   # current rod
SAMPLE  = "#E7C649"   # sampled feature points (the state s)
TARGET  = "#67C666"   # target rod / target state s*
GRIP_L  = "#56A1E8"   # left gripper
GRIP_R  = "#34C4A5"   # right gripper
INK     = "#1A3D6D"   # axis text / accents


def new_axes(title: str = "", size=(6.5, 5.0)):
    """A square-ish axis with equal aspect and a light grid."""
    fig, ax = plt.subplots(figsize=size)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontweight="bold")
    return fig, ax


def draw_rod(ax, nodes, color=ROD, lw=3.0, label=None, alpha=1.0, zorder=2):
    """Draw the rod centreline from its node positions (N, 2)."""
    ax.plot(nodes[:, 0], nodes[:, 1], "-", color=color, lw=lw,
            solid_capstyle="round", label=label, alpha=alpha, zorder=zorder)


def draw_samples(ax, pts, color=SAMPLE, s=90, label=None, zorder=4):
    """Draw the sampled feature points (P, 2), the measured state."""
    ax.scatter(pts[:, 0], pts[:, 1], s=s, color=color, edgecolors=INK,
               linewidths=0.8, label=label, zorder=zorder)


def draw_grippers(ax, left_pose, right_pose, size=170, zorder=5):
    """Mark the two gripper positions (the clamped rod ends)."""
    ax.scatter([left_pose[0]], [left_pose[1]], marker="s", s=size,
               color=GRIP_L, edgecolors=INK, linewidths=1.0, zorder=zorder)
    ax.scatter([right_pose[0]], [right_pose[1]], marker="s", s=size,
               color=GRIP_R, edgecolors=INK, linewidths=1.0, zorder=zorder)


def legend_below(ax, ncol=3, fontsize=8):
    """Place the legend in a strip *under* the axes so it never overlaps data."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol,
              fontsize=fontsize, frameon=True, borderaxespad=0.0)


def draw_matrix(ax, mat, title=None, xticklabels=None, cmap="coolwarm",
                vmax=None, fmt="{:.1f}", fontsize=6.5):
    """Render a matrix as an annotated heat-map (one cell per entry)."""
    mat = np.asarray(mat, float)
    vmax = vmax if vmax is not None else (np.abs(mat).max() or 1.0)
    ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, fmt.format(v), ha="center", va="center",
                fontsize=fontsize, color="0.12")
    ax.set_yticks([])
    if xticklabels is not None:
        ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(xticklabels,
                                                               fontsize=8)
    else:
        ax.set_xticks([])
    if title:
        ax.set_title(title, color=INK, fontsize=10, fontweight="bold")
    ax.set_anchor("N")


def draw_colvec(ax, v, title=None, cmap="coolwarm", vmax=None, fmt="{:.2f}",
                fontsize=7):
    """Render a vector as a single annotated column (the ds = J dr picture)."""
    v = np.asarray(v, float).reshape(-1, 1)
    vmax = vmax if vmax is not None else (np.abs(v).max() or 1.0)
    ax.imshow(v, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    for i, val in enumerate(v[:, 0]):
        ax.text(0, i, fmt.format(val), ha="center", va="center",
                fontsize=fontsize, color="0.12")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, color=INK, fontsize=10, fontweight="bold")
    ax.set_anchor("N")


def op_label(ax, symbol):
    """A bare operator (=, *, ...) used as a spacer between matrix/vector axes."""
    ax.axis("off")
    ax.text(0.5, 0.5, symbol, ha="center", va="center", fontsize=20,
            color=INK, transform=ax.transAxes)


def finish(fig, save_as: str | None = None, show: bool = True, tight: bool = True):
    """Tidy layout, optionally save to disk, and optionally display.

    ``show`` is silently skipped on non-interactive ("Agg") backends so the
    scripts also run cleanly headless. Pass ``tight=False`` for hand-laid-out
    gridspec figures where ``tight_layout`` would fight the manual spacing.
    """
    if tight:
        fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=130, bbox_inches="tight")
        print(f"[saved] {save_as}")
    if show and not plt.get_backend().lower().endswith("agg"):
        plt.show()
    plt.close(fig)


def save(fig, save_as: str, tight: bool = True):
    """Save a figure to disk *without* showing or closing it.

    The lab is one cumulative script that builds several figures and then
    displays them together at the end (see :func:`show_all`), so each part saves
    with this helper and leaves its window open. Pass ``tight=False`` for the
    hand-laid-out gridspec figure where ``tight_layout`` fights the spacing.
    """
    if tight:
        fig.tight_layout()
    fig.savefig(save_as, dpi=130, bbox_inches="tight")
    print(f"[saved] {save_as}")


def show_all():
    """Display every figure built so far, all at once.

    Silently skipped on non-interactive ("Agg") backends, so the script still
    runs headless and just writes the PNG files.
    """
    if not plt.get_backend().lower().endswith("agg"):
        plt.show()


# --------------------------------------------------------------------------- #
#  Higher-level lab figures (provided; the lab script just hands them the data)
# --------------------------------------------------------------------------- #
def plot_problem(nodes0, pts0, nodes_t, pts_t, left0, right0, err_norm, save_as):
    """Exercise 1: current shape, target, and the per-point error arrows."""
    fig, ax = new_axes(f"The control problem:  reduce  ||s* - s|| = {err_norm:.2f}  to 0")
    draw_rod(ax, nodes_t, color=TARGET, lw=2.0, alpha=0.9, label="target rod")
    draw_samples(ax, pts_t, color=TARGET, s=60, label="target state s*")
    draw_rod(ax, nodes0, label="current rod")
    draw_samples(ax, pts0, label="current state s")
    draw_grippers(ax, left0, right0)
    for p, q in zip(pts0, pts_t):
        ax.annotate("", xy=q, xytext=p,
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0))
    legend_below(ax, ncol=4, fontsize=8)
    save(fig, save_as)


def plot_representations(P, P_moved, names, vals, save_as):
    """Exercise 2: a shape under a rigid motion, and how much each rep reacts."""
    P, P_moved = np.asarray(P, float), np.asarray(P_moved, float)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    axL.set_aspect("equal"); axL.grid(True, color="0.9"); axL.set_axisbelow(True)
    axL.plot(*P.T, "-o", color=ROD, lw=2.5, ms=6, label="original")
    axL.plot(*P_moved.T, "-o", color=GRIP_L, lw=2.5, ms=6,
             label="after rigid transformation")
    axL.set_title("Same shape, rigid transformation\n(translation + rotation)",
                  color=INK, fontweight="bold")
    axL.legend(fontsize=9)
    bars = axR.bar(names, vals, color=[ROD, SAMPLE, TARGET], edgecolor=INK)
    axR.set_ylabel(r"$\|\Delta s\|$  after rigid transformation")
    axR.set_title("Change in each representation\nunder a rigid transformation",
                  color=INK, fontweight="bold")
    for b, v in zip(bars, vals):
        axR.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    save(fig, save_as)


def plot_jacobian(J, S, kappa, dof_labels, save_as):
    """Exercise 3: the estimated Jacobian as a heat-map and its singular values."""
    J, S = np.asarray(J, float), np.asarray(S, float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    v = float(np.abs(J).max()) or 1.0
    im = ax[0].imshow(J, cmap="coolwarm", vmin=-v, vmax=v)
    ax[0].set_title(r"estimate $\hat J$  (one column per DoF)", color=INK,
                    fontweight="bold")
    ax[0].set_xticks(range(len(dof_labels)))
    ax[0].set_xticklabels(dof_labels, fontsize=8)
    ax[0].set_ylabel("shape feature index")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[1].bar(range(1, len(S) + 1), S, color=GRIP_L, edgecolor=INK)
    ax[1].set_title(rf"singular values   $\kappa$ = {kappa:.1f}", color=INK,
                    fontweight="bold")
    ax[1].set_xlabel("channel i"); ax[1].set_ylabel(r"$\sigma_i$")
    ax[1].grid(True, axis="y", color="0.9"); ax[1].set_axisbelow(True)
    save(fig, save_as)


def plot_convergence(hist, actions, dof_labels, save_as):
    """Exercise 4: the control loop -- error decay and the per-iteration action."""
    hist, actions = np.asarray(hist, float), np.asarray(actions, float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(hist, "-", color=ROD, lw=2)
    ax[0].set_title("shape error vs. iteration", color=INK, fontweight="bold")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"$\|s^* - s\|$")
    ax[0].set_ylim(bottom=0); ax[0].grid(True, color="0.9")
    colors = [GRIP_L] * 3 + [GRIP_R] * 3
    styles = ["-", "--", ":"] * 2
    ncols = actions.shape[1] if (actions.ndim == 2 and actions.size) else 0
    for j in range(ncols):
        ax[1].plot(actions[:, j], styles[j], color=colors[j], lw=1.6,
                   label=dof_labels[j])
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_title(r"commanded action $\Delta r$ per iteration", color=INK,
                    fontweight="bold")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel(r"$\Delta r_j$")
    ax[1].grid(True, color="0.9")
    if ncols:
        ax[1].legend(ncol=2, fontsize=8)
    save(fig, save_as)


def plot_failure_modes(seps_taut, smin, w, kappa, deltas, full, feat,
                       seps_buck, lam, cross, save_as):
    """Exercise 5: the three failure signatures, side by side."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    ax[0].plot(seps_taut, smin, "-o", color=GRIP_L, ms=3, label=r"$\sigma_{\min}$")
    ax[0].plot(seps_taut, w, "-s", color=ROD, ms=3, label=r"$w$")
    ax[0].set_title("(1) Controllability: taut state", color=INK, fontweight="bold")
    ax[0].set_xlabel("gripper separation"); ax[0].set_ylabel(r"$\sigma_{\min}$, $w$")
    axk = ax[0].twinx()
    axk.plot(seps_taut, kappa, "--", color="0.45", label=r"$\kappa$")
    axk.set_ylabel(r"condition number $\kappa$", color="0.45")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(deltas, full, "-o", color=ROD, ms=3, label="full state $\\|\\Delta s\\|$")
    ax[1].plot(deltas, feat, "-s", color=GRIP_L, ms=3, label="midpoint-height feature")
    ax[1].set_title("(2) Observability: an insensitive feature", color=INK,
                    fontweight="bold")
    ax[1].set_xlabel("antisymmetric gripper motion $\\delta$")
    ax[1].set_ylabel(r"$|\Delta s|$"); ax[1].legend(fontsize=8)
    ax[2].plot(seps_buck, lam, "-o", color=GRIP_R, ms=3)
    ax[2].axhline(0, color="0.6", lw=1)
    ax[2].plot(seps_buck[cross], lam[cross], "o", color=ROD, ms=9,
               label="buckling onset")
    ax[2].set_title("(3) Mechanical instability: buckling", color=INK,
                    fontweight="bold")
    ax[2].set_xlabel("gripper separation  (decreasing = more compression)")
    ax[2].set_ylabel(r"$\lambda_{\min}(H_{\mathrm{energy}})$")
    ax[2].invert_xaxis(); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(True, color="0.9"); a.set_axisbelow(True)
    save(fig, save_as)
