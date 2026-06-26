"""
Exercise 5: the three ways the loop fails (stand-alone)
=======================================================
This is a separate, diagnostic study. Instead of driving the shape, you sweep the
rod through three singular situations and read each one's numerical signature. The
conditioning analysis uses the rod's `true_jacobian` oracle, a Jacobian the
model-free controller never has, used here only to reveal the mechanics.

This file reuses `svd_metrics` from Exercise 3. Run with:
    python ex5_failure_modes.py
"""
import numpy as np

from dlo import Rod
import labviz as viz
from ex3_offline_jacobian import svd_metrics


def taut_curve(rod, separations):
    """Case 1, controllability: pull the rod straight and track its conditioning.

    For each separation, return the smallest singular value, the manipulability,
    and the condition number of the Jacobian.
    """
    smin, w, kappa = [], [], []
    for sep in separations:
        # TODO (Exercise 5): build the straight grasp at this separation (left gripper
        #   at -sep/2, right gripper at +sep/2, both at height 1.0, angle 0); get its
        #   Jacobian from rod.true_jacobian(left, right); read sigma_min, w and kappa
        #   with svd_metrics; and append them to the three lists.
        smin.append(0.0); w.append(0.0); kappa.append(0.0)   # replace
    return np.array(smin), np.array(w), np.array(kappa)


def observability_curve(rod, deltas):
    """Case 2, observability: an antisymmetric motion that a symmetric feature misses.

    For each delta, return the full-state change and the midpoint-height change.
    """
    s0 = rod.shape_state([-3.0, 1.0, 0.0], [3.0, 1.0, 0.0], warm_start=False)
    full, feat = [], []
    for d in deltas:
        # TODO (Exercise 5): raise the left gripper by d and lower the right by d;
        #   measure the state with rod.shape_state(left, right); append the full-state
        #   change ||s - s0|| to `full` and the midpoint-height change |s[5] - s0[5]|
        #   to `feat`.
        full.append(0.0); feat.append(0.0)   # replace
    return np.array(full), np.array(feat)


def buckling_curve(rod, separations):
    """Case 3, mechanical instability: the smallest energy-Hessian eigenvalue.

    For each separation, return rod.straight_min_eig(sep, y=1.0); it crosses zero at
    the buckling onset.
    """
    # TODO (Exercise 5): query rod.straight_min_eig for each separation.
    return np.zeros(len(separations))


def main():
    rod = Rod()
    rod_sym = Rod(gravity=0.0)              # gravity-free rod, so the symmetry is exact

    seps_taut = np.linspace(6.0, 7.2, 18)
    smin, w, kappa = taut_curve(rod, seps_taut)
    deltas = np.linspace(0.0, 0.8, 18)
    full, feat = observability_curve(rod_sym, deltas)
    seps_buck = np.linspace(6.6, 4.8, 22)
    lam = buckling_curve(rod, seps_buck)
    cross = int(np.argmax(lam < 0))

    # ---- provided: print a summary and draw the three signatures (do not edit) --
    print(f"(1) taut:   sigma_min {smin[0]:.3f} -> {smin[-1]:.3f},  "
          f"kappa {kappa[0]:.0f} -> {kappa[-1]:.0f}")
    print(f"(2) observ: full-state ||ds|| up to {full[-1]:.2f}, "
          f"but the midpoint feature stays <= {feat.max():.4f}")
    print(f"(3) buckle: lambda_min crosses 0 near separation {seps_buck[cross]:.2f}")
    viz.plot_failure_modes(seps_taut, smin, w, kappa, deltas, full, feat,
                           seps_buck, lam, cross, "ex5_failure_modes.png")
    viz.show_all()


if __name__ == "__main__":
    main()
