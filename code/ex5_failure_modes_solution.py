"""
Exercise 5: the three ways the loop fails (solution, stand-alone)
================================================================
A separate, diagnostic study. Rather than driving the shape, we sweep the rod
through three singular situations and read each one's numerical signature. The
conditioning analysis uses the rod's `true_jacobian` oracle -- a Jacobian the
model-free controller never has, used here only to reveal the mechanics.

Reuses `svd_metrics` from Exercise 3; otherwise it stands alone.

Run from this `code/` directory:
    python ex5_failure_modes_solution.py
"""
import numpy as np

from dlo import Rod
import labviz as viz
from ex3_offline_jacobian_solution import svd_metrics


def taut_curve(rod, separations):
    """(1) Pull the rod straight and track the conditioning of its Jacobian."""
    smin, w, kappa = [], [], []
    for sep in separations:
        J = rod.true_jacobian([-sep / 2, 1.0, 0.0], [sep / 2, 1.0, 0.0])
        S, _, w_sep, kappa_sep = svd_metrics(J)
        smin.append(S[-1]); w.append(w_sep); kappa.append(kappa_sep)
    return np.array(smin), np.array(w), np.array(kappa)


def observability_curve(rod, deltas):
    """(2) An antisymmetric motion that a symmetric feature misses."""
    s0 = rod.shape_state([-3.0, 1.0, 0.0], [3.0, 1.0, 0.0], warm_start=False)
    full, feat = [], []
    for d in deltas:
        s = rod.shape_state([-3.0, 1.0 + d, 0.0], [3.0, 1.0 - d, 0.0],
                            warm_start=False)
        full.append(np.linalg.norm(s - s0))        # full-state change
        feat.append(abs(s[5] - s0[5]))             # midpoint-height change
    return np.array(full), np.array(feat)


def buckling_curve(rod, separations):
    """(3) The smallest eigenvalue of the straight-rod energy Hessian."""
    return np.array([rod.straight_min_eig(sep, y=1.0) for sep in separations])


def main():
    rod = Rod()
    rod_sym = Rod(gravity=0.0)                      # exact symmetry for case (2)

    seps_taut = np.linspace(6.0, 7.2, 18)
    smin, w, kappa = taut_curve(rod, seps_taut)
    deltas = np.linspace(0.0, 0.8, 18)
    full, feat = observability_curve(rod_sym, deltas)
    seps_buck = np.linspace(6.6, 4.8, 22)
    lam = buckling_curve(rod, seps_buck)
    cross = int(np.argmax(lam < 0))

    # ---- provided: report and visualise (do not edit) --------------------------
    print(f"(1) taut:   sigma_min {smin[0]:.3f} -> {smin[-1]:.3f},  "
          f"kappa {kappa[0]:.0f} -> {kappa[-1]:.0f}")
    print(f"(2) observ: full-state ||ds|| up to {full[-1]:.2f}, "
          f"but midpoint feature stays <= {feat.max():.4f}")
    print(f"(3) buckle: lambda_min crosses 0 near separation {seps_buck[cross]:.2f}")
    viz.plot_failure_modes(seps_taut, smin, w, kappa, deltas, full, feat,
                           seps_buck, lam, cross, "ex5_failure_modes.png")
    viz.show_all()


if __name__ == "__main__":
    main()
