"""
Exercise 3: offline Jacobian estimation (solution)
==================================================
We never have a model of the rod, so we never compute a "true" Jacobian. Instead
we estimate the interaction Jacobian from data: move each gripper DoF a little,
gather the input/output changes, and regress the single matrix that best maps
them. We then read its rank and conditioning from the singular values.

Imports the nominal grasp from Exercise 1 and the feature pipeline from
Exercise 2. Exercise 4 imports `offline_jacobian` from here.

Run from this `code/` directory:
    python ex3_offline_jacobian_solution.py
"""
import numpy as np

from dlo import Rod, grasp_vector, split_grasp
import labviz as viz
from ex1_setup_solution import LEFT0, RIGHT0
from ex2_representation_solution import features

DOF_LABELS = ["xL", "yL", "thL", "xR", "yR", "thR"]

# Exploration moves for the regression: each DoF, at a few signed amplitudes.
EXPLORE_AMP   = np.array([0.35, 0.35, 0.18, 0.35, 0.35, 0.18])
EXPLORE_FRACS = (-1.0, -0.5, 0.5, 1.0)


def offline_jacobian(rod, r0, explore_amp, explore_fracs):
    """Estimate J_hat by least-squares regression over exploration data."""
    r0 = np.asarray(r0, float)
    s0 = features(rod, *split_grasp(r0), warm_start=False)
    dR_cols, dS_cols = [], []
    for dof in range(6):
        for frac in explore_fracs:
            r = r0.copy(); r[dof] += frac * explore_amp[dof]
            s = features(rod, *split_grasp(r), warm_start=True)
            dR_cols.append(r - r0)
            dS_cols.append(s - s0)
    dR = np.column_stack(dR_cols)
    dS = np.column_stack(dS_cols)
    J_hat = dS @ np.linalg.pinv(dR)
    return J_hat, dR, dS


def svd_metrics(J, tol=1e-3):
    """Singular values and the rank / manipulability / condition number."""
    S = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(S > tol * S[0]))
    w = float(np.prod(S))
    kappa = float(S[0] / S[-1]) if S[-1] > 0 else np.inf
    return S, rank, w, kappa


def main():
    rod = Rod()
    r0 = grasp_vector(LEFT0, RIGHT0)
    J_hat, dR, dS = offline_jacobian(rod, r0, EXPLORE_AMP, EXPLORE_FRACS)
    S, rank, w, kappa = svd_metrics(J_hat)

    # ---- provided: report and visualise (do not edit) --------------------------
    print(f"Exercise 3: offline fit residual ||dS - J_hat dR|| = "
          f"{np.linalg.norm(dS - J_hat @ dR):.3f}")
    print(f"            J_hat shape {J_hat.shape}, rank {rank}/6, "
          f"w = {w:.3f}, kappa = {kappa:.1f}")
    viz.plot_jacobian(J_hat, S, kappa, DOF_LABELS, "ex3_jacobian.png")
    viz.show_all()


if __name__ == "__main__":
    main()
