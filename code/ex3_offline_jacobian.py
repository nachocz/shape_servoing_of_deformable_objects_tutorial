"""
Exercise 3: offline Jacobian estimation
=======================================
The controller has no model of the rod, so it never computes a true Jacobian.
Instead it estimates the interaction Jacobian from data: it moves each gripper
degree of freedom a little, records how the features change, and fits the matrix
that best maps commands to feature changes. You then read its rank and
conditioning from the singular values.

This file imports the nominal grasp from Exercise 1 and `features` from
Exercise 2. Exercise 4 imports `offline_jacobian` from here. Run with:
    python ex3_offline_jacobian.py
"""
import numpy as np

from dlo import Rod, grasp_vector, split_grasp
import labviz as viz
from ex1_setup import LEFT0, RIGHT0
from ex2_representation import features

DOF_LABELS = ["xL", "yL", "thL", "xR", "yR", "thR"]

# The exploration moves for the regression: a per-DoF amplitude (a length-6 array)
# and a few signed fractions of it.
# TODO (Exercise 3): choose these, e.g. amplitudes near 0.3 and fractions such as
#   (-1.0, -0.5, 0.5, 1.0).
EXPLORE_AMP   = None
EXPLORE_FRACS = None


def offline_jacobian(rod, r0, explore_amp, explore_fracs):
    """Estimate the Jacobian J_hat by least-squares regression over exploration data.

    Return J_hat together with the data matrices dR (6 x K) of command changes and
    dS (n x K) of feature changes. Measure the features at a command r with
    features(rod, *split_grasp(r)).
    """
    s0 = features(rod, *split_grasp(r0), warm_start=False)   # features at the start
    dR_cols, dS_cols = [], []
    for dof in range(6):                       # explore one DoF at a time
        for frac in explore_fracs:
            # TODO (Exercise 3): build the command r by shifting DoF `dof` of r0 by
            #   frac * explore_amp[dof]; measure its features; and append the command
            #   change (r - r0) to dR_cols and the feature change (s - s0) to dS_cols.
            pass
    # TODO (Exercise 3): stack the columns into dR and dS, then fit the least-squares
    #   estimate J_hat = dS @ pinv(dR).
    dR, dS = np.zeros((6, 1)), np.zeros((s0.size, 1))   # replace with np.column_stack(...)
    J_hat = np.zeros((s0.size, 6))                       # replace with the regression
    return J_hat, dR, dS


def svd_metrics(J, tol=1e-3):
    """Return the singular values of J and its rank, manipulability, and condition number."""
    S = np.linalg.svd(J, compute_uv=False)
    # TODO (Exercise 3): from the singular values S, compute the rank (how many lie
    #   above tol * S[0]), the manipulability (their product), and the condition
    #   number (the largest divided by the smallest).
    rank, w, kappa = 0, 0.0, np.inf
    return S, rank, w, kappa


def main():
    rod = Rod()
    if LEFT0 is None:
        print("Exercise 3: set the grasps in ex1_setup.py first.")
        return
    if EXPLORE_AMP is None or EXPLORE_FRACS is None:
        print("Exercise 3: set EXPLORE_AMP and EXPLORE_FRACS above, then re-run.")
        return

    r0 = grasp_vector(LEFT0, RIGHT0)
    J_hat, dR, dS = offline_jacobian(rod, r0, EXPLORE_AMP, EXPLORE_FRACS)
    S, rank, w, kappa = svd_metrics(J_hat)

    # ---- provided: print a summary and draw the estimate (do not edit) ----------
    print(f"Exercise 3: offline fit residual ||dS - J_hat dR|| = "
          f"{np.linalg.norm(dS - J_hat @ dR):.3f}")
    print(f"            J_hat is {J_hat.shape}, rank {rank}/6, "
          f"w = {w:.3f}, kappa = {kappa:.1f}")
    viz.plot_jacobian(J_hat, S, kappa, DOF_LABELS, "ex3_jacobian.png")
    viz.show_all()


if __name__ == "__main__":
    main()
