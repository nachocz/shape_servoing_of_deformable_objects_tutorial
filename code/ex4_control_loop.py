"""
Exercise 4: closing the loop with an online Jacobian update
===========================================================
This brings the pieces together. The estimate from Exercise 3 starts the loop.
Each step inverts the current estimate to command a gripper move, and a Broyden
update keeps the estimate accurate as the rod deforms.

This file imports the grasps (Exercise 1), `features` (Exercise 2), and
`offline_jacobian` (Exercise 3). Run with:
    python ex4_control_loop.py
"""
import numpy as np

from dlo import Rod, grasp_vector, split_grasp
import labviz as viz
from ex1_setup import LEFT0, RIGHT0, LEFT_T, RIGHT_T
from ex2_representation import features
from ex3_offline_jacobian import offline_jacobian, EXPLORE_AMP, EXPLORE_FRACS

DOF_LABELS = ["xL", "yL", "thL", "xR", "yR", "thR"]


def broyden_update(J_hat, dr, ds):
    """Return J_hat corrected by the rank-one Broyden update for the latest (dr, ds).

    The correction spreads the prediction residual (ds - J_hat @ dr) along the
    direction dr; np.outer builds the rank-one matrix, normalised by dr @ dr.
    """
    # TODO (Exercise 4): apply the Broyden update.
    return J_hat


def main():
    rod = Rod()
    if LEFT0 is None:
        print("Exercise 4: set the grasps in ex1_setup.py first.")
        return
    if EXPLORE_AMP is None:
        print("Exercise 4: set the exploration moves in ex3_offline_jacobian.py first.")
        return

    # Starting point: the nominal command, the target features, and the offline estimate.
    r0 = grasp_vector(LEFT0, RIGHT0)
    s_star = features(rod, LEFT_T, RIGHT_T, warm_start=False)
    Jh = offline_jacobian(rod, r0, EXPLORE_AMP, EXPLORE_FRACS)[0].copy()

    # Set USE_BROYDEN = False to compare: a fixed estimate converges much more slowly.
    USE_BROYDEN = True
    gain, tol, max_iters = 0.05, 0.01, 450

    # Loop state: the current command r, the current features s, and the records.
    r = r0.copy().astype(float)
    s = features(rod, *split_grasp(r), warm_start=False)
    hist, actions = [float(np.linalg.norm(s_star - s))], []
    for _ in range(max_iters):
        # TODO (Exercise 4): one control step using the current estimate Jh:
        #   1) compute the error e = s_star - s and stop if ||e|| < tol;
        #   2) compute dr = gain * np.linalg.pinv(Jh) @ e;
        #   3) apply the command r = r + dr and measure the new s;
        #   4) compute ds = s - previous s and update Jh with broyden_update if USE_BROYDEN;
        #   5) append the current error norm to hist and dr to actions.
        break   # remove once implemented

    # ---- provided: print a summary and draw the convergence (do not edit) -------
    print(f"Exercise 4: control loop ||e|| {hist[0]:.3f} -> {hist[-1]:.3f} in "
          f"{len(hist) - 1} iters  (gain {gain}, Broyden={USE_BROYDEN})")
    viz.plot_convergence(hist, actions, DOF_LABELS, "ex4_estimation.png")
    viz.show_all()


if __name__ == "__main__":
    main()
