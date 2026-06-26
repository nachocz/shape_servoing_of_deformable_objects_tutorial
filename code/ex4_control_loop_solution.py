"""
Exercise 4: closing the loop with an online Jacobian update (solution)
=====================================================================
The full picture. We initialise the estimate with the offline regression of
Exercise 3, then run the control loop: each step inverts the current estimate to
command a gripper move, and a Broyden update keeps the estimate current as the
rod deforms.

Imports the grasps (Exercise 1), the feature pipeline (Exercise 2), and the
offline estimator (Exercise 3).

Run from this `code/` directory:
    python ex4_control_loop_solution.py
"""
import numpy as np

from dlo import Rod, grasp_vector, split_grasp
import labviz as viz
from ex1_setup_solution import LEFT0, RIGHT0, LEFT_T, RIGHT_T
from ex2_representation_solution import features
from ex3_offline_jacobian_solution import offline_jacobian, EXPLORE_AMP, EXPLORE_FRACS

DOF_LABELS = ["xL", "yL", "thL", "xR", "yR", "thR"]


def broyden_update(J_hat, dr, ds):
    """Rank-one update so J_hat better predicts the latest (dr, ds)."""
    return J_hat + np.outer(ds - J_hat @ dr, dr) / (dr @ dr)


def main():
    rod = Rod()
    r0 = grasp_vector(LEFT0, RIGHT0)
    s_star = features(rod, LEFT_T, RIGHT_T, warm_start=False)
    J_hat0, _, _ = offline_jacobian(rod, r0, EXPLORE_AMP, EXPLORE_FRACS)

    # Set USE_BROYDEN = False to compare: a fixed offline estimate drifts as the
    # rod deforms, so the same loop converges much more slowly.
    USE_BROYDEN = True
    gain, tol, max_iters = 0.05, 0.01, 450

    r = r0.copy().astype(float)
    s = features(rod, *split_grasp(r), warm_start=False)
    Jh = J_hat0.copy()
    hist, actions = [float(np.linalg.norm(s_star - s))], []
    for _ in range(max_iters):
        e = s_star - s
        if np.linalg.norm(e) < tol:
            break
        dr = gain * (np.linalg.pinv(Jh) @ e)
        s_prev = s.copy()
        r = r + dr
        s = features(rod, *split_grasp(r), warm_start=True)
        ds = s - s_prev
        if USE_BROYDEN:
            Jh = broyden_update(Jh, dr, ds)
        actions.append(dr)
        hist.append(float(np.linalg.norm(s_star - s)))

    # ---- provided: report and visualise (do not edit) --------------------------
    print(f"Exercise 4: control loop ||e|| {hist[0]:.3f} -> {hist[-1]:.3f} in "
          f"{len(hist) - 1} iters  (gain {gain}, Broyden={USE_BROYDEN})")
    viz.plot_convergence(hist, actions, DOF_LABELS, "ex4_estimation.png")
    viz.show_all()


if __name__ == "__main__":
    main()
