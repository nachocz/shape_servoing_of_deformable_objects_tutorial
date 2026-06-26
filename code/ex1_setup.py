"""
Exercise 1: the shape-servoing problem
======================================
Define the nominal and target end-effector configurations, measure the rod shape
for each, and compute the feature vectors and the resulting shape error.
There is no control yet: the task is to make the actuation space and the feature
space explicit, and to form the error that later feedback control will reduce.

Later exercises import the grasps and `sampled_points` from this file, so work
through the exercises in order. The Rod API is documented at the top of `dlo.py`.

Run with:  python ex1_setup.py
"""
import numpy as np

from dlo import Rod
import labviz as viz

# Choose a nominal grasp and a reachable target grasp; each gripper pose is
# [x, y, theta]. These are reused by the later exercises, which import them.
# TODO (Exercise 1): replace None with your own poses, and try a few.
LEFT0,  RIGHT0  = None, None        # nominal grasp
LEFT_T, RIGHT_T = None, None        # target grasp


def sampled_points(rod, left, right, warm_start=True):
    """Return the P points sampled along the rod for one grasp, shape (P, 2).

    Solve the rod equilibrium with the provided left and right end-effector poses,
    then sample P points along the equilibrium curve. The controller uses the
    flattened sampled points as its feature vector.
    """
    # TODO (Exercise 1): solve for the equilibrium shape, then sample it.
    return np.zeros((5, 2))


def main():
    rod = Rod()
    if LEFT0 is None or LEFT_T is None:
        print("Exercise 1: set the grasps above, then re-run.")
        return

    # Measure the current and the target shapes (each a (P, 2) array of points).
    P0 = sampled_points(rod, LEFT0, RIGHT0, warm_start=False)
    Pt = sampled_points(rod, LEFT_T, RIGHT_T, warm_start=False)

    # The actuation space is the 6D end-effector pose vector; the feature space
    # is the 10D flattened sampled points. Build s0 and s_star from P0 and Pt,
    # and form the shape error e = s_star - s0.
    s0 = np.zeros(P0.size)          # current feature vector
    s_star = np.zeros(Pt.size)      # target feature vector
    e = np.zeros(P0.size)           # shape error

    # ---- provided: print a summary and draw the problem (do not edit) ----------
    print(f"Exercise 1: dim r = 6, dim s = {s0.size}, "
          f"||s* - s|| = {np.linalg.norm(e):.3f}")
    nodes0 = rod.solve(LEFT0, RIGHT0, warm_start=False)
    nodes_t = rod.solve(LEFT_T, RIGHT_T, warm_start=False)
    viz.plot_problem(nodes0, P0, nodes_t, Pt, LEFT0, RIGHT0,
                     float(np.linalg.norm(e)), "ex1_setup.png")
    viz.show_all()


if __name__ == "__main__":
    main()
