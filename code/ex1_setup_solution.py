"""
Exercise 1: the shape-servoing problem (solution)
=================================================
Define the nominal and target end-effector poses, measure the rod shape for each,
and compute the flattened sampled-point feature vectors and their shape error.
There is no control yet: this script makes the actuation space and feature space
explicit and forms the error that later feedback control will reduce.

Later exercises import the grasps and `sampled_points` from this file, so the
work is reused as the controller is built up.

Run from this `code/` directory:
    python ex1_setup_solution.py
"""
import numpy as np

from dlo import Rod
import labviz as viz

# Grasps used throughout the lab (set once here; later exercises import them).
LEFT0,  RIGHT0  = [-3.0, 1.0, 0.0],  [3.0, 1.0, 0.0]      # nominal grasp
LEFT_T, RIGHT_T = [-3.5, 1.5, 0.5],  [3.5, 0.5, -0.5]     # target grasp


def sampled_points(rod, left, right, warm_start=True):
    """Solve the rod equilibrium for a grasp and return the P sampled points (P, 2)."""
    nodes = rod.solve(left, right, warm_start=warm_start)
    return rod.sample_points(nodes)


def main():
    rod = Rod()

    # measure the current and target shapes, and form the features and error
    P0 = sampled_points(rod, LEFT0, RIGHT0, warm_start=False)
    Pt = sampled_points(rod, LEFT_T, RIGHT_T, warm_start=False)
    s0, s_star = P0.reshape(-1), Pt.reshape(-1)      # point-coordinate features
    e = s_star - s0

    # ---- provided: report and visualise (do not edit) --------------------------
    print(f"Exercise 1: dim r = 6, dim s = {s0.size}, "
          f"||s* - s|| = {np.linalg.norm(e):.3f}")
    nodes0 = rod.solve(LEFT0, RIGHT0, warm_start=False)
    nodes_t = rod.solve(LEFT_T, RIGHT_T, warm_start=False)
    viz.plot_problem(nodes0, P0, nodes_t, Pt, LEFT0, RIGHT0,
                     float(np.linalg.norm(e)), "ex1_setup.png")
    viz.show_all()


if __name__ == "__main__":
    main()
