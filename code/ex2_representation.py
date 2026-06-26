"""
Exercise 2: choosing the shape representation
=============================================
The feature vector is a design choice. You implement three representations of the
sampled points, observe how each responds to a rigid motion (a motion that moves
the rod without deforming it), and choose the one the controller will use in
Exercises 3 and 4.

This file imports `sampled_points` from Exercise 1. Run with:
    python ex2_representation.py
"""
import numpy as np

from dlo import Rod
import labviz as viz
from ex1_setup import sampled_points


def rep_points(P):
    """Point coordinates: the P sampled points flattened into one vector."""
    # TODO (Exercise 2): flatten P into a single vector.
    return np.zeros(P.size)


def rep_edges(P):
    """Edge vectors p_{i+1} - p_i, flattened. A translation leaves them unchanged."""
    # TODO (Exercise 2): take the differences between consecutive points, then flatten.
    return np.zeros(2 * (len(P) - 1))


def rep_curvature(P):
    """Turning angle between consecutive edges. Any rigid motion leaves it unchanged."""
    # TODO (Exercise 2): compute the direction (angle) of each edge, take the
    #   differences between consecutive directions, and wrap them into (-pi, pi].
    return np.zeros(len(P) - 2)


def rigid_motion(P, angle_deg=0.0, translation=(0.0, 0.0)):
    """Rotate the points about their centroid by angle_deg, then translate them."""
    # TODO (Exercise 2): build the 2x2 rotation matrix for angle_deg, rotate P about
    #   its centroid (P.mean(axis=0)), and add the translation.
    return np.asarray(P, float).copy()


# Choose the representation the controller uses in Exercises 3 and 4. Point
# coordinates keep the rod's position; edge vectors and curvature discard part of it.
FEATURE = rep_points


def features(rod, left, right, warm_start=True):
    """Map a grasp to a feature vector: sample the shape, then apply FEATURE."""
    return FEATURE(sampled_points(rod, left, right, warm_start))


def main():
    rod = Rod()
    # A clear U-shape, sampled at P points, used to compare the representations.
    U = sampled_points(rod, [-1.6, 1.6, -1.2], [1.6, 1.6, 1.2], warm_start=False)

    # ---- provided: compare each representation under a rigid motion (do not edit) ----
    reps = {"point coords": rep_points, "edge vectors": rep_edges,
            "curvature": rep_curvature}
    motions = {"translation":          dict(angle_deg=0,  translation=(2.5, -0.5)),
               "translation+rotation": dict(angle_deg=25, translation=(2.5, -0.5))}
    print("Exercise 2: change in each representation under a rigid motion")
    print(f"{'representation':<16}" + "".join(f"{m:>22}" for m in motions))
    vals_rot = []
    for name, f in reps.items():
        base = f(U)
        row = [float(np.linalg.norm(f(rigid_motion(U, **kw)) - base))
               for kw in motions.values()]
        vals_rot.append(row[1])
        print(f"{name:<16}" + "".join(f"{v:>22.3f}" for v in row))
    print("\nThe units differ across rows, so read each value as zero or non-zero.")
    viz.plot_representations(U, rigid_motion(U, **motions["translation+rotation"]),
                             list(reps), vals_rot, "ex2_representation.png")
    viz.show_all()


if __name__ == "__main__":
    main()
