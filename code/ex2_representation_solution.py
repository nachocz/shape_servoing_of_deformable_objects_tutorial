"""
Exercise 2: choosing the shape representation (solution)
=======================================================
The feature vector is a design choice. We implement three representations of the
sampled points and see how each responds to a rigid motion (which does not deform
the object), then pick the one the controller will use in Exercises 3 and 4.

Imports `sampled_points` from Exercise 1.

Run from this `code/` directory:
    python ex2_representation_solution.py
"""
import numpy as np

from dlo import Rod
import labviz as viz
from ex1_setup_solution import sampled_points


def rep_points(P):
    """Point coordinates, stacked into one flat vector."""
    return P.reshape(-1)


def rep_edges(P):
    """Edge vectors between consecutive points (invariant to translation)."""
    return np.diff(P, axis=0).reshape(-1)


def rep_curvature(P):
    """Turning angle between consecutive edges (invariant to any rigid motion)."""
    edges = np.diff(P, axis=0)
    theta = np.arctan2(edges[:, 1], edges[:, 0])
    dtheta = np.diff(theta)
    return np.arctan2(np.sin(dtheta), np.cos(dtheta))       # wrapped to (-pi, pi]


def rigid_motion(P, angle_deg=0.0, translation=(0.0, 0.0)):
    """Rotate the points about their centroid, then translate (no deformation)."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]])
    c = P.mean(axis=0)
    return (P - c) @ R.T + c + np.asarray(translation)


# The representation the controller uses in Exercises 3 and 4. Point coordinates
# keep the rod's pose, so the controller can place it as well as shape it.
FEATURE = rep_points


def features(rod, left, right, warm_start=True):
    """Grasp -> feature vector, via the Exercise 1 sampling and the chosen FEATURE."""
    return FEATURE(sampled_points(rod, left, right, warm_start))


def main():
    rod = Rod()
    U = sampled_points(rod, [-1.6, 1.6, -1.2], [1.6, 1.6, 1.2], warm_start=False)

    # ---- provided: compare the representations and visualise (do not edit) -----
    reps = {"point coords": rep_points, "edge vectors": rep_edges,
            "curvature": rep_curvature}
    motions = {"translation":          dict(angle_deg=0,  translation=(2.5, -0.5)),
               "translation+rotation": dict(angle_deg=25, translation=(2.5, -0.5))}
    print(f"Exercise 2: ||ds|| under a rigid motion")
    print(f"{'representation':<16}" + "".join(f"{m:>22}" for m in motions))
    vals_rot = []
    for name, f in reps.items():
        base = f(U)
        row = [float(np.linalg.norm(f(rigid_motion(U, **kw)) - base))
               for kw in motions.values()]
        vals_rot.append(row[1])
        print(f"{name:<16}" + "".join(f"{v:>22.3f}" for v in row))
    print("\n(units differ across rows: read each value as 'zero' vs 'nonzero')")
    viz.plot_representations(U, rigid_motion(U, **motions["translation+rotation"]),
                             list(reps), vals_rot, "ex2_representation.png")
    viz.show_all()


if __name__ == "__main__":
    main()
