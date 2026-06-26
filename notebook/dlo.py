"""
A minimal planar elastic rod, the "physics" you do *not* need to modify.
==========================================================================

This module is the black box used by every exercise in the lab. It models a
deformable linear object (DLO) as a chain of nodes joined by edges, with two
elastic energies:

  * **stretch**, each edge resists changing its rest length,
  * **bend**   , each interior node resists turning (discrete curvature),

plus a mild gravity so the rod visibly hangs. Two grippers *clamp* the rod:
each one pins the position of an end node **and** the tangent of the adjacent
edge, i.e. a 3-DoF grasp pose ``(x, y, theta)``. Given the two grasp poses, the
rod shape is the configuration that minimises the total energy (a quasi-static
equilibrium), found with ``scipy.optimize`` using the analytic energy gradient.

The controller in the exercises never looks inside this file: it only commands
grasp poses and measures the resulting sampled shape. That is the whole point
of *model-free* shape servoing, the mechanics stay hidden.

Public API used by the exercises
--------------------------------
    rod = Rod()                                  # default parameters
    nodes  = rod.solve(left_pose, right_pose)    # equilibrium, shape (N, 2)
    pts    = rod.sample_points(nodes)            # P feature points, shape (P, 2)
    s      = rod.shape_state(left_pose, right_pose)   # flattened state in R^2P

    grasp_vector(left_pose, right_pose)          # -> r in R^6
    split_grasp(r)                               # -> (left_pose, right_pose)

For the failure-mode exercise:
    rod.energy_hessian(left_pose, right_pose)        # Hessian of elastic energy
    rod.straight_min_eig(separation, y)              # buckling indicator
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
#  Small helpers shared by the exercises
# --------------------------------------------------------------------------- #
def grasp_vector(left_pose, right_pose) -> np.ndarray:
    """Stack the two 3-DoF grasp poses into the actuation vector r in R^6."""
    return np.concatenate([np.asarray(left_pose, float),
                           np.asarray(right_pose, float)])


def split_grasp(r: np.ndarray):
    """Inverse of :func:`grasp_vector`: r in R^6 -> (left_pose, right_pose)."""
    r = np.asarray(r, float)
    return r[:3].copy(), r[3:].copy()


def _perp(v: np.ndarray) -> np.ndarray:
    """Rotate a 2-vector by +90 degrees: (x, y) -> (-y, x)."""
    return np.array([-v[1], v[0]])


# --------------------------------------------------------------------------- #
#  The rod
# --------------------------------------------------------------------------- #
@dataclass
class Rod:
    """Planar elastic rod clamped at both ends by 3-DoF grippers.

    Parameters (all have sensible defaults; tweak them in the exercises that
    ask you to make the rod stiffer, floppier, or weightless):
        n_nodes   number of mass nodes along the rod
        length    rest length (world units)
        k_stretch resistance to changing edge length
        k_bend    resistance to bending (curvature)
        gravity   downward load per unit length (set 0.0 for no gravity)
    """
    n_nodes: int = 25
    length: float = 6.0
    k_stretch: float = 80.0
    k_bend: float = 4.0
    gravity: float = 0.45

    def __post_init__(self) -> None:
        self.n_edges = self.n_nodes - 1
        self.l0 = self.length / self.n_edges          # rest edge length
        # The four clamped nodes are 0, 1 (left gripper) and -2, -1 (right).
        self.free_idx = np.arange(2, self.n_nodes - 2)
        self._warm: np.ndarray | None = None

    # ---- boundary conditions ------------------------------------------------ #
    def _clamped_nodes(self, left_pose, right_pose):
        """Four pinned node positions implied by the two grasp poses."""
        xl, yl, thl = left_pose
        xr, yr, thr = right_pose
        p0 = np.array([xl, yl])
        p1 = p0 + self.l0 * np.array([np.cos(thl), np.sin(thl)])
        pN = np.array([xr, yr])
        pNm1 = pN - self.l0 * np.array([np.cos(thr), np.sin(thr)])
        return p0, p1, pNm1, pN

    def _assemble(self, x_free, p0, p1, pNm1, pN):
        """Full (n_nodes, 2) node array from the free coords and the clamps."""
        p = np.empty((self.n_nodes, 2))
        p[0], p[1] = p0, p1
        p[-2], p[-1] = pNm1, pN
        p[self.free_idx] = x_free.reshape(-1, 2)
        return p

    # ---- energy and gradient ------------------------------------------------ #
    def _energy_and_grad(self, x_free, p0, p1, pNm1, pN):
        p = self._assemble(x_free, p0, p1, pNm1, pN)
        grad = np.zeros_like(p)

        # --- stretch ---
        e = p[1:] - p[:-1]                       # edge vectors (n_edges, 2)
        elen = np.linalg.norm(e, axis=1)
        elen_safe = np.maximum(elen, 1e-9)
        stretch = elen - self.l0
        E_s = 0.5 * self.k_stretch * np.sum(stretch ** 2) / self.l0
        f = (self.k_stretch / self.l0) * (stretch / elen_safe)[:, None] * e
        grad[:-1] -= f
        grad[1:] += f

        # --- bend (turning angle at each interior node) ---
        a = e[:-1]                               # incoming edges
        b = e[1:]                                # outgoing edges
        la2 = np.maximum((a ** 2).sum(1), 1e-12)
        lb2 = np.maximum((b ** 2).sum(1), 1e-12)
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        dot = (a * b).sum(1)
        phi = np.arctan2(cross, dot)             # discrete curvature
        E_b = 0.5 * self.k_bend * np.sum(phi ** 2) / self.l0
        coeff = (self.k_bend / self.l0) * phi
        for k in range(len(phi)):
            ga = _perp(a[k]) / la2[k]
            gb = _perp(b[k]) / lb2[k]
            grad[k]     += coeff[k] * (ga)
            grad[k + 1] += coeff[k] * (-ga - gb)
            grad[k + 2] += coeff[k] * (gb)

        # --- gravity ---
        w = self.gravity * self.l0
        E_g = w * np.sum(p[:, 1])
        grad[:, 1] += w

        E = E_s + E_b + E_g
        return E, grad[self.free_idx].ravel()

    # ---- public solve ------------------------------------------------------- #
    def solve(self, left_pose, right_pose, warm_start: bool = True) -> np.ndarray:
        """Equilibrium node positions (n_nodes, 2) for the two grasp poses."""
        p0, p1, pNm1, pN = self._clamped_nodes(left_pose, right_pose)
        if warm_start and self._warm is not None:
            x0 = self._warm.copy()
        else:
            t = np.linspace(0, 1, self.n_nodes)[self.free_idx][:, None]
            x0 = ((1 - t) * p1 + t * pNm1).ravel()
        res = minimize(
            self._energy_and_grad, x0, jac=True, method="L-BFGS-B",
            args=(p0, p1, pNm1, pN),
            options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-8},
        )
        self._warm = res.x.copy()
        return self._assemble(res.x, p0, p1, pNm1, pN)

    # ---- feature sampling --------------------------------------------------- #
    def sample_points(self, nodes: np.ndarray, n_samples: int = 5) -> np.ndarray:
        """Sample ``n_samples`` points uniformly by arclength (P, 2).

        These stacked 2D coordinates are the measured shape used by the controller.
        """
        seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        fracs = np.linspace(0.12, 0.88, n_samples)
        targets = fracs * arc[-1]
        xs = np.interp(targets, arc, nodes[:, 0])
        ys = np.interp(targets, arc, nodes[:, 1])
        return np.column_stack([xs, ys])

    def shape_state(self, left_pose, right_pose, n_samples: int = 5,
                    warm_start: bool = True) -> np.ndarray:
        """Convenience: solve, sample, and flatten to the state s in R^{2P}."""
        nodes = self.solve(left_pose, right_pose, warm_start=warm_start)
        return self.sample_points(nodes, n_samples).reshape(-1)

    def true_jacobian(self, left_pose, right_pose, n_samples: int = 5,
                      eps: float = 1e-3) -> np.ndarray:
        """Interaction Jacobian ds/dr of the sampled point-coordinate features,
        by central finite differences.

        This is an *oracle*, provided for analysis and comparison: the model-free
        controller never has it (it estimates ``J_hat`` from data instead), but
        the failure-mode study uses it to reveal the mechanics behind a
        singularity.
        """
        r = grasp_vector(left_pose, right_pose)
        J = np.zeros((2 * n_samples, 6))
        for j in range(6):
            rp = r.copy(); rp[j] += eps
            rm = r.copy(); rm[j] -= eps
            s_plus  = self.shape_state(*split_grasp(rp), n_samples, warm_start=False)
            s_minus = self.shape_state(*split_grasp(rm), n_samples, warm_start=False)
            J[:, j] = (s_plus - s_minus) / (2 * eps)
        return J

    # ---- elastic stability (used only by the failure-mode exercise) --------- #
    def straight_free_coords(self, left_pose, right_pose) -> np.ndarray:
        """Free-node coordinates of the perfectly straight chord between the
        inner clamped nodes, the (possibly unstable) symmetric equilibrium."""
        p0, p1, pNm1, pN = self._clamped_nodes(left_pose, right_pose)
        t = np.linspace(0, 1, self.n_nodes)[self.free_idx][:, None]
        return ((1 - t) * p1 + t * pNm1).ravel()

    def energy_hessian(self, left_pose, right_pose,
                       x_free: np.ndarray | None = None,
                       eps: float = 1e-5) -> np.ndarray:
        """Hessian of the elastic energy w.r.t. the free node coordinates.

        Evaluated at ``x_free`` if given (e.g. the straight configuration),
        otherwise at the solved equilibrium. Built by central-differencing the
        analytic gradient and symmetrised.
        """
        p0, p1, pNm1, pN = self._clamped_nodes(left_pose, right_pose)
        if x_free is None:
            x_free = self.solve(left_pose, right_pose,
                                warm_start=False)[self.free_idx].ravel()
        n = x_free.size
        H = np.zeros((n, n))
        for i in range(n):
            d = np.zeros(n); d[i] = eps
            _, gp = self._energy_and_grad(x_free + d, p0, p1, pNm1, pN)
            _, gm = self._energy_and_grad(x_free - d, p0, p1, pNm1, pN)
            H[:, i] = (gp - gm) / (2 * eps)
        return 0.5 * (H + H.T)

    def straight_min_eig(self, separation: float, y: float = 1.0) -> float:
        """Smallest eigenvalue of the straight-rod energy Hessian for a given
        horizontal gripper separation. Positive -> straight rod stable; crossing
        zero -> buckling onset (the det(H_energy)=0 bifurcation)."""
        lp = np.array([-separation / 2, y, 0.0])
        rp = np.array([separation / 2, y, 0.0])
        xs = self.straight_free_coords(lp, rp)
        H = self.energy_hessian(lp, rp, x_free=xs)
        return float(np.linalg.eigvalsh(H)[0])
