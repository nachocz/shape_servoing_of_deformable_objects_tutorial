# Shape Servoing — Interactive Notebook

A single, self-contained Jupyter notebook that synthesises the lecture notes, the lab
exercises, and the simulation code into one runnable, visual tutorial. It walks through a
complete data-driven, mechanically model-free shape-servoing loop on a planar elastic rod, with a
figure, a labelled block diagram, or an animation at every step: two robot arms hold the rod, and
the notebook animates them reshaping it, probes the Jacobian one degree of freedom at a time, and
runs into each of the failure modes.

## Contents

```
notebook/
├── shape_servoing_tutorial.ipynb   # the tutorial (run top to bottom)
├── dlo.py                          # elastic-rod back-end (treated as a black box)
├── ssviz.py                        # plotting and animation helpers (called by name; no need to read)
└── README.md                       # this file
```

The notebook cells hold the method, the algorithm, and the quantities you may want to change; the
bulkier plotting and animation code lives in `ssviz.py`, which the cells call by name. Cells marked
**try it** expose an editable knob, for instance the target shape (`TARGET`), the control gain
(`alpha`), or the feature representation, so the examples can be modified and re-run in place.

The notebook opens with the motivation for data-driven, model-free deformable-object control
(distinguished from mechanical model-based control) and an animated preview of the goal, then
covers, in order:

1. **The setup, the control problem, and the two phases** — a labelled stage figure, then block
   diagrams of the offline initialisation and the online loop that introduce the whole method before
   any detail.
2. **Shape representations** — point coordinates, edge vectors, curvature, each drawn explicitly, and
   their invariances under rigid motion.
3. **The interaction Jacobian** — an animation that probes each gripper DoF and lights up the
   corresponding column of `J`, the least-squares estimate, and its conditioning (rank,
   manipulability, condition number).
4. **Closing the loop** — pseudo-inverse control and the Broyden online update, a live animation of
   the arms reshaping the rod, a Lyapunov argument (`V = ½ eᵀe`) for local asymptotic stability, and a
   single animation comparing the controller under all three feature spaces (point coordinates, edge
   vectors, curvature) on a target that both deforms and rigidly moves the rod, with the manipulation,
   the geometric error, and the commanded action shown together.
5. **When is zero error achievable?** — reachability: an *animated* worked example of an unreachable
   target, with the residual split into its reachable and unreachable components and a discussion of
   when a less-local controller could do better.
6. **Three ways the loop fails** — loss of controllability (taut state), loss of observability
   (a symmetric feature), and mechanical instability (buckling), with animations of the taut
   collapse and the buckling snap.

A closing **recap** collects the ideas the tutorial puts together. The reachability discussion and
the failure taxonomy follow the work of Cuiral-Zueco and López-Nicolás (*IEEE RA-L*, 2024); the full
reference list is at the end of the notebook.

## Requirements

Python 3.9+ with `numpy`, `scipy`, `matplotlib`, and Jupyter:

```bash
python -m pip install numpy scipy matplotlib jupyterlab
```

## Running it

```bash
cd notebook
jupyter lab shape_servoing_tutorial.ipynb     # or: jupyter notebook
```

Then run the cells in order (`Run > Run All Cells`). The notebook ships with its outputs
already embedded, so it can also be read without running.

`dlo.py` is the only local dependency: it returns the equilibrium shape for any commanded
gripper pose and never exposes its internals, which is what makes the method mechanically
model-free.
