# Shape Servoing of Deformable Objects — An Interactive Tutorial

A self-contained, visual introduction to **data-driven, mechanically model-free shape control** of
deformable objects, built around one running example: a planar elastic rod held by two robot arms.
The centerpiece is an interactive Jupyter notebook that builds the whole control loop step by step,
with a figure, a block diagram, or an animation at every stage.

## ▶ Open the tutorial online (no install)

[![Open in nbviewer](https://img.shields.io/badge/read-nbviewer-orange.svg)](https://nbviewer.org/github/nachocz/shape_servoing_of_deformable_objects_tutorial/blob/main/notebook/shape_servoing_tutorial.ipynb)
[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nachocz/shape_servoing_of_deformable_objects_tutorial/main?labpath=notebook/shape_servoing_tutorial.ipynb)

- **Just read it (instant):** open the notebook in
  [**nbviewer**](https://nbviewer.org/github/nachocz/shape_servoing_of_deformable_objects_tutorial/blob/main/notebook/shape_servoing_tutorial.ipynb).
  It ships with its outputs embedded, so all figures and the animations play in any browser, with
  nothing to install.
- **Run and edit it (interactive):** click
  [**Launch Binder**](https://mybinder.org/v2/gh/nachocz/shape_servoing_of_deformable_objects_tutorial/main?labpath=notebook/shape_servoing_tutorial.ipynb).
  After a short one-time build it opens a live JupyterLab in your browser, where you can change the
  target shape, the gain, or the feature representation in the cells marked **try it** and re-run.

## What the tutorial covers

1. The setup, the control problem, and the two phases (offline initialisation, online loop).
2. Shape representations (point coordinates, edge vectors, curvature) and their invariances.
3. Estimating the interaction Jacobian, and reading its conditioning from the singular values.
4. Closing the loop: pseudo-inverse control, the Broyden online update, a Lyapunov stability
   argument, and how the feature space changes what the controller does.
5. Reachability: when zero error is achievable, and why.
6. The three structural ways the loop fails (controllability, observability, mechanical instability).

## What is in this repository

```
.
├── notebook/                       # the interactive tutorial (start here)
│   ├── shape_servoing_tutorial.ipynb
│   ├── dlo.py                       # elastic-rod back-end (a black box to the controller)
│   ├── ssviz.py                     # plotting and animation helpers
│   └── README.md
├── lecture_notes.tex / .pdf         # the notes (theory, with intuition boxes)
├── lab_handout.tex   / .pdf         # the hands-on lab (five exercises)
├── code/                            # the lab code (exercises + reference solutions)
├── figs/                            # figures used by the notes
└── requirements.txt                 # Python dependencies (numpy, scipy, matplotlib)
```

## Running locally (optional)

You do not need this to read or run the tutorial online, but to work locally:

```bash
python -m pip install -r requirements.txt
cd notebook
jupyter lab shape_servoing_tutorial.ipynb
```

The lab exercises in `code/` are run in order (`python ex1_setup.py`, …); see
[notebook/README.md](notebook/README.md) and the lab handout for details. The notes and lab handout
are standard LaTeX (`pdflatex lecture_notes.tex`, `pdflatex lab_handout.tex`).

## Acknowledgement

This material — the notebook, the lecture notes, the lab, and the code — was prepared by Ignacio
Cuiral-Zueco with the assistance of Anthropic's Claude, used as a writing and coding aid.
