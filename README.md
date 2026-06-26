# Shape Servoing of Deformable Objects — An Interactive Tutorial

A self-contained, visual introduction to **data-driven, mechanically model-free shape control** of
deformable objects, built around one running example: a planar elastic rod held by two robot arms.
The centerpiece is an interactive Jupyter notebook that builds the whole control loop step by step,
with a figure, a block diagram, or an animation at every stage.

## ▶ Open the tutorial online (no install)

[![Run in browser (JupyterLite)](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://nachocz.github.io/shape_servoing_of_deformable_objects_tutorial/lab/index.html?path=shape_servoing_tutorial.ipynb)
[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nachocz/shape_servoing_of_deformable_objects_tutorial/main?labpath=notebook/shape_servoing_tutorial.ipynb)
[![Open in nbviewer](https://img.shields.io/badge/read-nbviewer-orange.svg)](https://nbviewer.org/github/nachocz/shape_servoing_of_deformable_objects_tutorial/blob/main/notebook/shape_servoing_tutorial.ipynb)

- **Run it in your browser, no server, no install — [JupyterLite](https://nachocz.github.io/shape_servoing_of_deformable_objects_tutorial/lab/index.html?path=shape_servoing_tutorial.ipynb).**
  Python runs entirely client-side via WebAssembly (Pyodide). The notebook opens with all figures and
  animations already rendered, so you can read it immediately, and you can edit the cells marked **try
  it** (target shape, gain, feature representation) and re-run them. The animation-heavy cells run
  slowly under WebAssembly, so for re-running those at full speed use Binder below.
- **Run at full speed in the cloud — [Binder](https://mybinder.org/v2/gh/nachocz/shape_servoing_of_deformable_objects_tutorial/main?labpath=notebook/shape_servoing_tutorial.ipynb).**
  A short one-time build, then a live JupyterLab with a real CPython kernel.
- **Just read it — [nbviewer](https://nbviewer.org/github/nachocz/shape_servoing_of_deformable_objects_tutorial/blob/main/notebook/shape_servoing_tutorial.ipynb).**
  Static render with the animations playing (nbviewer.org is occasionally overloaded; if it shows a
  503, try again later or use the JupyterLite link above).

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
