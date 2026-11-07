# SVM Playground

A Python-first, local Streamlit website with interactive Plotly charts. Python 3.11–3.13 is recommended.

## Run locally

Unzip the archive, open a terminal in `svm-playground`, and run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1` (use `python` instead of `python3` when needed). If activation is restricted, call `.venv\Scripts\python.exe` directly for installation and launch.

Open http://localhost:8501 if the browser does not open automatically. Stop with Ctrl+C. No account, API keys, Jupyter extensions, or external datasets are required. Installation needs internet access.

## Architecture

```text
svm-playground/
├── app.py             # Streamlit controls, metrics, Plotly plots and explanations
├── model.py           # Synthetic 2D/3D datasets, split, pipeline, 3D decision grid, cross-validation
├── requirements.txt   # Compatible dependency ranges
├── requirements-lock.txt # Exact versions used for verification, if present
├── verify.py          # Reproducibility, leakage, boundary and UI smoke checks
└── README.md
```

Flow: controls → seeded synthetic dataset → seeded stratified train/test split → StandardScaler + SVC pipeline → predictions and metrics. Plot meshes (a 2D grid, or a 3D grid in 3D mode) are passed through the same pipeline, preserving original feature coordinates. Support vector indices refer only to the training array.

## Features

- **View** toggle in the sidebar: 2D (default) or 3D. Kernel, C, gamma, degree, coef0, seed and test split controls apply to both.
- 2D: moons, circles and two-class blobs; 3D: two-class three-feature blobs. 20–1000 points, adjustable noise/spread and seed.
- RBF, linear, polynomial and sigmoid kernels; logarithmic C and custom gamma, gamma scale/auto, degree and coef0.
- Reproducible stratified 10–40% test split. Scaling is fitted only on training data.
- Two-dimensional decision regions, zero decision contour, support-vector rings, train circles and test diamonds.
- 3D: rotatable Plotly scatter of train circles, test diamonds and support-vector rings, with the decision boundary drawn as the zero-level isosurface of the decision function. Optional ±1 margin surfaces and a 20–40 points-per-axis grid resolution.
- Train/test accuracy, support-vector count and test confusion matrix.
- Explicitly triggered seven-value C sweep: up to five stratified validation folds on the training portion, fresh preprocessing within each fold, fold variability and CSV download.

Changing controls automatically refits the main model. Exploration runs only when requested and is cached by data and settings. Blobs (2D and 3D) use `cluster_std = 0.15 + 2 × noise`, with centres at ±(1, 1) or ±(1, 1, 1); moons/circles use their generators' Gaussian noise parameter. The datasets are binary and approximately balanced. Plot regions represent classes, not confidence or probabilities. A 160 × 160 mesh is an approximation; extreme settings can produce narrow regions that it misses. High C or degree may increase compute time.

### What the 3D view shows

The three axes are the three **input features**, in original units. The surface is the set of input points where the SVM decision function equals zero. It is **not** the kernel's feature space. RBF, polynomial and sigmoid kernels implicitly map inputs into a higher-dimensional space (infinite-dimensional for RBF) where the separator is a flat hyperplane. That space cannot be plotted. The 3D view shows the hyperplane mapped back into the input space, which is why it can curve. With a linear kernel the surface is a flat plane.

The surface is interpolated from decision values on a regular grid (30³ by default), so very local regions at high gamma can be missed. If the decision function does not cross zero inside the plotted box, no surface is drawn and the app says so. Decision grids are cached by training data, settings and resolution.

Accuracy on a tiny test set can fluctuate greatly. Repeated test-set inspection during tuning compromises its independence; prefer the training-only validation sweep and use a fresh test set for final evaluation. The app does not automatically choose a best model from test results.

## Verify

```sh
python verify.py
```

Checks exercise every 2D and 3D dataset/kernel at both point-count limits, seed reproducibility, scaler training statistics, support-vector indexing, finite predictions, 3D grid shape and ordering, a linear-kernel check that the zero isosurface is exactly the plane w·x + b = 0 in original units, cross-validation, default Streamlit rendering and the exploration button. They also switch to 3D, check the isosurface and point traces, toggle margins and test points, change kernels, and confirm that switching back to 2D keeps the previous dataset choice. These are automated smoke checks, not a substitute for visual browser review. In particular, AppTest inspects the Plotly figure spec but does not run Plotly.js in a browser.

For the exact verified environment, install `requirements-lock.txt` instead of `requirements.txt` when included. The normal requirements allow compatible upgrades.

## Future phases

1. CSV upload: size limits, numeric feature/target selection, missing-value validation and class-count checks; fit any imputation only within training pipelines.
2. 3D extensions: more three-feature generators (for example, concentric spheres), choosing three columns from uploaded data, and axis-aligned decision slices.
3. Optional hosting on a Python-capable Streamlit host; this ZIP runs locally and is not deployed.

## Documentation

- https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
- https://plotly.com/python/3d-isosurface-plots/
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://scikit-learn.org/stable/common_pitfalls.html

Inspired by the idea of https://svmplayground.vercel.app/simulation; this project uses its own Python implementation.
