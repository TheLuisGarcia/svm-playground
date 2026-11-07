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

Open http://localhost:8501 if the browser does not open automatically. Stop with Ctrl+C. No account, API keys, Jupyter extensions, or external datasets are required (you can optionally upload your own CSV). Installation needs internet access.

## Architecture

```text
svm-playground/
├── app.py             # Streamlit controls, upload flow, metrics, Plotly plots and explanations
├── model.py           # Synthetic 2D/3D datasets, split, pipeline, 3D decision grid, cross-validation
├── csv_data.py        # CSV reading, validation, invalid-row removal, stratified sampling, example CSVs
├── examples/          # example-2d.csv and example-3d.csv (same files as the in-app downloads)
├── requirements.txt   # Compatible dependency ranges
├── requirements-lock.txt # Exact versions used for verification, if present
├── verify.py          # Reproducibility, leakage, boundary and UI smoke checks (also runs verify_csv.py)
├── verify_csv.py      # CSV parsing, validation, sampling, leakage and upload UI checks
└── README.md
```

Flow: controls → seeded synthetic dataset *or* validated CSV columns (optionally a seeded stratified sample) → seeded stratified train/test split → StandardScaler + SVC pipeline → predictions and metrics. Plot meshes (a 2D grid, or a 3D grid in 3D mode) are passed through the same pipeline, preserving original feature coordinates. Support vector indices refer only to the training array; for uploads, original row numbers travel through the same split so each support vector maps back to its row in your file.

## Features

- **View** toggle in the sidebar: 2D (default) or 3D. Kernel, C, gamma, degree, coef0, seed and test split controls apply to both.
- **Data source**: synthetic data (below) or **Upload CSV** (see [Using your own CSV](#using-your-own-csv)).
- 2D: moons, circles and two-class blobs; 3D: two-class three-feature blobs. 20–1000 points, adjustable noise/spread and seed.
- RBF, linear, polynomial and sigmoid kernels; logarithmic C and custom gamma, gamma scale/auto, degree and coef0.
- Reproducible stratified 10–40% test split. Scaling is fitted only on training data.
- Two-dimensional decision regions, zero decision contour, support-vector rings, train circles and test diamonds.
- 3D: rotatable Plotly scatter of train circles, test diamonds and support-vector rings, with the decision boundary drawn as the zero-level isosurface of the decision function. Optional ±1 margin surfaces and a 20–40 points-per-axis grid resolution.
- Train/test accuracy, support-vector count and test confusion matrix.
- Explicitly triggered seven-value C sweep: up to five stratified validation folds on the training portion, fresh preprocessing within each fold, fold variability and CSV download.

Changing controls automatically refits the main model. Exploration runs only when requested and is remembered per browser session by data and settings. Blobs (2D and 3D) use `cluster_std = 0.15 + 2 × noise`, with centres at ±(1, 1) or ±(1, 1, 1); moons/circles use their generators' Gaussian noise parameter. The datasets are binary and approximately balanced. Plot regions represent classes, not confidence or probabilities. A 160 × 160 mesh is an approximation; extreme settings can produce narrow regions that it misses. High C or degree may increase compute time.

### What the 3D view shows

The three axes are the three **input features**, in original units. The surface is the set of input points where the SVM decision function equals zero. It is **not** the kernel's feature space. RBF, polynomial and sigmoid kernels implicitly map inputs into a higher-dimensional space (infinite-dimensional for RBF) where the separator is a flat hyperplane. That space cannot be plotted. The 3D view shows the hyperplane mapped back into the input space, which is why it can curve. With a linear kernel the surface is a flat plane.

The surface is interpolated from decision values on a regular grid (30³ by default), so very local regions at high gamma can be missed. If the decision function does not cross zero inside the plotted box, no surface is drawn and the app says so. Decision grids are remembered per browser session by training data, settings and resolution.

## Using your own CSV

Choose **Data source → Upload CSV** in the sidebar. Not sure what a valid file looks like? Download the **2D example** or **3D example** from the sidebar (also in [examples/](examples/)) and upload it.

### File format

- Comma-separated, UTF-8 text (a byte-order mark is fine), at most **5 MB**, `.csv` extension.
- First row is a header with **unique** column names. Every row has the same number of values as the header; wrap text containing commas in double quotes.
- One row per example. Blank cells, `NA` and `NaN` count as missing; `inf` counts as non-finite.

```csv
x_position,y_position,ring
-1.9823,4.5405,inner
-4.6207,9.2542,outer
```

### Choosing columns

- **Target**: the class to predict (defaults to the last column). It must have **exactly two classes**, as text (`yes`/`no`, `spam`/`ham`) or numbers (`0`/`1`, `-5`/`7`, `True`/`False`). Surrounding spaces are ignored. Charts, legends and the confusion matrix show your original class names.
- **Features**: two numeric columns for the 2D view or three for 3D. The target is never offered as a feature, and the order you pick sets the x, y (and z) axes, which are labelled with your column names. The defaults prefer columns without missing values. A sidebar expander explains why text columns aren't offered, including columns that are "mostly numbers" except for a few text cells.
- Changing the file, target, features or view resets choices that no longer fit, and removes any results from the earlier selection.

### Validation and limits

- **Binary only (for now).** A target with three or more classes is rejected, and the message lists the classes and their counts. Numeric targets with non-integer values or many distinct values look continuous. They are rejected and **never** binned into classes automatically.
- **Missing or invalid values.** Rows with a missing target or a missing/non-finite value in a *selected* feature are reported by cause. They are removed only if you tick **Remove the N invalid rows**, and the app then reports how many were removed. Nothing is imputed.
- **At least 10 rows per class** after cleaning (and after sampling). Each split must leave both classes in the training and test sets, with at least two training rows per class for validation. Constant features are rejected.
- **1,000-row limit.** The model, which means train plus test, uses at most 1,000 rows. With more usable rows, tick **Use a stratified sample of 1,000 rows**. The sample keeps each class's share and is reproducible with the random seed. A table shows class counts before and after sampling. Strongly imbalanced classes trigger a warning to check the confusion matrix.
- Problems are reported as messages (empty, header-only, ragged, unclosed quotes, non-UTF-8, Excel workbooks, semicolon-separated, duplicate names, too few numeric columns) instead of tracebacks.

### Leakage and privacy

Uploads use the same seeded stratified split and `StandardScaler + SVC` pipeline as synthetic data. The scaler is fitted on training rows only, and afresh inside each cross-validation fold. Removing rows and sampling use per-row rules and class labels only; nothing is learned from the data before the split.

Uploads are processed by the server running this app. When you run it locally, that server is your own computer. The file and anything derived from it stay in that browser session's memory: nothing is written to disk on purpose, and Streamlit's cross-session cache isn't used. Removing the file or switching back to synthetic data discards it. If you host the app elsewhere, the host processes uploads, so avoid uploading sensitive data.

## Notes on evaluation

Accuracy on a tiny test set can fluctuate greatly. Repeated test-set inspection during tuning compromises its independence; prefer the training-only validation sweep and use a fresh test set for final evaluation. The app does not automatically choose a best model from test results.

## Verify

```sh
python verify.py
```

Checks exercise every 2D and 3D dataset/kernel at both point-count limits, seed reproducibility, scaler training statistics, support-vector indexing, finite predictions, 3D grid shape and ordering, a linear-kernel check that the zero isosurface is exactly the plane w·x + b = 0 in original units, cross-validation, default Streamlit rendering and the exploration button. They also switch to 3D, check the isosurface and point traces, toggle margins and test points, change kernels, and confirm that switching back to 2D keeps the previous dataset choice. `verify_csv.py` (run by `verify.py`) covers CSV uploads:

- Malformed and unreadable files, including the 5 MB limit, and both example files.
- Text, numeric and whitespace-padded labels; single-class, multi-class and continuous targets; selection rules.
- Invalid-row counting and opt-in removal.
- Imbalanced classes across every test-set percentage.
- Stratified sampling: exact class counts, seed reproducibility, and cleaning before sampling.
- Leakage: scaler statistics come from training rows; perturbing test rows leaves the fitted model unchanged; each CV fold has its own scaler.
- Support-vector row mapping.
- A Streamlit AppTest walk-through of the upload flow, from uploading the examples through errors, opt-ins, resets and the switch back to synthetic data.

These are automated smoke checks, not a substitute for visual browser review. In particular, AppTest inspects the Plotly figure spec but does not run Plotly.js in a browser.

For the exact verified environment, install `requirements-lock.txt` instead of `requirements.txt` when included. The normal requirements allow compatible upgrades.

## Future phases

1. CSV extensions: multi-class targets (one-vs-rest), other delimiters and encodings, optional imputation fitted inside the training pipeline only, and keeping an upload across data-source switches.
2. 3D extensions: more three-feature generators (for example, concentric spheres) and axis-aligned decision slices.
3. Optional hosting on a Python-capable Streamlit host; this ZIP runs locally and is not deployed.

## Documentation

- https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
- https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- https://plotly.com/python/3d-isosurface-plots/
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://scikit-learn.org/stable/common_pitfalls.html

Inspired by the idea of https://svmplayground.vercel.app/simulation; this project uses its own Python implementation.
