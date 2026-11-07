# Verification

Verified on Python 3.13 with exact dependency versions in requirements-lock.txt, by running `python verify.py`. That script also runs `verify_csv.py`.

## Synthetic data

- Passed 64 dataset / sample-count / split / kernel combinations (48 in 2D, 16 in 3D).
- Generation and splitting are deterministic. Scaler statistics come from training data only. Support-vector indexing and decision predictions (finite) were confirmed in 2D and 3D.
- 3D: decision-grid shape and point ordering are correct. With a linear kernel, decision values equal the plane w·x + b in original units. The 3D scaler uses training statistics, not full-dataset ones.
- Passed the seven-value training-only cross-validation sweep.
- AppTest passed:
  - Default rendering and the exploration button.
  - 20-point circles with a polynomial kernel.
  - 3D mode: isosurface, explanation, margin and test-point toggles, linear and sigmoid kernels at 1000 points.
  - Returning to 2D keeps the previous dataset.

## CSV uploads

- Each malformed input gets a clear message: empty, whitespace-only, header-only, duplicate or whitespace-duplicate names, short or long rows (including the case pandas would silently shift into an index), unclosed quote, Latin-1, UTF-16, Excel workbook, semicolons, over 5 MB.
- BOM, padded column names and padded labels are accepted. "Mostly numeric" and True/False columns get explanations.
- Both example files are deterministic, match the generator, map labels and rows exactly, and reach >90% RBF test accuracy.
- Labels: text, padded text, float 0.0/1.0 with a missing value, arbitrary integers (−5/7).
- Rejected targets:
  - One class.
  - Three classes, text or numeric; the message lists the class counts.
  - Continuous floats.
  - Many distinct integers.
- Selection rules: wrong feature count, duplicate features, target used as a feature, non-numeric feature, missing target column, constant feature.
- Invalid rows (NaN, ±inf, blank or whitespace target, a row with two problems) are counted per cause and rejected unless removal is chosen. After removal, exactly those rows are gone and the coordinates match the file.
- Imbalance: 9 minority rows are rejected. With 10 minority vs 500, every test percentage from 10–40% and seeds 0–4 leave both classes in train and test with at least two training rows per class. The split checker rejects bad splits.
- Sampling:
  - Over 1,000 rows requires opt-in.
  - Class counts are exact (2400/600 → 800/200).
  - The same seed gives identical rows; a different seed gives different rows.
  - Rows and labels match the file.
  - Exactly 1,000 rows are not sampled.
  - Cleaning happens before sampling.
  - A class that falls below 10 rows after sampling is rejected.
- Leakage:
  - Scaler mean and scale equal the training-row statistics.
  - Multiplying test rows by 1000 leaves the fitted support set, dual coefficients and scaler unchanged.
  - In every CV fold the scaler mean equals that fold's training-row mean.
- AppTest upload walk-through:
  - Empty state, privacy note and two example downloads.
  - 2D example: default target and features, legend names with original classes, axis titles, support-vector hover rows matching the model's training rows and the file's coordinates, confusion matrix labels, the exploration button.
  - 3D with too few columns: explained error.
  - 3D example: isosurface and axis titles.
  - Changing the target resets features and excludes the new target; a continuous target gets an explained error.
  - Switching back to 2D resets the feature count.
  - Malformed, Latin-1, empty and multi-class files: messages, no charts.
  - Invalid rows need opt-in, then "Removed 3 invalid row(s); 77 remain".
  - A 1,500-row file needs sampling opt-in, then shows the counts table, the support-vector total and the imbalance warning.
  - Returning to synthetic data clears the upload from session state.
- Manual AppTest probes: column names containing HTML, `%{x}` and commas are escaped in Plotly; True/False and numeric 0/1 targets work in 2D and 3D with exploration.
- The Streamlit server started headless and answered its health check.

No manual visual browser review was performed. AppTest checks the Plotly figure spec but does not render it with Plotly.js, and it does not enforce the browser-side upload size limit; the 5 MB check in `csv_data.read_csv` is tested directly. Hosted deployment is future work.
