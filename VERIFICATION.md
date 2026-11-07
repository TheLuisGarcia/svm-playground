# Verification

Verified on Python 3.13 with exact dependency versions in requirements-lock.txt.

- Passed 64 dataset / sample-count / split / kernel combinations (48 in 2D, 16 in 3D).
- Confirmed deterministic generation and splitting, training-only scaler statistics, support-vector indexing and finite decision predictions in 2D and 3D.
- 3D: confirmed decision-grid shape and point ordering, and that with a linear kernel the decision values equal the original-unit plane w·x + b. Also confirmed that the 3D scaler uses training statistics, not full-dataset ones.
- Passed seven-value training-only cross-validation sweep.
- Passed Streamlit AppTest default rendering, exploration-button execution, and switching to 20-point circles with a polynomial kernel.
- Passed AppTest 3D mode: zero-level isosurface plus train/test/support-vector traces, the input-space explanation, margin and test-point toggles, linear and sigmoid kernels at 1000 points, the exploration button, and returning to 2D with the previous dataset kept.
- The Streamlit server started headless and answered its health check.

No manual visual browser review was performed; AppTest checks the Plotly figure spec but does not render it with Plotly.js. CSV upload and hosted deployment are future work.
