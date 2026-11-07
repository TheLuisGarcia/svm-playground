"""Run with: python -m streamlit run app.py"""
import hashlib
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

import csv_data
from csv_data import CSVError, MAX_TRAIN_ROWS, MAX_UPLOAD_MB, MIN_PER_CLASS
from model import (DATASETS_2D, DATASETS_3D, generate_data, split_data, build_model,
                   explore_C, plot_bounds, decision_grid_3d)

CLASS_COLORS = ['#2563eb', '#ea580c']
DATASET_LABELS = {'Blobs 3D': 'Blobs (3 features)'}
SYNTHETIC, UPLOAD = 'Synthetic data', 'Upload CSV'

st.set_page_config(page_title='SVM Playground', page_icon='〰️', layout='wide')
st.title('SVM Playground')
st.markdown('Explore how a support vector machine draws a boundary between two classes.')


def plot_text(s):
    """User-supplied names, made safe for Plotly titles, legends and hover templates."""
    return html.escape(str(s)).replace('%', '％')


def short_hash(*parts):
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:12]


def session_memo(slot, key_parts, compute, keep=4):
    """Per-session memo (st.session_state), so uploaded data never enters Streamlit's global cache."""
    h = hashlib.sha256()
    for p in key_parts:
        if isinstance(p, np.ndarray):
            h.update(repr((p.shape, p.dtype.str)).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(repr(p).encode())
        h.update(b'|')
    store = st.session_state.setdefault('_memo', {}).setdefault(slot, {})
    key = h.hexdigest()
    if key not in store:
        while len(store) >= keep:
            store.pop(next(iter(store)))
        store[key] = compute()
    return store[key]


def forget_upload():
    """Drop the parsed upload and results derived from it (Streamlit drops the file itself)."""
    if st.session_state.pop('_csv', None) is not None:
        st.session_state.pop('_memo', None)


def load_csv(uploaded):
    """Parse an upload once per file and keep the result in this session only."""
    cached = st.session_state.get('_csv')
    if cached is None or cached['file_id'] != uploaded.file_id:
        try:
            df, error = csv_data.read_csv(uploaded.getvalue()), None
        except CSVError as e:
            df, error = None, str(e)
        cached = {'file_id': uploaded.file_id, 'name': uploaded.name, 'df': df, 'error': error}
        forget_upload()
        st.session_state['_csv'] = cached
    return cached


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    view = st.radio('View', ['2D', '3D'], horizontal=True,
                    help='2D: two input features. 3D: three input features with a decision surface.')
    dims = 2 if view == '2D' else 3
    source = st.radio('Data source', [SYNTHETIC, UPLOAD], horizontal=True, key='source')
    st.header('01 · Make your data')
    upload = target = features = None
    if source == SYNTHETIC:
        forget_upload()
        # Streamlit discards state of widgets that are not rendered, so remember each view's choice separately.
        key = f'dataset_{view}'
        if key not in st.session_state and f'_{key}' in st.session_state:
            st.session_state[key] = st.session_state[f'_{key}']
        dataset = st.selectbox('Dataset', DATASETS_2D if view == '2D' else DATASETS_3D,
                               key=key, format_func=lambda d: DATASET_LABELS.get(d, d))
        st.session_state[f'_{key}'] = dataset
        n = st.slider('Number of points', 20, 1000, 300, step=1)
        noise = st.slider('Noise / spread', 0.0, 0.6, 0.15, step=0.01)
        if dataset.startswith('Blobs'):
            st.caption('Blob standard deviation = 0.15 + 2 × noise.')
    else:
        uploaded = st.file_uploader(f'CSV file (up to {MAX_UPLOAD_MB} MB)', type=['csv'],
                                    max_upload_size=MAX_UPLOAD_MB, key='csv_file',
                                    help='Comma-separated, UTF-8, with a header row. One row per example.')
        st.caption('Uploads are processed by the server running this app (your own computer when run '
                   'locally) and kept only in this session\'s memory. Avoid uploading sensitive data.')
        st.caption('No file handy? Download an example, then upload it:')
        e1, e2 = st.columns(2)
        e1.download_button('2D example', csv_data.example_csv(2), 'svm-example-2d.csv', 'text/csv')
        e2.download_button('3D example', csv_data.example_csv(3), 'svm-example-3d.csv', 'text/csv')
        if uploaded is None:
            forget_upload()
        else:
            upload = load_csv(uploaded)
        if upload and upload['df'] is not None:
            df = upload['df']
            fid = upload['file_id']
            columns = list(df.columns)
            target = st.selectbox('Target column (the class to predict)', columns, index=len(columns) - 1,
                                  key=f'target_{fid}')
            options = [c for c in csv_data.numeric_columns(df) if c != target]
            complete = [c for c in options if df[c].notna().all()]
            default = (complete + [c for c in options if c not in complete])[:dims]
            features = st.multiselect(f'{dims} numeric feature columns', options, default=default,
                                      max_selections=dims, key=f'features_{fid}_{view}_{target}',
                                      help='Chosen order sets the axes: first = x, second = y'
                                           + (', third = z.' if dims == 3 else '.'))
    seed = int(st.number_input('Random seed', min_value=0, max_value=1000000, value=42,
                               help='Controls data generation or sampling, and the train/test split.'))
    test_pct = st.slider('Test set (%)', 10, 40, 25, step=5)
    st.header('02 · Shape the boundary')
    kernel = st.selectbox('Kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    log_c = st.slider('log₁₀ C', -3.0, 3.0, 0.0, step=0.25)
    C = 10.0 ** log_c
    st.caption(f'C = {C:.4g} · Larger C penalizes training errors more.')
    mode = st.selectbox('Gamma', ['scale', 'auto', 'custom'], disabled=kernel == 'linear')
    gamma = mode
    if mode == 'custom':
        gamma = 10.0 ** st.slider('log₁₀ gamma', -3.0, 2.0, -0.5, step=0.25,
                                  disabled=kernel == 'linear')
    degree = st.slider('Polynomial degree', 2, 5, 3, disabled=kernel != 'poly')
    coef0 = st.slider('coef0', 0.0, 2.0, 0.0, step=0.1, disabled=kernel not in ['poly', 'sigmoid'])
    st.caption('Gamma affects RBF, polynomial and sigmoid kernels. High gamma can create very local boundaries. Sigmoid may behave poorly on some settings.')


# ------------------------------------------------------------------ data
def uploaded_data():
    """Show the upload preview and checks; return PreparedData or stop the script with guidance."""
    if upload is None:
        st.info(f'**Upload a CSV file** in the sidebar to train the SVM on your own data. You need a header '
                f'row, a target column with exactly two classes (text or numbers), and at least {dims} numeric '
                f'feature columns. Use the example downloads in the sidebar to try it out.')
        st.stop()
    if upload['error']:
        st.error(f'**Could not read "{upload["name"]}".** {upload["error"]}')
        st.stop()
    df = upload['df']
    st.subheader(f'Your data · {upload["name"]}')
    st.caption(f'{len(df):,} rows × {df.shape[1]} columns · first rows shown')
    st.dataframe(df.head(8), hide_index=True)
    hints = csv_data.non_numeric_hints(df)
    if hints:
        with st.expander('Why are some columns not offered as features?'):
            for col, why in hints.items():
                st.markdown(f'- **{col}**: {why}')
    numeric = [c for c in csv_data.numeric_columns(df) if c != target]
    if len(numeric) < dims:
        st.error(f'The {view} view needs {dims} numeric feature columns besides the target, but this file has '
                 f'{len(numeric)}' + (f' ({", ".join(numeric)})' if numeric else '') + '. '
                 + ('Switch View to 2D, or ' if dims == 3 and len(numeric) == 2 else '')
                 + 'choose another target or add numeric columns.')
        st.stop()
    if len(features) != dims:
        st.info(f'Choose {dims} feature columns in the sidebar ({len(features)} selected).')
        st.stop()
    sig = short_hash(upload['file_id'], view, target, tuple(features))
    try:
        csv_data.check_selection(df, target, features, dims)
        csv_data.check_target_classes(df, target)
        issues = csv_data.find_row_issues(df, target, features)
        drop = False
        if issues.n_invalid:
            st.warning(f'**{issues.n_invalid} of {len(df):,} rows cannot be used:** {csv_data.describe_issues(issues)}. '
                       'The SVM needs a class and a finite number for every selected column. You can fix '
                       'these cells in your file, or remove these rows here.')
            drop = st.checkbox(f'Remove the {issues.n_invalid} invalid rows', key=f'drop_{sig}')
            if not drop:
                st.stop()
        usable = len(df) - issues.n_invalid
        sample = False
        if usable > MAX_TRAIN_ROWS:
            st.info(f'**{usable:,} usable rows** is more than the {MAX_TRAIN_ROWS:,}-row limit for this '
                    f'playground. A class-stratified sample keeps each class\'s share of the data and is '
                    f'reproducible with the random seed.')
            sample = st.checkbox(f'Use a stratified sample of {MAX_TRAIN_ROWS:,} rows', key=f'sample_{sig}')
            if not sample:
                st.stop()
        data = csv_data.prepare(df, target, features, dims, drop_invalid=drop, sample=sample, seed=seed)
    except CSVError as e:
        st.error(str(e))
        st.stop()
    if data.n_removed:
        st.success(f'Removed {data.n_removed} invalid row(s); {len(df) - data.n_removed:,} remain.')
    counts = pd.DataFrame({'Class': data.class_names,
                           'Usable rows': list(data.counts_before_sampling.values())})
    if data.sampled:
        counts[f'In the {MAX_TRAIN_ROWS:,}-row sample'] = list(data.counts_used.values())
    st.dataframe(counts, hide_index=True)
    used = list(data.counts_used.values())
    if max(used) >= 4 * min(used):
        st.warning('The classes are imbalanced. Accuracy can look high just by predicting the larger class, '
                   'so check the confusion matrix.')
    return data


if source == UPLOAD:
    try:
        data = uploaded_data()
    except Exception as e:  # st.stop() is a BaseException, so it passes through
        st.error(f'Something unexpected went wrong while checking this file ({type(e).__name__}: {e}). '
                 'Try re-saving it as a plain UTF-8 CSV with a header row, or choose other columns.')
        st.stop()
    X, y, rows = data.X, data.y, data.rows
    class_names = [f'{target} = {c}' for c in data.class_names]
    feature_names = data.features
else:
    X, y = generate_data(dataset, n, noise, seed)
    rows = np.arange(len(y))
    class_names = ['class 0', 'class 1']
    feature_names = [f'Feature {i + 1}' for i in range(X.shape[1])]

try:
    X_train, X_test, y_train, y_test, rows_train, rows_test = split_data(X, y, test_pct / 100, seed, rows)
    csv_data.check_split(y_train, y_test, class_names)
    model = build_model(kernel, C, gamma, degree, coef0).fit(X_train, y_train)
except (CSVError, ValueError) as e:
    st.error(f'The model could not be trained with these settings: {e}')
    st.stop()
svc = model.named_steps['svc']
a, b, c = st.columns(3)
a.metric('Train accuracy', f'{model.score(X_train, y_train):.1%}', help=f'{len(y_train)} training points')
b.metric('Test accuracy', f'{model.score(X_test, y_test):.1%}', help=f'{len(y_test)} held-out points')
c.metric('Support vectors', f'{len(svc.support_)} / {len(y_train)}')
is_csv = source == UPLOAD
names = [plot_text(f) for f in feature_names]
axis_titles = names if is_csv else [f'{f} · original units' for f in names]


def show_confusion_matrix():
    cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
    st.write('Test confusion matrix (rows: actual class, columns: predicted class)')
    st.dataframe({'Actual': class_names, f'Predicted {class_names[0]}': cm[:, 0],
                  f'Predicted {class_names[1]}': cm[:, 1]}, hide_index=True)


def hover(split, label, axes):
    coords = '<br>'.join(f'{names[i]}=%{{{a}:.4g}}' for i, a in enumerate(axes))
    row = '<br>Data row %{customdata}' if is_csv else ''
    return f'{split} · {plot_text(class_names[label])}<br>{coords}{row}<extra></extra>'


def render_2d():
    show_test = st.checkbox('Show held-out test points', value=True)
    lo, hi = plot_bounds(X, min_pad=0 if is_csv else 0.3)
    xs, ys = np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    pred = model.predict(grid).reshape(xx.shape)
    decision = model.decision_function(grid).reshape(xx.shape)
    fig = go.Figure(go.Contour(x=xs, y=ys, z=pred, zmin=0, zmax=1,
        colorscale=[[0, '#dbeafe'], [0.5, '#dbeafe'], [0.5, '#ffedd5'], [1, '#ffedd5']],
        contours=dict(start=0, end=1, size=1, coloring='fill', showlines=False),
        showscale=False, hoverinfo='skip'))
    fig.add_trace(go.Contour(x=xs, y=ys, z=decision, contours=dict(start=0, end=0, size=1, coloring='lines'),
                            line=dict(color='#334155', width=2), showscale=False, hoverinfo='skip', name='Boundary'))
    for points, labels, ids, split, symbol in [(X_train, y_train, rows_train, 'Train', 'circle'),
                                                (X_test, y_test, rows_test, 'Test', 'diamond')]:
        if split == 'Test' and not show_test:
            continue
        for label, color in enumerate(CLASS_COLORS):
            m = labels == label
            fig.add_trace(go.Scatter(x=points[m, 0], y=points[m, 1], mode='markers', customdata=ids[m] + 1,
                name=f'{split} · {plot_text(class_names[label])}',
                marker=dict(color=color, symbol=symbol, size=8, line=dict(color='white', width=0.7)),
                hovertemplate=hover(split, label, 'xy')))
    support = X_train[svc.support_]
    fig.add_trace(go.Scatter(x=support[:, 0], y=support[:, 1], mode='markers', name='Support vectors',
                            customdata=rows_train[svc.support_] + 1,
                            hovertemplate='Support vector' + ('<br>Data row %{customdata}' if is_csv else '')
                                          + '<extra></extra>',
                            marker=dict(symbol='circle-open', size=14, color='#111827', line=dict(width=2))))
    fig.update_layout(height=580, margin=dict(l=10, r=10, t=15, b=10),
                      xaxis_title=axis_titles[0], yaxis_title=axis_titles[1],
                      xaxis_range=[lo[0], hi[0]], yaxis_range=[lo[1], hi[1]],
                      legend=dict(orientation='h', y=-0.15), plot_bgcolor='white')
    if not is_csv:  # synthetic features share units; uploaded columns may not
        fig.update_yaxes(scaleanchor='x', scaleratio=1)
    st.plotly_chart(fig, width='stretch')
    st.caption('Background = predicted class · dark line = decision boundary · rings = training support vectors. Diamonds are test points. Regions are predictions, not probabilities.'
               + (' Axes use your columns\' original units and are not drawn to the same scale.' if is_csv else ''))
    show_confusion_matrix()


def render_3d():
    st.info('**What you are seeing:** the three axes are the three *input* features in original units. '
            'The grey surface is where the SVM decision function equals zero, so it separates points predicted '
            'as one class from the other. This is **not** the kernel\'s feature space: RBF, polynomial and sigmoid '
            'kernels implicitly map inputs into a higher-dimensional (for RBF, infinite-dimensional) space where '
            'the boundary is a flat hyperplane. That space cannot be plotted; what appears here is that '
            'hyperplane pulled back into the input space, which is why it can curve.')
    left, mid, right = st.columns(3)
    show_test = left.checkbox('Show held-out test points', value=True, key='show_test_3d')
    show_margins = mid.checkbox('Show margin surfaces (±1)', value=False)
    res = right.select_slider('Grid resolution', [20, 25, 30, 35, 40], value=30,
                              help='Points per axis for the isosurface. Higher is smoother but slower.')
    lo, hi = plot_bounds(X, min_pad=0 if is_csv else 0.3)
    _, grid, decision = session_memo('grid3d', [X_train, y_train, kernel, C, gamma, degree, coef0,
                                                tuple(lo), tuple(hi), res],
                                     lambda: decision_grid_3d(model, lo, hi, res))
    fig = go.Figure()
    surfaces = [(0.0, '#475569', 0.45, 'Decision boundary (f = 0)')]
    if show_margins:
        surfaces += [(-1.0, CLASS_COLORS[0], 0.15, f'Margin, {plot_text(class_names[0])} side (f = −1)'),
                     (1.0, CLASS_COLORS[1], 0.15, f'Margin, {plot_text(class_names[1])} side (f = +1)')]
    drawn = 0
    for level, color, opacity, name in surfaces:
        if not decision.min() < level < decision.max():
            continue
        drawn += level == 0.0
        fig.add_trace(go.Isosurface(x=grid[:, 0], y=grid[:, 1], z=grid[:, 2], value=decision,
            isomin=level, isomax=level, surface_count=1, caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=[[0, color], [1, color]], showscale=False, opacity=opacity, name=name,
            showlegend=True, hoverinfo='skip', lighting=dict(ambient=0.7, diffuse=0.5, specular=0.1)))
    for points, labels, ids, split, symbol in [(X_train, y_train, rows_train, 'Train', 'circle'),
                                                (X_test, y_test, rows_test, 'Test', 'diamond')]:
        if split == 'Test' and not show_test:
            continue
        for label, color in enumerate(CLASS_COLORS):
            m = labels == label
            p = points[m]
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers', customdata=ids[m] + 1,
                name=f'{split} · {plot_text(class_names[label])}',
                marker=dict(color=color, symbol=symbol, size=4 if split == 'Train' else 5,
                            line=dict(color='white', width=0.5)),
                hovertemplate=hover(split, label, 'xyz')))
    support = X_train[svc.support_]
    fig.add_trace(go.Scatter3d(x=support[:, 0], y=support[:, 1], z=support[:, 2], mode='markers',
        name='Support vectors', customdata=rows_train[svc.support_] + 1,
        marker=dict(symbol='circle-open', size=8, color='#111827', line=dict(width=2)),
        hovertemplate='Support vector' + ('<br>Data row %{customdata}' if is_csv else '') + '<extra></extra>'))
    axis = lambda i: dict(title=axis_titles[i] if is_csv else f'Feature {i + 1}', range=[lo[i], hi[i]],
                          backgroundcolor='white', gridcolor='#e2e8f0', zerolinecolor='#cbd5e1')
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation='h', y=-0.02),
                      scene=dict(xaxis=axis(0), yaxis=axis(1), zaxis=axis(2), aspectmode='cube'),
                      uirevision='svm-3d')
    st.plotly_chart(fig, width='stretch')
    if not drawn:
        st.warning('The decision function does not cross zero inside the plotted region, so no boundary '
                   'surface is drawn. Every grid point is predicted as the same class.')
    st.caption(f'Drag to rotate, scroll to zoom. Circles = training points · diamonds = test points · black rings = '
               f'training support vectors. The surface is interpolated from a {res}³ grid of decision values; '
               'thin or very local regions (high gamma) may be missed. Axes use original units; the model scales '
               'internally using training statistics only.')
    show_confusion_matrix()


boundary_label = 'Decision regions' if view == '2D' else '3D decision boundary'
boundary_tab, accuracy_tab, guide_tab = st.tabs([boundary_label, 'Accuracy exploration', 'How it works'])
with boundary_tab:
    if view == '2D':
        render_2d()
    else:
        render_3d()

with accuracy_tab:
    st.subheader('How does C affect accuracy?')
    st.write('Compare seven C values using stratified cross-validation on training data only. Other settings stay fixed. Each fold fits a fresh scaler.')
    st.warning('Repeatedly adjusting settings while watching test accuracy makes the test set less independent. Use validation to explore; reserve a fresh final test set for a real project.')
    if st.button('Run C exploration', type='primary'):
        with st.spinner('Comparing C values…'):
            results, folds = session_memo('explore', [X_train, y_train, kernel, gamma, degree, coef0, seed],
                                          lambda: explore_C(X_train, y_train, kernel, gamma, degree, coef0, seed))
        chart = go.Figure()
        for name in ['Train accuracy', 'Validation accuracy']:
            chart.add_trace(go.Scatter(x=results['C'], y=results[name], name=name, mode='lines+markers',
                error_y=dict(type='data', array=results['Validation std'], visible=True) if name == 'Validation accuracy' else None))
        chart.update_layout(xaxis=dict(type='log', title='C'), yaxis=dict(title='Accuracy', range=[0, 1.03]), height=360)
        st.plotly_chart(chart, width='stretch')
        st.caption(f'{folds} folds · error bars show ±1 fold standard deviation, not confidence intervals. Change a setting and rerun to refresh.')
        st.dataframe(results, hide_index=True)
        st.download_button('Download exploration CSV', results.to_csv(index=False), 'svm-c-exploration.csv', 'text/csv')
with guide_tab:
    st.markdown(f'''### A quick experiment
1. Start with Moons, RBF, and C = 1.
2. Switch to linear: a straight boundary struggles with curved classes.
3. Return to RBF and increase gamma and C. Watch for fragmented regions and a train–test accuracy gap.
4. Increase noise and compare validation accuracy across C values.

**C** controls the penalty for training errors; smaller values apply stronger regularization.
**Gamma** controls how locally points influence nonlinear kernels. `scale` uses the variance of the scaled training features; `auto` uses 1 / number of features.
**Support vectors** are training observations used to determine the SVM decision function.

Data generation and the stratified split use your seed. A pipeline learns the scaler only from training data; plot coordinates stay in original units.

### 3D mode
Switch **View** to 3D to fit the same SVM on three features. The surface is the zero-level isosurface
of the decision function, evaluated on a regular grid over the three input features.

- With a **linear** kernel the surface is a flat plane.
- With **RBF** and a large gamma, the surface wraps tightly around groups of training points.
- Turn on margin surfaces to see where the decision function equals ±1; support vectors lie on their own class's margin surface or on the wrong side of it.

The plot shows the **input space**, not the kernel's feature space. A kernel implicitly maps inputs into a
higher-dimensional space where the separator is a flat hyperplane; the curved surface you see is that hyperplane
viewed back in the original three features.

### Your own data (Upload CSV)
Choose **Upload CSV** as the data source. The file needs a header row, comma separators and UTF-8 text.
Pick a **target** column with exactly two classes (text such as `yes`/`no`, or numbers such as `0`/`1`) and
{dims} numeric **feature** columns for the {view} view. Each class needs at least {MIN_PER_CLASS} rows.

- Rows with a missing class or a blank, `NaN` or infinite feature value are only removed if you tick the option, and the app reports how many.
- Files with more than {MAX_TRAIN_ROWS:,} usable rows can use a stratified sample of {MAX_TRAIN_ROWS:,} rows, chosen with the random seed.
- Continuous targets are not binned automatically; the playground does classification only.
- Scaling is still learned from the training rows only, and afresh inside each validation fold.
''')
