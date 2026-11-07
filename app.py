"""Run with: python -m streamlit run app.py"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix
from model import (DATASETS_2D, DATASETS_3D, generate_data, split_data, build_model,
                   explore_C, plot_bounds, decision_grid_3d)

CLASS_COLORS = [(0, '#2563eb'), (1, '#ea580c')]
DATASET_LABELS = {'Blobs 3D': 'Blobs (3 features)'}

st.set_page_config(page_title='SVM Playground', page_icon='〰️', layout='wide')
st.title('SVM Playground')
st.markdown('Explore how a support vector machine draws a boundary between two classes.')

with st.sidebar:
    view = st.radio('View', ['2D', '3D'], horizontal=True,
                    help='2D: two input features. 3D: three input features with a decision surface.')
    st.header('01 · Make your data')
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
    seed = int(st.number_input('Random seed', min_value=0, max_value=1000000, value=42))
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

X, y = generate_data(dataset, n, noise, seed)
X_train, X_test, y_train, y_test = split_data(X, y, test_pct / 100, seed)
model = build_model(kernel, C, gamma, degree, coef0).fit(X_train, y_train)
svc = model.named_steps['svc']
a, b, c = st.columns(3)
a.metric('Train accuracy', f'{model.score(X_train, y_train):.1%}', help=f'{len(y_train)} training points')
b.metric('Test accuracy', f'{model.score(X_test, y_test):.1%}', help=f'{len(y_test)} held-out points')
c.metric('Support vectors', f'{len(svc.support_)} / {len(y_train)}')


def show_confusion_matrix():
    cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
    st.write('Test confusion matrix (rows: actual class, columns: predicted class)')
    st.dataframe({'Actual class': ['0', '1'], 'Predicted 0': cm[:, 0], 'Predicted 1': cm[:, 1]}, hide_index=True)


def render_2d():
    show_test = st.checkbox('Show held-out test points', value=True)
    lo, hi = plot_bounds(X)
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
    for points, labels, split, symbol in [(X_train, y_train, 'Train', 'circle'), (X_test, y_test, 'Test', 'diamond')]:
        if split == 'Test' and not show_test:
            continue
        for label, color in CLASS_COLORS:
            p = points[labels == label]
            fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode='markers', name=f'{split} · class {label}',
                marker=dict(color=color, symbol=symbol, size=8, line=dict(color='white', width=0.7)),
                hovertemplate=f'{split} · class {label}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>'))
    support = X_train[svc.support_]
    fig.add_trace(go.Scatter(x=support[:, 0], y=support[:, 1], mode='markers', name='Support vectors',
                            marker=dict(symbol='circle-open', size=14, color='#111827', line=dict(width=2))))
    fig.update_layout(height=580, margin=dict(l=10, r=10, t=15, b=10),
                      xaxis_title='Feature 1 · original units', yaxis_title='Feature 2 · original units',
                      legend=dict(orientation='h', y=-0.15), plot_bgcolor='white')
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    st.plotly_chart(fig, width='stretch')
    st.caption('Background = predicted class · dark line = decision boundary · rings = training support vectors. Diamonds are test points. Regions are predictions, not probabilities.')
    show_confusion_matrix()


@st.cache_data(show_spinner=False, max_entries=10)
def cached_grid_3d(_model, Xt, yt, k, C_, g, d, co, lo, hi, res):
    # _model is not hashed; the training data and settings that determine it form the cache key.
    return decision_grid_3d(_model, lo, hi, res)


def render_3d():
    st.info('**What you are seeing:** the three axes are the three *input* features in original units. '
            'The grey surface is where the SVM decision function equals zero, so it separates points predicted '
            'as class 0 from class 1. This is **not** the kernel\'s feature space: RBF, polynomial and sigmoid '
            'kernels implicitly map inputs into a higher-dimensional (for RBF, infinite-dimensional) space where '
            'the boundary is a flat hyperplane. That space cannot be plotted; what appears here is that '
            'hyperplane pulled back into the input space, which is why it can curve.')
    left, mid, right = st.columns(3)
    show_test = left.checkbox('Show held-out test points', value=True, key='show_test_3d')
    show_margins = mid.checkbox('Show margin surfaces (±1)', value=False)
    res = right.select_slider('Grid resolution', [20, 25, 30, 35, 40], value=30,
                              help='Points per axis for the isosurface. Higher is smoother but slower.')
    lo, hi = plot_bounds(X)
    _, grid, decision = cached_grid_3d(model, X_train, y_train, kernel, C, gamma, degree, coef0,
                                       tuple(lo), tuple(hi), res)
    fig = go.Figure()
    surfaces = [(0.0, '#475569', 0.45, 'Decision boundary (f = 0)')]
    if show_margins:
        surfaces += [(-1.0, CLASS_COLORS[0][1], 0.15, 'Margin, class 0 side (f = −1)'),
                     (1.0, CLASS_COLORS[1][1], 0.15, 'Margin, class 1 side (f = +1)')]
    drawn = 0
    for level, color, opacity, name in surfaces:
        if not decision.min() < level < decision.max():
            continue
        drawn += level == 0.0
        fig.add_trace(go.Isosurface(x=grid[:, 0], y=grid[:, 1], z=grid[:, 2], value=decision,
            isomin=level, isomax=level, surface_count=1, caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=[[0, color], [1, color]], showscale=False, opacity=opacity, name=name,
            showlegend=True, hoverinfo='skip', lighting=dict(ambient=0.7, diffuse=0.5, specular=0.1)))
    for points, labels, split, symbol in [(X_train, y_train, 'Train', 'circle'), (X_test, y_test, 'Test', 'diamond')]:
        if split == 'Test' and not show_test:
            continue
        for label, color in CLASS_COLORS:
            p = points[labels == label]
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers', name=f'{split} · class {label}',
                marker=dict(color=color, symbol=symbol, size=4 if split == 'Train' else 5,
                            line=dict(color='white', width=0.5)),
                hovertemplate=f'{split} · class {label}<br>x₁=%{{x:.3f}}<br>x₂=%{{y:.3f}}<br>x₃=%{{z:.3f}}<extra></extra>'))
    support = X_train[svc.support_]
    fig.add_trace(go.Scatter3d(x=support[:, 0], y=support[:, 1], z=support[:, 2], mode='markers',
        name='Support vectors', marker=dict(symbol='circle-open', size=8, color='#111827', line=dict(width=2)),
        hovertemplate='Support vector<br>x₁=%{x:.3f}<br>x₂=%{y:.3f}<br>x₃=%{z:.3f}<extra></extra>'))
    axis = lambda i: dict(title=f'Feature {i + 1}', range=[lo[i], hi[i]], backgroundcolor='white',
                          gridcolor='#e2e8f0', zerolinecolor='#cbd5e1')
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


@st.cache_data(show_spinner=False, max_entries=20)
def cached_explore(Xt, yt, k, g, d, co, s):
    return explore_C(Xt, yt, k, g, d, co, s)


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
            results, folds = cached_explore(X_train, y_train, kernel, gamma, degree, coef0, seed)
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
    st.markdown('''### A quick experiment
1. Start with Moons, RBF, and C = 1.
2. Switch to linear: a straight boundary struggles with curved classes.
3. Return to RBF and increase gamma and C. Watch for fragmented regions and a train–test accuracy gap.
4. Increase noise and compare validation accuracy across C values.

**C** controls the penalty for training errors; smaller values apply stronger regularization.
**Gamma** controls how locally points influence nonlinear kernels. `scale` uses the variance of the scaled training features; `auto` uses 1 / number of features.
**Support vectors** are training observations used to determine the SVM decision function.

Data generation and the stratified split use your seed. A pipeline learns the scaler only from training data; plot coordinates stay in original units.

### 3D mode
Switch **View** to 3D to fit the same SVM on a three-feature blobs dataset. The surface is the zero-level isosurface
of the decision function, evaluated on a regular grid over the three input features.

- With a **linear** kernel the surface is a flat plane.
- With **RBF** and a large gamma, the surface wraps tightly around groups of training points.
- Turn on margin surfaces to see where the decision function equals ±1; support vectors lie on their own class's margin surface or on the wrong side of it.

The plot shows the **input space**, not the kernel's feature space. A kernel implicitly maps inputs into a
higher-dimensional space where the separator is a flat hyperplane; the curved surface you see is that hyperplane
viewed back in the original three features.

### Next phases
CSV upload: validate numeric columns, missing values and target labels, then let users select features
(including three features for the 3D view).
''')
