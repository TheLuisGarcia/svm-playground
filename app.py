"""Run with: python -m streamlit run app.py"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix
from model import generate_data, split_data, build_model, explore_C

st.set_page_config(page_title='SVM Playground', page_icon='〰️', layout='wide')
st.title('SVM Playground')
st.markdown('Explore how a support vector machine draws a boundary between two classes.')

with st.sidebar:
    st.header('01 · Make your data')
    dataset = st.selectbox('Dataset', ['Moons', 'Circles', 'Blobs'])
    n = st.slider('Number of points', 20, 1000, 300, step=1)
    noise = st.slider('Noise / spread', 0.0, 0.6, 0.15, step=0.01)
    if dataset == 'Blobs':
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

boundary_tab, accuracy_tab, guide_tab = st.tabs(['Decision regions', 'Accuracy exploration', 'How it works'])
with boundary_tab:
    show_test = st.checkbox('Show held-out test points', value=True)
    pad = np.maximum(np.ptp(X, axis=0) * 0.12, 0.3)
    lo, hi = X.min(axis=0) - pad, X.max(axis=0) + pad
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
        for label, color in [(0, '#2563eb'), (1, '#ea580c')]:
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
    cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
    st.write('Test confusion matrix (rows: actual class, columns: predicted class)')
    st.dataframe({'Actual class': ['0', '1'], 'Predicted 0': cm[:, 0], 'Predicted 1': cm[:, 1]}, hide_index=True)

@st.cache_data(show_spinner=False, max_entries=20)
def cached_explore(Xt, yt, k, g, d, co, s):
    return explore_C(Xt, yt, k, g, d, co, s)

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

### Next phases
CSV upload: validate numeric columns, missing values and target labels, then let users select features.
3D: add three-feature data and Plotly slices or an isosurface; the current synthetic examples have two features.
''')
