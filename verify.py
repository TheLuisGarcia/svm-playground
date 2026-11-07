"""Execute from the project folder: python verify.py."""
import json
from pathlib import Path
import numpy as np
from model import (DATASETS_2D, DATASETS_3D, generate_data, split_data, build_model,
                   explore_C, plot_bounds, decision_grid_3d)


def figure_traces(app, i=0):
    spec = json.loads(app.get('plotly_chart')[i].proto.spec)
    return [(t['type'], t.get('name'), t.get('isomin')) for t in spec['data']]


def main():
    cases = 0
    cases_3d = 0
    for dataset in DATASETS_2D + DATASETS_3D:
        dims = 3 if dataset in DATASETS_3D else 2
        for n in [20, 1000]:
            X, y = generate_data(dataset, n, 0.15, 42)
            X2, y2 = generate_data(dataset, n, 0.15, 42)
            np.testing.assert_array_equal(X, X2)
            np.testing.assert_array_equal(y, y2)
            assert X.shape == (n, dims) and set(y) == {0, 1}
            for test_size in [0.1, 0.4]:
                Xt, Xv, yt, yv = split_data(X, y, test_size, 42)
                again = split_data(X, y, test_size, 42)
                for a, b in zip((Xt, Xv, yt, yv), again):
                    np.testing.assert_array_equal(a, b)
                for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
                    model = build_model(kernel).fit(Xt, yt)
                    np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, Xt.mean(axis=0))
                    assert model.named_steps['standardscaler'].n_samples_seen_ == len(Xt)
                    svc = model.named_steps['svc']
                    np.testing.assert_allclose(svc.support_vectors_, model[0].transform(Xt[svc.support_]))
                    assert set(model.predict(Xv)).issubset({0, 1})
                    assert np.isfinite(model.decision_function(np.zeros((256, dims)))).all()
                    assert 0 <= model.score(Xv, yv) <= 1
                    if dims == 3:
                        lo, hi = plot_bounds(X)
                        assert (lo < X.min(axis=0)).all() and (hi > X.max(axis=0)).all()
                        axes, grid, decision = decision_grid_3d(model, lo, hi, 12)
                        assert grid.shape == (12 ** 3, 3) and decision.shape == (12 ** 3,)
                        assert np.isfinite(decision).all()
                        # Flattened 'ij' order: index (i, j, k) -> axes[0][i], axes[1][j], axes[2][k].
                        i, j, k = 3, 7, 10
                        np.testing.assert_array_equal(grid[(i * 12 + j) * 12 + k], [axes[0][i], axes[1][j], axes[2][k]])
                        cases_3d += 1
                    cases += 1
    # Linear 3D boundary: the zero isosurface is the plane w.x + b = 0 in original units.
    X, y = generate_data('Blobs 3D', 300, 0.15, 42)
    Xt, Xv, yt, yv = split_data(X, y, 0.25, 42)
    lin = build_model('linear').fit(Xt, yt)
    w = lin[1].coef_[0] / lin[0].scale_
    b = lin[1].intercept_[0] - w @ lin[0].mean_
    _, grid, decision = decision_grid_3d(lin, *plot_bounds(X), 15)
    np.testing.assert_allclose(grid @ w + b, decision, atol=1e-8)
    assert decision.min() < 0 < decision.max()
    # The test split never influences the 3D model's scaler.
    np.testing.assert_allclose(lin[0].mean_, Xt.mean(axis=0))
    assert not np.allclose(lin[0].mean_, X.mean(axis=0))
    results, folds = explore_C(Xt, yt, 'rbf', 'scale', 3, 0, 42)
    assert len(results) == 7 and folds == 5
    assert results['Validation accuracy'].between(0, 1).all()
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=60)
    assert not app.exception, str(app.exception)
    assert len(app.metric) == 3
    app.button[0].click().run(timeout=60)
    assert not app.exception, str(app.exception)
    app.sidebar.selectbox[0].select('Circles')
    app.sidebar.slider[0].set_value(20)
    app.sidebar.selectbox[1].select('poly')
    app.run(timeout=60)
    assert not app.exception, str(app.exception)
    # 3D mode: isosurface + train/test/support-vector traces.
    app.sidebar.radio[0].set_value('3D').run(timeout=60)
    assert not app.exception, str(app.exception)
    assert app.sidebar.selectbox[0].value == 'Blobs 3D'
    traces = figure_traces(app)
    assert ('isosurface', 'Decision boundary (f = 0)', 0.0) in traces, traces
    assert sum(t[0] == 'scatter3d' for t in traces) == 5, traces
    assert any('not** the kernel' in i.value for i in app.info)
    app.checkbox(key='show_test_3d').uncheck()
    app.checkbox[1].check()
    app.run(timeout=60)
    assert not app.exception, str(app.exception)
    traces = figure_traces(app)
    assert [t[2] for t in traces if t[0] == 'isosurface'] == [0.0, -1.0, 1.0], traces
    assert sum(t[0] == 'scatter3d' for t in traces) == 3, traces
    for kernel in ['linear', 'sigmoid']:
        app.sidebar.selectbox[1].select(kernel)
        app.sidebar.slider[0].set_value(1000)
        app.run(timeout=60)
        assert not app.exception, str(app.exception)
    app.button[0].click().run(timeout=60)
    assert not app.exception, str(app.exception)
    app.sidebar.radio[0].set_value('2D').run(timeout=60)
    assert not app.exception, str(app.exception)
    assert app.sidebar.selectbox[0].value == 'Circles'
    assert not any(t[0] == 'isosurface' for t in figure_traces(app))
    print(f'PASS: {cases} dataset/split/kernel cases ({cases_3d} in 3D), reproducibility, train-only scaling, '
          'support vectors, 3D grid and linear plane, CV and Streamlit 2D/3D UI smoke checks.')


if __name__ == '__main__':
    main()
