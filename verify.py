"""Execute from the project folder: python verify.py."""
from pathlib import Path
import numpy as np
from model import generate_data, split_data, build_model, explore_C


def main():
    cases = 0
    for dataset in ['Moons', 'Circles', 'Blobs']:
        for n in [20, 1000]:
            X, y = generate_data(dataset, n, 0.15, 42)
            X2, y2 = generate_data(dataset, n, 0.15, 42)
            np.testing.assert_array_equal(X, X2)
            np.testing.assert_array_equal(y, y2)
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
                    assert np.isfinite(model.decision_function(np.zeros((256, 2)))).all()
                    assert 0 <= model.score(Xv, yv) <= 1
                    cases += 1
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
    print(f'PASS: {cases} dataset/split/kernel cases, reproducibility, train-only scaling, support vectors, CV and Streamlit UI smoke checks.')


if __name__ == '__main__':
    main()
