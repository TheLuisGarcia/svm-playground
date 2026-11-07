"""CSV upload checks. Run all checks with: python verify.py (or this file alone)."""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

import csv_data
from csv_data import CSVError, read_csv, prepare, MAX_TRAIN_ROWS, MIN_PER_CLASS
from model import split_data, build_model, make_cv

HERE = Path(__file__).parent


def to_bytes(df):
    return df.to_csv(index=False).encode()


def raises(fn, *needles):
    try:
        fn()
    except CSVError as e:
        msg = str(e)
        for n in needles:
            assert n in msg, f'expected {n!r} in: {msg}'
        return msg
    raise AssertionError(f'expected CSVError containing {needles}')


def frame(n0, n1, seed=0, labels=('no', 'yes')):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    return pd.DataFrame({'a': rng.normal(y * 2.0), 'b': rng.normal(-y * 1.5), 'c': rng.normal(y),
                         'label': np.array(labels, dtype=object)[y]})


def check_parsing():
    bad = {
        b'': 'empty',
        b'   \n\n': 'empty',
        b'a,b,label\n': 'no data rows',
        b'a,a,label\n1,2,x\n': 'unique',
        b'a,b,label\n1,2\n3,4,x\n': 'line 2 has 2',
        b'a,b,label\n1,2,x,9\n': 'line 2 has 4',
        b'a,b\n1,2,3\n4,5,6\n': 'same number of values',   # pandas would silently use column 1 as an index
        b'a,b,label\n1,2,"x\n3,4,y\n': 'quotation marks',
        'a,b,label\n1,2,café\n'.encode('latin-1'): 'UTF-8',
        'a,b,label\n1,2,x\n'.encode('utf-16'): 'UTF-16',
        b'PK\x03\x04\x14\x00rest-of-xlsx': 'Excel',
        b'a;b;label\n1;2;x\n': 'semicolons',
    }
    for data, needle in bad.items():
        raises(lambda: read_csv(data), needle)
    raises(lambda: read_csv(b'a,b\n' + b'1,2\n' * 10, max_bytes=20), 'limit')
    big = b'a,b,label\n' + b'1.0,2.0,x\n' * (csv_data.MAX_UPLOAD_BYTES // 10 + 1)
    assert len(big) > csv_data.MAX_UPLOAD_BYTES
    raises(lambda: read_csv(big), '5 MB')
    df = read_csv('﻿a, b ,label\n1,2, yes\n3,4,no\n'.encode())   # BOM, padded names and labels
    assert list(df.columns) == ['a', 'b', 'label']
    hints = csv_data.non_numeric_hints(read_csv(b'a,b,label\n1,2,x\n?,4,y\n5,6,x\n7,8,y\n9,1,x\n2,2,y\n3,3,x\n4,4,y\n5,5,x\n6,6,y\n7,7,x\n'))
    assert '"?"' in hints['a'] and 'target' in hints['label']
    flags = read_csv(b'a,flag\n1,True\n2,False\n')
    assert flags['flag'].dtype == bool and 'flag' not in csv_data.numeric_columns(flags)
    assert 'True/False' in csv_data.non_numeric_hints(flags)['flag']


def check_examples():
    for dims, target, names in [(2, 'ring', ['inner', 'outer']), (3, 'region', ['core', 'shell'])]:
        text = csv_data.example_csv(dims)
        assert text == csv_data.example_csv(dims), 'examples must be deterministic'
        assert (HERE / 'examples' / f'example-{dims}d.csv').read_text() == text, 'examples/ out of date'
        df = read_csv(text.encode())
        features = [c for c in df.columns if c != target]
        assert len(features) == dims and csv_data.numeric_columns(df) == features
        data = prepare(df, target, features, dims)
        assert data.class_names == names and not data.sampled and data.n_removed == 0
        np.testing.assert_array_equal(data.X, df[features].to_numpy(float))
        np.testing.assert_array_equal(np.array(names)[data.y], df[target].to_numpy())
        Xt, Xv, yt, yv, rt, rv = split_data(data.X, data.y, 0.25, 42, data.rows)
        model = build_model('rbf').fit(Xt, yt)
        assert model.score(Xv, yv) > 0.9, 'examples should be learnable with RBF'
        # Support vectors map back to the right uploaded rows, in original units.
        sv = model.named_steps['svc'].support_
        np.testing.assert_array_equal(Xt[sv], df.loc[rt[sv], features].to_numpy(float))
        assert set(df.loc[rt[sv], target]) <= set(names)


def check_labels_and_classes():
    df = frame(30, 30)
    df['label'] = np.where(df['label'] == 'yes', ' yes', 'no ')   # whitespace is not a separate class
    data = prepare(df, 'label', ['a', 'b'], 2)
    assert data.class_names == ['no', 'yes'] and data.counts_used == {'no': 30, 'yes': 30}
    df['label'] = (df['label'].str.strip() == 'yes').astype(float)   # numeric 0.0/1.0 labels
    df.loc[0, 'label'] = np.nan
    data = prepare(read_csv(to_bytes(df)), 'label', ['a', 'b'], 2, drop_invalid=True)
    assert data.class_names == ['0', '1'] and data.n_removed == 1
    df = frame(30, 30)
    df['label'] = np.where(df['label'] == 'yes', -5, 7)              # labels that are not 0/1
    assert prepare(df, 'label', ['a', 'b'], 2).class_names == ['-5', '7']
    one = frame(30, 0)
    raises(lambda: prepare(one, 'label', ['a', 'b'], 2), 'only one class')
    multi = frame(30, 30)
    multi.loc[:9, 'label'] = 'maybe'
    raises(lambda: prepare(multi, 'label', ['a', 'b'], 2), '3 classes', 'exactly two classes', '"maybe" (10)')
    numeric_multi = frame(30, 30).assign(label=np.arange(60) % 3)
    raises(lambda: prepare(numeric_multi, 'label', ['a', 'b'], 2), '3 classes')
    continuous = frame(30, 30).assign(label=np.linspace(0, 1, 60))
    raises(lambda: prepare(continuous, 'label', ['a', 'b'], 2), 'continuous', 'will not turn numbers into classes')
    many_ints = frame(30, 30).assign(label=np.arange(60))
    raises(lambda: prepare(many_ints, 'label', ['a', 'b'], 2), 'continuous')


def check_selection_rules():
    df = frame(30, 30).assign(name='row')
    raises(lambda: prepare(df, 'label', ['a'], 2), 'exactly 2')
    raises(lambda: prepare(df, 'label', ['a', 'b'], 3), 'exactly 3')
    raises(lambda: prepare(df, 'label', ['a', 'a'], 2), 'distinct')
    raises(lambda: prepare(df, 'a', ['a', 'b'], 2), 'cannot also be a feature')
    raises(lambda: prepare(df, 'label', ['a', 'name'], 2), 'not a numeric column')
    raises(lambda: prepare(df, 'nope', ['a', 'b'], 2), 'target')
    raises(lambda: prepare(df.assign(c=1.0), 'label', ['a', 'c'], 2), 'same value in every')
    assert 'label' not in csv_data.numeric_columns(df) and 'name' not in csv_data.numeric_columns(df)


def check_invalid_rows():
    df = frame(40, 40)
    df.loc[3, 'a'] = np.nan
    df.loc[5, 'b'] = np.inf
    df.loc[6, 'b'] = -np.inf
    df.loc[7, 'label'] = None
    df.loc[8, 'label'] = '   '
    df.loc[9, ['a', 'b']] = np.nan       # one row with two problems is removed once
    df = read_csv(to_bytes(df))          # round-trip: "inf", "", "nan" written and read back
    issues = csv_data.find_row_issues(df, 'label', ['a', 'b'])
    assert issues.n_invalid == 6 and issues.missing_target == 2 and issues.bad_features == {'a': 2, 'b': 3}
    raises(lambda: prepare(df, 'label', ['a', 'b'], 2), '6 row(s) cannot be used', 'remove them')
    data = prepare(df, 'label', ['a', 'b'], 2, drop_invalid=True)
    assert data.n_removed == 6 and len(data.y) == 74 and np.isfinite(data.X).all()
    assert not set(data.rows) & {3, 5, 6, 7, 8, 9}
    np.testing.assert_array_equal(data.X, df.loc[data.rows, ['a', 'b']].to_numpy(float))
    # Invalid cells only in an unselected column do not remove rows.
    assert prepare(df.drop(index=[7, 8]).reset_index(drop=True), 'label', ['c', 'a'], 2, drop_invalid=True).n_removed == 2
    nothing = frame(15, 15).assign(a=np.nan)
    raises(lambda: prepare(nothing, 'label', ['a', 'b'], 2, drop_invalid=True), 'No rows remain')


def check_imbalance_and_split():
    raises(lambda: prepare(frame(500, MIN_PER_CLASS - 1), 'label', ['a', 'b'], 2), 'Too few examples per class', '"yes" has 9')
    data = prepare(frame(500, MIN_PER_CLASS), 'label', ['a', 'b'], 2)
    for pct in range(10, 45, 5):          # every test-set percentage the slider offers
        for seed in range(5):
            yt, yv = split_data(data.X, data.y, pct / 100, seed)[2:4]
            csv_data.check_split(yt, yv, data.class_names)
            assert np.bincount(yt).min() >= 2 and np.bincount(yv).min() >= 1
            make_cv(yt, seed)
    raises(lambda: csv_data.check_split(np.array([0, 0, 1, 1]), np.array([0, 0]), ['no', 'yes']),
           'test set has no rows of class "yes"')
    raises(lambda: csv_data.check_split(np.array([0, 0, 0, 1]), np.array([0, 1]), ['no', 'yes']),
           'at least two training rows')


def check_sampling():
    big = frame(2400, 600, seed=3)
    raises(lambda: prepare(big, 'label', ['a', 'b'], 2), '3,000 usable rows', 'stratified sample')
    s1 = prepare(big, 'label', ['a', 'b'], 2, sample=True, seed=11)
    s2 = prepare(big, 'label', ['a', 'b'], 2, sample=True, seed=11)
    s3 = prepare(big, 'label', ['a', 'b'], 2, sample=True, seed=12)
    assert s1.sampled and len(s1.y) == MAX_TRAIN_ROWS and len(set(s1.rows)) == MAX_TRAIN_ROWS
    assert s1.counts_before_sampling == {'no': 2400, 'yes': 600} and s1.counts_used == {'no': 800, 'yes': 200}
    np.testing.assert_array_equal(s1.rows, s2.rows)
    assert not np.array_equal(s1.rows, s3.rows)
    np.testing.assert_array_equal(s1.X, big.loc[s1.rows, ['a', 'b']].to_numpy(float))
    assert (np.array(['no', 'yes'])[s1.y] == big.loc[s1.rows, 'label']).all()
    exact = prepare(frame(500, 500), 'label', ['a', 'b'], 2)       # exactly at the limit: no sampling
    assert not exact.sampled and len(exact.y) == 1000
    # Removal happens before sampling and the sample is drawn from valid rows only.
    dirty = big.copy()
    dirty.loc[:99, 'a'] = np.nan
    s = prepare(dirty, 'label', ['a', 'b'], 2, drop_invalid=True, sample=True, seed=1)
    assert s.n_removed == 100 and s.rows.min() >= 100 and sum(s.counts_before_sampling.values()) == 2900
    # A minority class that is large enough overall can still be too small after sampling.
    rare = frame(4960, 40)                                         # 40 → 8 in a 1,000-row sample
    raises(lambda: prepare(rare, 'label', ['a', 'b'], 2, sample=True), 'in the 1,000-row sample', '"yes" has')


def check_leakage():
    data = prepare(frame(150, 120, seed=5), 'label', ['a', 'b', 'c'], 3)
    Xt, Xv, yt, yv, rt, rv = split_data(data.X, data.y, 0.3, 42, data.rows)
    assert not set(rt) & set(rv) and len(rt) + len(rv) == len(data.y)
    model = build_model('rbf').fit(Xt, yt)
    np.testing.assert_allclose(model[0].mean_, Xt.mean(axis=0))
    np.testing.assert_allclose(model[0].scale_, Xt.std(axis=0))
    # Wildly changing test rows changes nothing about the fitted model.
    X_poisoned = data.X.copy()
    test_pos = np.isin(data.rows, rv)
    X_poisoned[test_pos] = X_poisoned[test_pos] * 1000 + 50
    Xt2, Xv2 = split_data(X_poisoned, data.y, 0.3, 42)[:2]
    np.testing.assert_array_equal(Xt, Xt2)
    other = build_model('rbf').fit(Xt2, yt)
    np.testing.assert_array_equal(model[1].support_, other[1].support_)
    np.testing.assert_allclose(model[1].dual_coef_, other[1].dual_coef_)
    np.testing.assert_allclose(model[0].mean_, other[0].mean_)
    # Each validation fold refits the scaler on that fold's training rows only.
    cv = make_cv(yt, 42)
    out = cross_validate(build_model('rbf'), Xt, yt, cv=cv, return_estimator=True, return_indices=True)
    for est, train_idx, val_idx in zip(out['estimator'], out['indices']['train'], out['indices']['test']):
        np.testing.assert_allclose(est[0].mean_, Xt[train_idx].mean(axis=0))
        assert not np.allclose(est[0].mean_, Xt.mean(axis=0))
        assert not set(train_idx) & set(val_idx)


# ------------------------------------------------------------------ Streamlit UI

def traces(app, i=0):
    return json.loads(app.get('plotly_chart')[i].proto.spec)


def array(v):
    """Plotly 6 serializes arrays as base64 typed buffers."""
    if isinstance(v, dict) and 'bdata' in v:
        return np.frombuffer(base64.b64decode(v['bdata']), dtype=v['dtype'])
    return np.asarray(v)


def ok(app):
    assert not app.exception, str(app.exception)
    return app


def check_ui():
    from streamlit.testing.v1 import AppTest
    app = ok(AppTest.from_file(str(HERE / 'app.py')).run(timeout=60))
    app.sidebar.radio(key='source').set_value('Upload CSV')
    ok(app.run(timeout=60))
    assert any('Upload a CSV file' in i.value for i in app.info) and not app.get('plotly_chart')
    assert any('sensitive' in c.value for c in app.sidebar.caption)
    assert len(app.get('download_button')) == 2
    uploader = app.sidebar.file_uploader[0]
    assert uploader.allowed_type == ['.csv'] or uploader.allowed_type == ['csv']

    # Valid 2D example: defaults, labels, metrics, confusion matrix, exploration.
    uploader.set_value(('ex2.csv', csv_data.example_csv(2).encode(), 'text/csv'))
    ok(app.run(timeout=60))
    assert app.sidebar.selectbox[0].value == 'ring'
    assert app.sidebar.multiselect[0].value == ['x_position', 'y_position']
    spec = traces(app)
    names = [t.get('name') for t in spec['data']]
    assert 'Train · ring = inner' in names and 'Test · ring = outer' in names and 'Support vectors' in names
    assert spec['layout']['xaxis']['title']['text'] == 'x_position'
    assert spec['layout']['yaxis']['title']['text'] == 'y_position'
    assert 'scaleanchor' not in spec['layout']['yaxis']
    # Support-vector hover rows are the 1-based uploaded rows whose coordinates are plotted.
    sv = next(t for t in spec['data'] if t.get('name') == 'Support vectors')
    ex = read_csv(csv_data.example_csv(2).encode())
    sv_rows = array(sv['customdata']).astype(int) - 1
    np.testing.assert_allclose(np.c_[array(sv['x']), array(sv['y'])], ex.loc[sv_rows, ['x_position', 'y_position']])
    data = prepare(ex, 'ring', ['x_position', 'y_position'], 2)
    Xt, _, yt, _, rt, _ = split_data(data.X, data.y, 0.25, 42, data.rows)
    np.testing.assert_array_equal(sv_rows, rt[build_model('rbf').fit(Xt, yt)[1].support_])
    assert len(app.metric) == 3
    cm = app.dataframe[-1].value
    assert list(cm['Actual']) == ['ring = inner', 'ring = outer'] and cm.iloc[:, 1:].to_numpy().sum() == 100
    app.button[0].click()
    ok(app.run(timeout=60))
    assert len(app.get('plotly_chart')) == 2

    # Switching to 3D with only two numeric columns explains the problem.
    app.sidebar.radio[0].set_value('3D')
    ok(app.run(timeout=60))
    assert any('needs 3 numeric feature columns' in e.value for e in app.error) and not app.get('plotly_chart')

    # Valid 3D example.
    app.sidebar.file_uploader[0].set_value(('ex3.csv', csv_data.example_csv(3).encode(), 'text/csv'))
    ok(app.run(timeout=60))
    assert app.sidebar.multiselect[0].value == ['width_cm', 'height_cm', 'depth_cm']
    spec = traces(app)
    assert any(t['type'] == 'isosurface' for t in spec['data'])
    assert [spec['layout']['scene'][f'{a}axis']['title']['text'] for a in 'xyz'] == ['width_cm', 'height_cm', 'depth_cm']
    assert 'Train · region = core' in [t.get('name') for t in spec['data']]

    # Changing the target resets incompatible feature choices.
    df = read_csv(csv_data.example_csv(3).encode()).assign(code=lambda d: (d['region'] == 'core').astype(int))
    app.sidebar.file_uploader[0].set_value(('code.csv', to_bytes(df), 'text/csv'))
    ok(app.run(timeout=60))
    assert app.sidebar.selectbox[0].value == 'code'                      # last column by default
    assert 'code' not in app.sidebar.multiselect[0].options
    app.sidebar.selectbox[0].select('width_cm')
    ok(app.run(timeout=60))
    assert 'width_cm' not in app.sidebar.multiselect[0].value + app.sidebar.multiselect[0].options
    assert app.sidebar.multiselect[0].value == ['height_cm', 'depth_cm', 'code']
    assert any('continuous' in e.value for e in app.error)
    app.sidebar.selectbox[0].select('region')
    ok(app.run(timeout=60))
    assert app.get('plotly_chart') and not app.error
    app.sidebar.radio[0].set_value('2D')
    ok(app.run(timeout=60))
    assert len(app.sidebar.multiselect[0].value) == 2 and app.get('plotly_chart')

    # Malformed and multi-class files show messages, not tracebacks.
    for name, data, needle in [('bad.csv', b'a,b,label\n1,2\n', 'same number of values'),
                               ('latin.csv', 'a,b,label\n1,2,é\n'.encode('latin-1'), 'UTF-8'),
                               ('empty.csv', b'', 'empty')]:
        app.sidebar.file_uploader[0].set_value((name, data, 'text/csv'))
        ok(app.run(timeout=60))
        assert any(needle in e.value and name in e.value for e in app.error), [e.value for e in app.error]
        assert not app.get('plotly_chart')
    multi = frame(30, 30)
    multi.loc[:9, 'label'] = 'maybe'
    app.sidebar.file_uploader[0].set_value(('multi.csv', to_bytes(multi), 'text/csv'))
    ok(app.run(timeout=60))
    assert any('exactly two classes' in e.value for e in app.error)

    # Invalid rows need explicit opt-in, and the removed count is reported.
    dirty = frame(40, 40)
    dirty.loc[[1, 2], 'a'] = np.nan
    dirty.loc[3, 'label'] = None
    app.sidebar.file_uploader[0].set_value(('dirty.csv', to_bytes(dirty), 'text/csv'))
    ok(app.run(timeout=60))
    assert app.sidebar.multiselect[0].value == ['b', 'c'], 'defaults should prefer complete columns'
    app.sidebar.multiselect[0].set_value(['a', 'b'])
    ok(app.run(timeout=60))
    assert any('3 of 80 rows cannot be used' in w.value for w in app.warning) and not app.get('plotly_chart')
    app.checkbox[0].check()
    ok(app.run(timeout=60))
    assert any('Removed 3 invalid row(s); 77 remain' in s.value for s in app.success) and app.get('plotly_chart')

    # Large files need explicit opt-in to sampling; class counts before and after are shown.
    app.sidebar.file_uploader[0].set_value(('big.csv', to_bytes(frame(1200, 300)), 'text/csv'))
    ok(app.run(timeout=60))
    assert any('1,500 usable rows' in i.value for i in app.info) and not app.get('plotly_chart')
    app.checkbox[0].check()
    ok(app.run(timeout=60))
    counts = next(d.value for d in app.dataframe if 'Class' in getattr(d.value, 'columns', []))
    assert list(counts['Usable rows']) == [1200, 300] and list(counts['In the 1,000-row sample']) == [800, 200]
    assert app.metric[2].value.endswith(f'/ {int(1000 * 0.75)}')
    assert any('imbalanced' in w.value for w in app.warning)

    # Back to synthetic data: the original controls still work.
    app.sidebar.radio(key='source').set_value('Synthetic data')
    ok(app.run(timeout=60))
    assert app.sidebar.selectbox[0].value == 'Moons' and len(app.metric) == 3 and '_csv' not in app.session_state


def main():
    for check in [check_parsing, check_examples, check_labels_and_classes, check_selection_rules,
                  check_invalid_rows, check_imbalance_and_split, check_sampling, check_leakage, check_ui]:
        check()
    print('PASS: CSV parsing/malformed inputs, examples, text and numeric labels, class rules, selection rules, '
          'invalid-row removal, imbalance and split, sampling, leakage, and Streamlit upload UI checks.')


if __name__ == '__main__':
    main()
