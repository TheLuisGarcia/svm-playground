"""CSV parsing, validation, cleaning and sampling for uploaded datasets.

Nothing here learns statistics from the data. Rows are kept or removed by
per-row rules (missing or non-finite values), classes are mapped to 0/1 in
sorted order, and optional sampling is stratified by class. Scaling still
happens inside the model pipeline, fitted on training rows only.
"""
import csv
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split

MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_TRAIN_ROWS = 1000
MIN_PER_CLASS = 10


class CSVError(ValueError):
    """A problem with the uploaded data, phrased for the person who uploaded it."""


@dataclass
class RowIssues:
    """Rows that cannot be used for the chosen target and features."""
    missing_target: int
    bad_features: dict          # feature name -> rows with a missing or non-finite value
    invalid_mask: np.ndarray    # True for rows that would be removed

    @property
    def n_invalid(self):
        return int(self.invalid_mask.sum())


@dataclass
class PreparedData:
    X: np.ndarray               # float, original units, columns in feature order
    y: np.ndarray               # int 0/1
    rows: np.ndarray            # 0-based data-row positions in the uploaded file
    class_names: list           # display names; class_names[i] is encoded as i
    features: list
    target: str
    n_uploaded: int
    n_removed: int
    sampled: bool
    counts_before_sampling: dict = field(default_factory=dict)
    counts_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------- reading

def read_csv(data, max_bytes=MAX_UPLOAD_BYTES):
    """Parse uploaded bytes into a DataFrame, or raise CSVError with a helpful message."""
    if len(data) > max_bytes:
        raise CSVError(f'The file is {len(data) / 1024 / 1024:.1f} MB. The limit is '
                       f'{max_bytes / 1024 / 1024:.0f} MB; remove rows or columns and try again.')
    if not data.strip():
        raise CSVError('The file is empty.')
    if data[:2] == b'PK' or data[:4] == b'\xd0\xcf\x11\xe0':
        raise CSVError('This looks like an Excel workbook, not a CSV file. In Excel, use '
                       'File → Save As → "CSV UTF-8 (comma delimited)" and upload that file.')
    if b'\x00' in data[:4096]:
        raise CSVError('This file contains binary data (or UTF-16 text), so it cannot be read as a CSV. '
                       'Save it as a UTF-8 CSV and try again.')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError as e:
        raise CSVError(f'The file is not valid UTF-8 text (first bad byte at position {e.start}). '
                       'Re-save it with UTF-8 encoding, for example "CSV UTF-8" in Excel or '
                       '"Unicode (UTF-8)" in LibreOffice or Google Sheets.') from None
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
        ragged = []
        for row in reader:
            if row and len(row) != len(header) and len(ragged) < 3:
                ragged.append(f'line {reader.line_num} has {len(row)}')
    except (csv.Error, StopIteration):
        raise CSVError('The file could not be read as CSV. Check for unmatched quotation marks.') from None
    if ragged:
        raise CSVError(f'Every row must have the same number of values as the header ({len(header)}), but '
                       + ', '.join(ragged) + '. Check for missing or extra commas, or text containing '
                       'commas that is not wrapped in quotation marks.')
    names = [h.strip() for h in header]
    named = [h for h in names if h]
    dupes = sorted({h for h in named if named.count(h) > 1})
    if dupes:
        raise CSVError('Column names must be unique. Duplicated: ' + ', '.join(f'"{d}"' for d in dupes) + '.')
    try:
        df = pd.read_csv(io.StringIO(text), skipinitialspace=True, index_col=False)
    except pd.errors.EmptyDataError:
        raise CSVError('The file has no columns.') from None
    except pd.errors.ParserError as e:
        raise CSVError('The file is not a well-formed CSV. Check that quotation marks are closed and every '
                       f'row has the same number of values as the header. Details: {str(e).strip()}') from None
    except (ValueError, csv.Error) as e:
        raise CSVError(f'The file could not be parsed as CSV: {e}') from None
    df.columns = [n if n else c for n, c in zip(names, df.columns)] if len(names) == len(df.columns) else df.columns
    if df.empty:
        raise CSVError('The file has a header row but no data rows.')
    if df.shape[1] == 1 and any(sep in str(df.columns[0]) for sep in ';\t|'):
        raise CSVError('Only one column was found. The file seems to use semicolons, tabs or pipes '
                       'between values; this app expects commas. Re-export it as comma-separated CSV.')
    return df.reset_index(drop=True)


def numeric_columns(df):
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def non_numeric_hints(df):
    """Explain why text columns are not offered as features, with an example offending value."""
    hints = {}
    for c in df.columns:
        if c in numeric_columns(df):
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            hints[c] = 'contains True/False values, so it can be a target but not a feature.'
            continue
        col = df[c].dropna()
        parsed = pd.to_numeric(col, errors='coerce')
        bad = col[parsed.isna()]
        if len(col) and 0 < len(bad) <= 0.1 * len(col):
            hints[c] = (f'mostly numbers, but {len(bad)} cell(s) contain text such as "{bad.iloc[0]}". '
                        'Fix or blank those cells to use it as a feature.')
        else:
            hints[c] = 'contains text, so it can be a target but not a feature.'
    return hints


# ---------------------------------------------------------------- validation

def _clean_labels(series):
    """Target labels with surrounding whitespace removed; blank strings count as missing."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series
    s = series.astype('string').str.strip()
    return s.mask(s == '')


def _label_name(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def check_selection(df, target, features, dims):
    if target not in df.columns:
        raise CSVError('Choose a target column.')
    if len(features) != dims:
        raise CSVError(f'Choose exactly {dims} feature columns for the {dims}D view '
                       f'({len(features)} selected).')
    if len(set(features)) != len(features):
        raise CSVError('Choose distinct feature columns.')
    if target in features:
        raise CSVError(f'"{target}" is the target, so it cannot also be a feature.')
    numeric = numeric_columns(df)
    for f in features:
        if f not in numeric:
            raise CSVError(f'"{f}" is not a numeric column, so it cannot be a feature.')


def check_target_classes(df, target):
    """Raise CSVError unless the non-missing target values form exactly two classes."""
    labels = _clean_labels(df[target]).dropna()
    if labels.empty:
        raise CSVError(f'The target column "{target}" has no values.')
    counts = labels.value_counts()
    k = len(counts)
    if k == 2:
        return
    if k == 1:
        raise CSVError(f'The target column "{target}" has only one class ("{_label_name(counts.index[0])}"). '
                       'A classifier needs examples of two classes.')
    is_float = pd.api.types.is_float_dtype(labels) and not np.all(np.mod(labels, 1) == 0)
    if pd.api.types.is_numeric_dtype(labels) and (is_float or k > 20):
        raise CSVError(f'The target column "{target}" has {k} different numeric values, so it looks like a '
                       'continuous quantity. This playground does binary classification and will not turn '
                       'numbers into classes automatically. Choose a column with exactly two classes, or '
                       'add a two-valued column (for example "high"/"low") to your file yourself.')
    shown = ', '.join(f'"{_label_name(v)}" ({n})' for v, n in counts.head(8).items())
    more = f' and {k - 8} more' if k > 8 else ''
    raise CSVError(f'The target column "{target}" has {k} classes: {shown}{more}. This playground currently '
                   'supports exactly two classes (binary classification). To try it, make a copy of your '
                   'file and keep only two classes, or combine classes into two groups such as "A" vs "not A".')


def find_row_issues(df, target, features):
    missing_target = _clean_labels(df[target]).isna().to_numpy()
    invalid = missing_target.copy()
    bad = {}
    for f in features:
        values = df[f].to_numpy(dtype=float)
        mask = ~np.isfinite(values)
        if mask.any():
            bad[f] = int(mask.sum())
        invalid |= mask
    return RowIssues(int(missing_target.sum()), bad, invalid)


def describe_issues(issues):
    parts = []
    if issues.missing_target:
        parts.append(f'{issues.missing_target} with a missing target')
    parts += [f'{n} with a missing or non-finite (blank, NaN, inf) "{f}"' for f, n in issues.bad_features.items()]
    return '; '.join(parts)


def _counts(y, names):
    return {names[i]: int((y == i).sum()) for i in range(len(names))}


def _require_min_per_class(counts, where):
    low = {k: v for k, v in counts.items() if v < MIN_PER_CLASS}
    if low:
        detail = ', '.join(f'"{k}" has {v}' for k, v in low.items())
        raise CSVError(f'Too few examples per class {where}: {detail}. Each class needs at least '
                       f'{MIN_PER_CLASS} rows so both the training and test sets contain it.')


def prepare(df, target, features, dims, drop_invalid=False, sample=False, seed=0,
            max_rows=MAX_TRAIN_ROWS):
    """Validate the selection and return a PreparedData, or raise CSVError."""
    check_selection(df, target, features, dims)
    check_target_classes(df, target)
    issues = find_row_issues(df, target, features)
    if issues.n_invalid and not drop_invalid:
        raise CSVError(f'{issues.n_invalid} row(s) cannot be used: {describe_issues(issues)}. '
                       'Fix them in your file, or choose to remove them.')
    keep = ~issues.invalid_mask
    if not keep.any():
        raise CSVError('No rows remain after removing invalid rows.')
    labels = _clean_labels(df[target])[keep]
    classes = sorted(labels.unique(), key=lambda v: (str(type(v)), v))
    if len(classes) != 2:
        raise CSVError(f'After removing invalid rows, the target has {len(classes)} class(es); exactly two are needed.')
    names = [_label_name(c) for c in classes]
    if names[0] == names[1]:
        names = [repr(c) for c in classes]
    y = (labels.to_numpy() == classes[1]).astype(int)
    X = df.loc[keep, features].to_numpy(dtype=float)
    rows = np.flatnonzero(keep)
    for i, f in enumerate(features):
        if np.ptp(X[:, i]) == 0:
            raise CSVError(f'"{f}" has the same value in every usable row, so it cannot separate the classes. '
                           'Choose a feature that varies.')
    counts_before = _counts(y, names)
    _require_min_per_class(counts_before, 'in the usable rows')
    sampled = len(y) > max_rows
    if sampled and not sample:
        raise CSVError(f'{len(y):,} usable rows exceed the {max_rows:,}-row training limit. '
                       f'Choose to use a stratified sample of {max_rows:,} rows.')
    if sampled:
        idx = stratified_sample(y, max_rows, seed)
        X, y, rows = X[idx], y[idx], rows[idx]
        _require_min_per_class(_counts(y, names), f'in the {max_rows:,}-row sample')
    return PreparedData(X, y, rows, names, list(features), target, len(df), issues.n_invalid if drop_invalid else 0,
                        sampled, counts_before, _counts(y, names))


def stratified_sample(y, n, seed):
    """Indices of a reproducible class-stratified sample of n rows, in original order."""
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def check_split(y_train, y_test, names):
    """Both classes must appear in both sets, with at least two training rows each for validation."""
    for part, ys in [('training', y_train), ('test', y_test)]:
        missing = [names[i] for i in (0, 1) if not (ys == i).any()]
        if missing:
            raise CSVError(f'The {part} set has no rows of class "{missing[0]}". Change the test-set '
                           'percentage, or add more rows of that class.')
    if np.bincount(y_train, minlength=2).min() < 2:
        raise CSVError('Each class needs at least two training rows. Lower the test-set percentage.')


# ---------------------------------------------------------------- examples

def example_csv(dims):
    """Deterministic example CSV text: 2D 'inner'/'outer' rings or 3D 'core'/'shell' points."""
    if dims == 2:
        X, y = make_circles(n_samples=400, noise=0.08, factor=0.5, random_state=7)
        df = pd.DataFrame({'x_position': X[:, 0] * 10, 'y_position': X[:, 1] * 10,
                           'ring': np.where(y == 1, 'inner', 'outer')})
    else:
        rng = np.random.default_rng(7)
        direction = rng.normal(size=(500, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        y = np.repeat([0, 1], 250)
        radius = np.where(y == 0, rng.uniform(0, 1.0, 500), rng.uniform(1.6, 2.4, 500))
        X = direction * radius[:, None] + rng.normal(scale=0.12, size=(500, 3))
        df = pd.DataFrame({'width_cm': X[:, 0] + 5, 'height_cm': X[:, 1] + 5, 'depth_cm': X[:, 2] + 5,
                           'region': np.where(y == 0, 'core', 'shell')})
        df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    return df.round(4).to_csv(index=False)
