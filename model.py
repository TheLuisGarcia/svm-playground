"""Data generation, leakage-safe fitting, and training-only exploration."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATASETS_2D = ['Moons', 'Circles', 'Blobs']
DATASETS_3D = ['Blobs 3D']


def generate_data(dataset, n, noise, seed):
    """Return (X, y). 2D datasets have two features; 'Blobs 3D' has three."""
    if dataset == 'Moons':
        return make_moons(n_samples=n, noise=noise, random_state=seed)
    if dataset == 'Circles':
        return make_circles(n_samples=n, noise=noise, factor=0.45, random_state=seed)
    if dataset == 'Blobs':
        return make_blobs(n_samples=n, centers=[[-1, -1], [1, 1]],
                          cluster_std=0.15 + 2 * noise, random_state=seed)
    if dataset == 'Blobs 3D':
        return make_blobs(n_samples=n, centers=[[-1, -1, -1], [1, 1, 1]],
                          cluster_std=0.15 + 2 * noise, random_state=seed)
    raise ValueError(f'Unknown dataset: {dataset}')


def split_data(X, y, test_size, seed):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def build_model(kernel='rbf', C=1.0, gamma='scale', degree=3, coef0=0.0):
    return make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma,
                                               degree=degree, coef0=coef0, cache_size=128))


def plot_bounds(X, frac=0.12, min_pad=0.3):
    """Padded per-feature plotting range, in original feature units."""
    pad = np.maximum(np.ptp(X, axis=0) * frac, min_pad)
    return X.min(axis=0) - pad, X.max(axis=0) + pad


def decision_grid_3d(model, lo, hi, resolution):
    """Evaluate the fitted pipeline's decision function on a regular 3D grid.

    Grid points are in original units; the pipeline applies the training-fitted
    scaler before the SVC. Returns (axes, grid points, decision values).
    """
    axes = [np.linspace(lo[i], hi[i], resolution) for i in range(3)]
    mesh = np.meshgrid(*axes, indexing='ij')
    grid = np.column_stack([m.ravel() for m in mesh])
    return axes, grid, model.decision_function(grid)


def explore_C(X_train, y_train, kernel, gamma, degree, coef0, seed):
    """Each fold fits its own scaler; held-out test data is never used here."""
    folds = min(5, int(np.bincount(y_train).min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for C in np.logspace(-2, 2, 7):
        scores = cross_validate(build_model(kernel, C, gamma, degree, coef0),
                                X_train, y_train, cv=cv, return_train_score=True)
        rows.append({'C': C, 'Train accuracy': scores['train_score'].mean(),
                     'Validation accuracy': scores['test_score'].mean(),
                     'Validation std': scores['test_score'].std()})
    return pd.DataFrame(rows), folds
