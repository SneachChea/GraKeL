"""Focused tests for GraphletSampling optimizations."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import isspmatrix_csr

from grakel.kernels import GraphletSampling
import grakel.kernels.graphlet_sampling as graphlet_sampling_module
from grakel.kernels.graphlet_sampling import _canonical_form_key
from grakel.kernels._isomorphism import Graph as BlissGraph


def adjacency(n, edges):
    """Build an undirected binary adjacency matrix."""
    matrix = np.zeros((n, n), dtype=float)
    for u, v in edges:
        matrix[u, v] = matrix[v, u] = 1
    return matrix


def test_canonical_form_key_matches_isomorphism():
    graph = BlissGraph(4, [(0, 1), (1, 2), (2, 3)])
    permuted = BlissGraph(4, [(3, 0), (0, 2), (2, 1)])
    non_isomorphic = BlissGraph(4, [(0, 1), (1, 2), (2, 0)])

    assert graph.isomorphic(permuted)
    assert _canonical_form_key(graph) == _canonical_form_key(permuted)
    assert not graph.isomorphic(non_isomorphic)
    assert _canonical_form_key(graph) != _canonical_form_key(non_isomorphic)


def test_canonical_form_key_falls_back_for_older_bliss_builds(monkeypatch):
    graph = BlissGraph(3, [(0, 1), (1, 2), (2, 0)])

    class LegacyGraph:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self._vertices = wrapped._vertices

        def canonical_labeling(self):
            return self._wrapped.canonical_labeling()

    monkeypatch.setattr(
        graphlet_sampling_module, "_legacy_bliss_warning_issued", False
    )
    with pytest.warns(UserWarning, match="lacks canonical_form_key"):
        legacy_key = _canonical_form_key(LegacyGraph(graph))
    assert legacy_key == _canonical_form_key(graph)


def test_seeded_sampling_is_reproducible_and_sparse():
    graphs = [
        adjacency(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        adjacency(5, [(0, 1), (1, 2), (2, 0), (2, 3)]),
    ]
    kwargs = {"k": 5, "sampling": {"n_samples": 40}, "random_state": 42}

    first = GraphletSampling(**kwargs)
    second = GraphletSampling(**kwargs)
    first_matrix = first.fit_transform([(graph,) for graph in graphs])
    second_matrix = second.fit_transform([(graph,) for graph in graphs])

    assert_allclose(first_matrix, second_matrix)
    assert isspmatrix_csr(first._phi_X)


def test_transform_diagonal_includes_target_only_bins():
    triangle = adjacency(3, [(0, 1), (1, 2), (2, 0)])
    triangle_with_tail = adjacency(
        4, [(0, 1), (1, 2), (2, 0), (2, 3)]
    )

    raw = GraphletSampling(k=3, sampling=None, random_state=42)
    raw.fit_transform([(triangle,)])
    raw_cross = raw.transform([(triangle_with_tail,)])
    x_diag, y_diag = raw.diagonal()

    assert raw._Y_graph_bins
    assert raw._phi_Y.shape == (1, len(raw._graph_bins))
    assert y_diag[0] > float(raw._phi_Y.multiply(raw._phi_Y).sum())

    normalized = GraphletSampling(
        k=3, sampling=None, random_state=42, normalize=True
    )
    normalized.fit_transform([(triangle,)])
    normalized_cross = normalized.transform([(triangle_with_tail,)])

    expected = raw_cross / np.sqrt(np.outer(y_diag, x_diag))
    assert_allclose(normalized_cross, expected)


def test_refit_discards_old_feature_caches():
    triangle = adjacency(3, [(0, 1), (1, 2), (2, 0)])
    path = adjacency(3, [(0, 1), (1, 2)])

    kernel = GraphletSampling(k=3, sampling=None, random_state=42)
    kernel.fit_transform([(triangle,)])
    kernel.fit([(path,)])

    assert not hasattr(kernel, "_phi_X")
    assert not hasattr(kernel, "_X_diag")

    refit_result = kernel.transform([(path,)])
    reference = GraphletSampling(k=3, sampling=None, random_state=42)
    reference.fit([(path,)])
    reference_result = reference.transform([(path,)])
    assert_allclose(refit_result, reference_result)
