"""A random input test for Kernel Oblects similar to test_estimator."""
# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
import sys
import warnings
import numpy as np

from grakel.datasets import generate_dataset

from grakel import GraphKernel
from grakel.kernels import GraphletSampling
from grakel.kernels import RandomWalk
from grakel.kernels import RandomWalkLabeled
from grakel.kernels import ShortestPath
from grakel.kernels import ShortestPathAttr
from grakel.kernels import WeisfeilerLehman
from grakel.kernels import WeisfeilerLehmanOptimalAssignment
from grakel.kernels import NeighborhoodHash
from grakel.kernels import PyramidMatch
from grakel.kernels import SubgraphMatching
from grakel.kernels import NeighborhoodSubgraphPairwiseDistance
from grakel.kernels import LovaszTheta
from grakel.kernels import SvmTheta
from grakel.kernels import OddSth
from grakel.kernels import Propagation
from grakel.kernels import PropagationAttr
from grakel.kernels import HadamardCode
from grakel.kernels import MultiscaleLaplacian
from grakel.kernels import VertexHistogram
from grakel.kernels import EdgeHistogram
from grakel.kernels import GraphHopper
from grakel.kernels import CoreFramework

import pytest

is_windows = sys.platform.lower().startswith("win")

verbose, normalize = False, True
default_eigvalue_precision = float("-1e-5")
rs = np.random.RandomState(42)
warnings.filterwarnings("ignore")

cvxopt = True
try:
    import cvxopt
except ImportError:
    cvxopt = False


def test_random_walk():
    """Random input test for the Simple Random Walk kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(0.01, 12.0),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=None)

    rw_kernel = RandomWalk(verbose=verbose, normalize=normalize)
    try:
        rw_kernel.fit_transform(train)
        rw_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_random_walk_pd():
    """Random input test for the Simple Random Walk kernel [n_jobs=-1/generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(0.01, 12.0),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=None)

    gk = GraphKernel(kernel="RW", verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_random_walk_labels():
    """Random input test for the Labelled Random Walk kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(0.01, 12.0),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    rw_kernel = RandomWalkLabeled(verbose=verbose, normalize=normalize)

    try:
        rw_kernel.fit_transform(train)
        rw_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_random_walk_labels_pd():
    """Random input test for the Labelled Random Walk kernel [n_jobs=-1/generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(0.01, 12.0),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    gk = GraphKernel(kernel={"name": "RW", "with_labels": True},
                     verbose=verbose, normalize=normalize, )

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_shortest_path():
    """Random input test for the Shortest Path kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    sp_kernel = ShortestPath(verbose=verbose, normalize=normalize)

    try:
        sp_kernel.fit_transform(train)
        sp_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    train, test = generate_dataset(n_graphs=50,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=20,
                                   random_state=rs,
                                   features=('na', 5))

    sp_kernel = ShortestPathAttr(verbose=verbose, normalize=normalize)

    try:
        sp_kernel.fit_transform(train)
        sp_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_shortest_path_pd():
    """Random input test for the Shortest Path kernel [n_jobs=-1 (for attributed)/decorator]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    gk = GraphKernel(kernel="SP", verbose=verbose, normalize=normalize)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    train, test = generate_dataset(n_graphs=50,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=20,
                                   random_state=rs,
                                   features=('na', 5))

    gk = GraphKernel(kernel={"name": "SP", "as_attributes": True},
                     verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_graphlet_sampling():
    """Random input test for the Graphlet Sampling Kernel [+ generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    gs_kernel = GraphletSampling(verbose=verbose, normalize=normalize, sampling=dict(n_samples=50))
    gk = GraphKernel(kernel={"name": "GR", "sampling": {"n_samples": 50}},
                     verbose=verbose, normalize=normalize)

    try:
        gs_kernel.fit_transform(train)
        gs_kernel.transform(test)
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_weisfeiler_lehman():
    """Random input test for the Weisfeiler Lehman kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    wl_st_kernel = WeisfeilerLehman(verbose=verbose, normalize=normalize,
                                    base_graph_kernel=VertexHistogram)

    try:
        wl_st_kernel.fit_transform(train)
        wl_st_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_weisfeiler_lehman_optimal_assignment():
    """Random input test for the Weisfeiler Lehman Optimal Assignment kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    wl_oa_kernel = WeisfeilerLehmanOptimalAssignment(
        verbose=verbose, normalize=normalize)

    try:
        wl_oa_kernel.fit_transform(train)
        wl_oa_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_weisfeiler_lehman_pd():
    """Random input test for the Weisfeiler Lehman kernel [n_jobs=-1/generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    gk = GraphKernel(kernel="WL", verbose=verbose, normalize=normalize)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_pyramid_match():
    """Random input test for the Pyramid Match kernel [+ generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    pm_kernel = PyramidMatch(verbose=verbose, normalize=normalize)
    gk = GraphKernel("PM", verbose=verbose, normalize=normalize)

    try:
        pm_kernel.fit_transform(train)
        pm_kernel.transform(test)
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_pyramid_match_no_labels():
    """Random input test for the Pyramid Match kernel with no labels [+ generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=None)

    pm_kernel = PyramidMatch(verbose=verbose, normalize=normalize, with_labels=False)
    gk = GraphKernel(kernel={"name": "PM", "with_labels": False},
                     verbose=verbose, normalize=normalize)

    try:
        pm_kernel.fit_transform(train)
        pm_kernel.transform(test)
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_neighborhood_hash():
    """Random input test for the Neighborhood Hash kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    nh_kernel = NeighborhoodHash(verbose=verbose, normalize=normalize)

    try:
        nh_kernel.fit_transform(train)
        nh_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_neighborhood_hash_pd():
    """Random input test for the Neighborhood Hash kernel [n_jobs=-1/generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3))

    gk = GraphKernel(kernel="NH", verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_subgraph_matching():
    """Random input test for the Subgraph Matching kernel."""
    # node-label/edge-label
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3, 'el', 4))

    sm_kernel = SubgraphMatching(verbose=verbose, normalize=normalize)

    try:
        sm_kernel.fit_transform(train)
        sm_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    # node-label/edge-attribute
    train, test = generate_dataset(n_graphs=50,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=20,
                                   random_state=rs,
                                   features=('nl', 3, 'ea', 5))

    sm_kernel = SubgraphMatching(verbose=verbose, normalize=normalize, ke=np.dot)

    try:
        sm_kernel.fit_transform(train)
        sm_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    # node-attribute/edge-label
    train, test = generate_dataset(n_graphs=50,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=20,
                                   random_state=rs,
                                   features=('na', 4, 'el', 3))

    sm_kernel = SubgraphMatching(verbose=verbose, normalize=normalize, kv=np.dot)

    try:
        sm_kernel.fit_transform(train)
        sm_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    # node-attribute/edge-attribute
    train, test = generate_dataset(n_graphs=50,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=20,
                                   random_state=rs,
                                   features=('na', 4, 'ea', 6))

    sm_kernel = SubgraphMatching(verbose=verbose, normalize=normalize, ke=np.dot, kv=np.dot)

    try:
        sm_kernel.fit_transform(train)
        sm_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_subgraph_matching_pd():
    """Random input test for the Subgraph Matching kernel [n_jobs=-1/generic-wrapper]."""
    # node-label/edge-label
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 3, 'el', 4))

    gk = GraphKernel(kernel={"name": "SM"}, verbose=verbose,
                     normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    # node-label/edge-attribute
    train, test = generate_dataset(n_graphs=50,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=20,
                                   random_state=rs,
                                   features=('nl', 3, 'ea', 5))

    gk = GraphKernel(kernel={"name": "SM", "ke": np.dot},
                     verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    # node-attribute/edge-label
    train, test = generate_dataset(n_graphs=50,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=20,
                                   random_state=rs,
                                   features=('na', 4, 'el', 3))

    gk = GraphKernel(kernel={"name": "SM", "kv": np.dot},
                     verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    # node-attribute/edge-attribute
    train, test = generate_dataset(n_graphs=50,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=20,
                                   random_state=rs,
                                   features=('na', 4, 'ea', 6))

    gk = GraphKernel(kernel={"name": "SM", "kv": np.dot, "ke": np.dot},
                     verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_neighborhood_subgraph_pairwise_distance():
    """Random input test for the Neighborhood Subgraph Pairwise Distance kernel [+ generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 5, 'el', 4))

    nspd_kernel = NeighborhoodSubgraphPairwiseDistance(verbose=verbose, normalize=normalize)
    gk = GraphKernel(kernel="NSPD", verbose=verbose, normalize=normalize)

    try:
        nspd_kernel.fit_transform(train)
        nspd_kernel.transform(test)
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


if cvxopt:
    @pytest.mark.xfail(
        condition=is_windows,
        reason="See https://github.com/ysig/GraKeL/pull/83#issuecomment-1267069069"
    )
    def test_lovasz_theta():
        """Random input test for the Lovasz-theta distance kernel."""
        train, test = generate_dataset(n_graphs=50,
                                       r_vertices=(5, 10),
                                       r_connectivity=(0.4, 0.8),
                                       r_weight_edges=(1, 1),
                                       n_graphs_test=20,
                                       random_state=rs,
                                       features=None)

        lt_kernel = LovaszTheta(verbose=verbose, normalize=normalize)

        try:
            lt_kernel.fit_transform(train)
            lt_kernel.transform(test)
            assert True
        except Exception as exception:
            assert False, exception

    @pytest.mark.xfail(
        condition=is_windows,
        reason="See https://github.com/ysig/GraKeL/pull/83#issuecomment-1267069069"
    )
    def test_lovasz_theta_pd():
        """Random input test for the Lovasz-theta distance kernel [n_jobs=-1/generic-wrapper]."""
        train, test = generate_dataset(n_graphs=50,
                                       r_vertices=(5, 10),
                                       r_connectivity=(0.4, 0.8),
                                       r_weight_edges=(1, 1),
                                       n_graphs_test=20,
                                       random_state=rs,
                                       features=None)

        gk = GraphKernel(kernel="lovasz_theta",
                         verbose=verbose, normalize=normalize, n_jobs=-1)

        try:
            gk.fit_transform(train)
            gk.transform(test)
            assert True
        except Exception as exception:
            assert False, exception


def test_svm_theta():
    """Random input test for the SVM-theta distance kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=None)

    svm_kernel = SvmTheta(verbose=verbose, normalize=normalize)

    try:
        svm_kernel.fit_transform(train)
        svm_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_svm_theta_pd():
    """Random input test for the SVM-theta distance kernel [n_jobs=-1/generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=None)

    gk = GraphKernel(kernel="svm_theta",
                     verbose=verbose, normalize=normalize)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_odd_sth():
    """Random input test for the ODD-STh kernel [+ generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 4))

    odd_sth_kernel = OddSth(verbose=verbose, normalize=normalize)
    gk = GraphKernel(kernel="ODD",
                     verbose=verbose, normalize=normalize)

    try:
        odd_sth_kernel.fit_transform(train)
        odd_sth_kernel.transform(test)
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_propagation():
    """Random input test for the Propagation kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(float("1e-5"), 10),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 4))

    propagation_kernel = Propagation(verbose=verbose, normalize=normalize)

    try:
        propagation_kernel.fit_transform(train)
        propagation_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(float("1e-5"), 10),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('na', 5))

    propagation_kernel_attr = PropagationAttr(verbose=verbose, normalize=normalize)

    try:
        propagation_kernel_attr.fit_transform(train)
        propagation_kernel_attr.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_propagation_pd():
    """Random input test for the Propagation kernel [n_jobs=-1/generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(float("1e-5"), 10),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 4))

    gk = GraphKernel(kernel="PR", verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception

    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(float("1e-5"), 10),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('na', 5))

    gk = GraphKernel(kernel={"name": "PR", "with_attributes": True},
                     verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_hadamard_code():
    """Random input test for the Hadamard Code kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 5))

    hadamard_code_kernel = HadamardCode(verbose=verbose, normalize=normalize,
                                        base_graph_kernel=VertexHistogram)

    try:
        hadamard_code_kernel.fit_transform(train)
        hadamard_code_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_hadamard_code_pd():
    """Random input test for the Hadamard Code kernel [n_jobs=-1/generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 5))

    gk = GraphKernel(kernel="HC",
                     verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_multiscale_laplacian():
    """Random input test for the Multiscale Laplacian kernel."""
    # Initialise kernel
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('na', 5))

    mlf_kernel = MultiscaleLaplacian(verbose=verbose, normalize=normalize)

    try:
        mlf_kernel.fit_transform(train)
        mlf_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_multiscale_laplacian_pd():
    """Random input test for the Multiscale Laplacian kernel [n_jobs=-1/generic-wrapper]."""
    # Initialise kernel
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('na', 5))

    gk = GraphKernel(kernel="ML", verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_vertex_histogram():
    """Random input test for the Vertex Histogram Kernel [+ generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 5))

    vh_kernel = VertexHistogram(verbose=verbose, normalize=normalize)
    gk = GraphKernel(kernel="VH", verbose=verbose, normalize=normalize)

    try:
        vh_kernel.fit_transform(train)
        vh_kernel.transform(test)
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_edge_histogram():
    """Random input test for the Edge Histogram kernel [+ generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('el', 4))

    eh_kernel = EdgeHistogram(verbose=verbose, normalize=normalize)
    gk = GraphKernel(kernel="EH", verbose=verbose, normalize=normalize)

    try:
        eh_kernel.fit_transform(train)
        eh_kernel.transform(test)
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_graph_hopper():
    """Random input test for the Graph Hopper kernel."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('na', 4))

    gh_kernel = GraphHopper(verbose=verbose, normalize=normalize)

    try:
        gh_kernel.fit_transform(train)
        gh_kernel.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_graph_hopper_pd():
    """Random input test for the Graph Hopper kernel [n_jobs=-1/generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('na', 4))

    gk = GraphKernel(kernel="GH",
                     verbose=verbose, normalize=normalize, n_jobs=-1)

    try:
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


def test_graph_hopper_isolated_vertex():
    """GraphHopper survives a graph with an isolated vertex.

    od_vectors_dag squeezed the single row of a one-node shortest path
    DAG into a 0-d array, which numpy >= 2 refuses to feed to np.where,
    so any graph with an isolated vertex crashed the kernel.
    """
    train = [
        (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
         {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([1.0, 1.0])}),
        (np.array([[0.0, 1.0], [1.0, 0.0]]),
         {0: np.array([0.5, 0.5]), 1: np.array([1.0, -1.0])}),
    ]

    gh_kernel = GraphHopper(verbose=verbose, normalize=normalize)

    try:
        gh_kernel.fit_transform(train)
        gh_kernel.transform(train)
        assert True
    except Exception as exception:
        assert False, exception


def test_graph_hopper_single_vertex():
    """GraphHopper accepts a graph with a single vertex.

    The root slice of the occurrence and descendant matrices was squeezed
    into a 0-d array on a single-vertex graph, which no numpy version
    accepts to index -- the kernel always crashed on such graphs.
    """
    train = [
        (np.zeros((1, 1)), {0: np.array([1.0, 0.5])}),
        (np.array([[0.0, 1.0], [1.0, 0.0]]),
         {0: np.array([0.5, 0.5]), 1: np.array([1.0, -1.0])}),
    ]

    gh_kernel = GraphHopper(verbose=verbose, normalize=normalize)

    try:
        gh_kernel.fit_transform(train)
        gh_kernel.transform(train)
        assert True
    except Exception as exception:
        assert False, exception


def test_graph_hopper_kernel_type_callable():
    """A callable kernel_type follows the documented tuple contract.

    The contract, as described in the GraphHopper docstring: a function that
    takes two tuples of np.arrays, the flattened M matrix and the attribute
    matrix of each graph, and returns a number. Reproducing the built-in
    linear kernel through that contract must give the same matrix.
    """
    train, test = generate_dataset(n_graphs=20,
                                   r_vertices=(5, 10),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=8,
                                   random_state=rs,
                                   features=('na', 3))

    def linear_through_contract(x, y):
        M_i, NA_i = x
        M_j, NA_j = y
        return np.dot(np.dot(M_i, M_j.T).flat, np.dot(NA_i, NA_j.T).flat)

    gh_callable = GraphHopper(kernel_type=linear_through_contract,
                              verbose=verbose, normalize=normalize)
    gh_linear = GraphHopper(kernel_type="linear",
                            verbose=verbose, normalize=normalize)

    try:
        K_callable = gh_callable.fit_transform(train)
        K_linear = gh_linear.fit_transform(train)
        assert np.allclose(K_callable, K_linear)
        assert np.allclose(gh_callable.transform(test), gh_linear.transform(test))
    except Exception as exception:
        assert False, exception


def _enumerate_shortest_paths(adj):
    """All shortest paths of a unit-weight graph, as tuples of vertices.

    For every source, every shortest path to each reachable vertex, each
    path counted once per multiplicity, including the trivial
    one-vertex path of the source with itself.
    """
    n = adj.shape[0]
    all_paths = []
    for s in range(n):
        dist = {s: 0}
        order = [s]
        frontier = [s]
        while frontier:
            nxt = []
            for x in frontier:
                for y in range(n):
                    if adj[x, y] > 0 and y not in dist:
                        dist[y] = dist[x] + 1
                        nxt.append(y)
            order.extend(nxt)
            frontier = nxt
        paths_from_s = {s: [(s,)]}
        for t in order[1:]:
            parents = [x for x in range(n) if adj[x, t] > 0 and dist.get(x) == dist[t] - 1]
            paths_from_s[t] = tuple(p + (t,) for x in parents for p in paths_from_s[x])
        for plist in paths_from_s.values():
            all_paths.extend(plist)
    return all_paths


def _brute_force_graph_hopper(graphs, node_kernel):
    """A reference GraphHopper: hop-matched pairs of shortest paths.

    Sums ``node_kernel`` over all pairs of shortest paths of equal
    discrete length, position by position. Independent of the M matrix
    machinery used by the implementation.
    """
    parsed = []
    for adj, labels in graphs:
        values = [labels[j] for j in range(adj.shape[0])]
        parsed.append((_enumerate_shortest_paths(adj), values))
    K = np.zeros((len(graphs), len(graphs)))
    for i, (P, vi) in enumerate(parsed):
        for j, (Q, vj) in enumerate(parsed):
            total = 0.0
            for p in P:
                for q in Q:
                    if len(p) == len(q):
                        total += sum(node_kernel(vi[a], vj[b]) for a, b in zip(p, q))
            K[i, j] = total
    return K


def _issue118_node_kernel(u, v):
    """The node kernel requested in GraKeL issue #118.

    Averages a delta kernel on the discrete label with a gaussian
    similarity on the continuous features.
    """
    label_u, feat_u = u
    label_v, feat_v = v
    return 0.5 * (float(label_u == label_v) + np.exp(-np.linalg.norm(feat_u - feat_v)))


def _mixed_label_attribute_graphs():
    """Small graphs whose nodes carry (discrete label, feature vector) pairs."""
    return [
        # a triangle with one pendant vertex, diameter 2
        (np.array([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0],
                   [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
         {0: ("a", np.array([1.0, 0.0])), 1: ("b", np.array([0.0, 1.0])),
          2: ("a", np.array([1.0, 1.0])), 3: ("b", np.array([-1.0, 0.0]))}),
        # a path of three vertices, diameter 2
        (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
         {0: ("b", np.array([0.5, 0.5])), 1: ("a", np.array([1.0, -1.0])),
          2: ("b", np.array([0.0, 1.0]))}),
        # two vertices plus an isolated one, diameter 1
        (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
         {0: ("a", np.array([2.0, 2.0])), 1: ("a", np.array([1.0, 0.0])),
          2: ("c", np.array([0.0, 0.0]))}),
        # a single vertex
        (np.zeros((1, 1)), {0: ("d", np.array([0.3, 0.7]))}),
    ]


def test_graph_hopper_node_kernel_matches_reference():
    """A custom node kernel agrees with a brute-force path enumeration.

    The reference sums the node kernel over position-matched pairs of
    shortest paths, enumerated independently of the implementation. It
    must reproduce the built-in linear kernel on attribute graphs (which
    validates the reference) and the custom node kernel path with a dot
    product node kernel (which validates the plumbing).
    """
    train, test = generate_dataset(n_graphs=6,
                                   r_vertices=(3, 6),
                                   r_connectivity=(0.3, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=2,
                                   random_state=rs,
                                   features=('na', 2))

    dot_kernel = lambda u, v: float(np.dot(u, v))  # noqa: E731

    try:
        K_linear = GraphHopper(kernel_type="linear").fit_transform(train)
        assert np.allclose(K_linear, _brute_force_graph_hopper(train, dot_kernel))

        K_custom = GraphHopper(node_kernel=dot_kernel).fit_transform(train)
        assert np.allclose(K_custom, K_linear)

        # the transform side, on graphs of diameters unseen in fit
        K_ref = _brute_force_graph_hopper(train + test, dot_kernel)
        gh = GraphHopper(node_kernel=dot_kernel)
        gh.fit(train)
        assert np.allclose(gh.transform(test), K_ref[len(train):, :len(train)])
    except Exception as exception:
        assert False, exception


def test_graph_hopper_node_kernel_mixed_labels_and_attributes():
    """The node kernel of issue #118 on graphs with labels AND attributes.

    Nodes carry (discrete label, continuous feature) pairs, which no
    built-in kernel of GraKeL can compare; the values must reach the
    callable exactly as they were stored on the graph.
    """
    graphs = _mixed_label_attribute_graphs()

    # the callable itself must be symmetric
    for i, (adj, labels) in enumerate(graphs):
        for u in labels.values():
            for v in labels.values():
                assert _issue118_node_kernel(u, v) == _issue118_node_kernel(v, u)

    try:
        K = GraphHopper(node_kernel=_issue118_node_kernel).fit_transform(graphs)
        assert np.allclose(K, _brute_force_graph_hopper(graphs, _issue118_node_kernel))
    except Exception as exception:
        assert False, exception


def test_graph_hopper_node_kernel_discrete_labels():
    """A delta node kernel makes discrete labels usable in GraphHopper."""
    graphs = [
        (np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
         {0: "a", 1: "b", 2: "a"}),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), {0: "a", 1: "a"}),
    ]
    delta_kernel = lambda u, v: float(u == v)  # noqa: E731

    try:
        K = GraphHopper(node_kernel=delta_kernel, normalize=normalize).fit_transform(graphs)
        assert K.shape == (2, 2)
        # the reference sees the same strings
        assert np.allclose(GraphHopper(node_kernel=delta_kernel).fit_transform(graphs),
                           _brute_force_graph_hopper(graphs, delta_kernel))
    except Exception as exception:
        assert False, exception


def test_graph_hopper_node_kernel_validation():
    """A misbehaving node kernel is reported with its node pair."""
    graphs = _mixed_label_attribute_graphs()

    # not a callable
    with pytest.raises(TypeError, match="node_kernel"):
        GraphHopper(node_kernel="linear").fit(graphs)

    # cannot steal the place of an explicit kernel_type
    with pytest.raises(ValueError, match="node_kernel"):
        GraphHopper(node_kernel=_issue118_node_kernel, kernel_type="gaussian").fit(graphs)

    # fit only parses the input: the callable runs on the first real
    # computation, and every returned value is validated there
    gh = GraphHopper(node_kernel=lambda u, v: np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="node at position"):
        gh.fit_transform(graphs)

    gh = GraphHopper(node_kernel=lambda u, v: 1 + 2j)
    with pytest.raises(TypeError, match="node at position"):
        gh.fit_transform(graphs)

    gh = GraphHopper(node_kernel=lambda u, v: float("nan"))
    with pytest.raises(ValueError, match="finite"):
        gh.fit_transform(graphs)

    # a callable that raises keeps the original exception as its cause
    def exploding(u, v):
        raise ZeroDivisionError("boom")

    gh = GraphHopper(node_kernel=exploding)
    with pytest.raises(TypeError, match="node_kernel failed") as excinfo:
        gh.fit_transform(graphs)
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


def test_graph_hopper_node_kernel_sklearn():
    """node_kernel behaves as a regular scikit-learn parameter."""
    graphs = _mixed_label_attribute_graphs()

    # a delta kernel on the discrete label of the (label, feature) tuples
    def label_delta_kernel(u, v):
        return float(u[0] == v[0])

    gh = GraphHopper(node_kernel=_issue118_node_kernel, normalize=True)
    assert "node_kernel" in gh.get_params()

    # get_params/set_params round trip with a different callable
    K_first = gh.fit_transform(graphs)
    gh.set_params(node_kernel=label_delta_kernel)
    gh.fit(graphs)
    K_second = gh.transform(graphs)

    fresh = GraphHopper(node_kernel=label_delta_kernel, normalize=True)
    fresh.fit(graphs)
    assert np.allclose(K_second, fresh.transform(graphs))
    assert not np.allclose(K_second, K_first)

    # clone keeps the callable and reproduces the same matrix
    from sklearn.base import clone
    K_clone = clone(gh).fit_transform(graphs)
    assert np.allclose(K_clone, K_second)

    # the generic wrapper accepts node_kernel as a kernel parameter
    gk = GraphKernel(kernel={"name": "graph_hopper", "node_kernel": _issue118_node_kernel},
                     verbose=verbose, normalize=normalize)
    try:
        K_wrapped = gk.fit_transform(graphs)
        assert np.allclose(K_wrapped,
                           GraphHopper(node_kernel=_issue118_node_kernel,
                                       normalize=normalize).fit_transform(graphs))
    except Exception as exception:
        assert False, exception

    # parallel evaluation agrees with the sequential one
    K_jobs = GraphHopper(node_kernel=_issue118_node_kernel, n_jobs=2,
                         normalize=normalize).fit_transform(graphs)
    assert np.allclose(K_jobs,
                       GraphHopper(node_kernel=_issue118_node_kernel,
                                   normalize=normalize).fit_transform(graphs))


def test_refit_transform_normalized():
    """Refit clears the diagonal cached by a previous fit.

    Kernel.fit used to leave _X_diag in place, so transform(normalize=True)
    after a refit either crashed on a broadcast mismatch (different dataset
    sizes) or silently normalized against the stale diagonal (same sizes).
    """
    graphs = [
        (np.array([[0, 1], [1, 0]]), {0: "a", 1: "a"}),
        (np.array([[0, 1], [1, 0]]), {0: "a", 1: "b"}),
        (np.array([[0, 1], [1, 0]]), {0: "b", 1: "b"}),
    ]

    def as_fresh(train, test):
        fresh = VertexHistogram(normalize=True)
        fresh.fit(train)
        return fresh.transform(test)

    try:
        # same sizes after refit: values must match a fresh estimator
        reused = VertexHistogram(normalize=True)
        reused.fit_transform(graphs[:2])
        reused.fit(graphs[1:])
        assert np.allclose(reused.transform(graphs[:2]),
                           as_fresh(graphs[1:], graphs[:2]))

        # different sizes after refit: no broadcast error
        reused = VertexHistogram(normalize=True)
        reused.fit_transform(graphs)
        reused.fit(graphs[:2])
        assert np.allclose(reused.transform(graphs[1:]),
                           as_fresh(graphs[:2], graphs[1:]))
    except Exception as exception:
        assert False, exception


def test_core_framework():
    """Random input test for the Core kernel Framework [+ generic-wrapper]."""
    train, test = generate_dataset(n_graphs=100,
                                   r_vertices=(10, 20),
                                   r_connectivity=(0.4, 0.8),
                                   r_weight_edges=(1, 1),
                                   n_graphs_test=40,
                                   random_state=rs,
                                   features=('nl', 4))

    base_graph_kernel = (WeisfeilerLehman, dict(base_graph_kernel=VertexHistogram))
    core_framework = CoreFramework(verbose=verbose, normalize=normalize, base_graph_kernel=base_graph_kernel)

    kernel = ["CORE", "WL"]
    gk = GraphKernel(kernel=kernel, verbose=verbose, normalize=normalize)
    try:
        core_framework.fit_transform(train)
        core_framework.transform(test)
        gk.fit_transform(train)
        gk.transform(test)
        assert True
    except Exception as exception:
        assert False, exception


if __name__ == "__main__":
    warnings.filterwarnings("once")
    verbose = True

    test_random_walk()
    test_random_walk_pd()
    test_random_walk_labels()
    test_random_walk_labels_pd()
    test_shortest_path()
    test_shortest_path_pd()
    test_graphlet_sampling()
    test_weisfeiler_lehman()
    test_weisfeiler_lehman_optimal_assignment()
    test_weisfeiler_lehman_pd()
    test_pyramid_match()
    test_pyramid_match_no_labels()
    test_neighborhood_hash()
    test_neighborhood_hash_pd()
    test_subgraph_matching()
    test_subgraph_matching_pd()
    test_neighborhood_subgraph_pairwise_distance()
    if cvxopt:
        test_lovasz_theta()
        test_lovasz_theta_pd()
    test_svm_theta()
    test_svm_theta_pd()
    test_odd_sth()
    test_propagation()
    test_propagation_pd()
    test_hadamard_code()
    test_hadamard_code_pd()
    test_multiscale_laplacian()
    test_multiscale_laplacian_pd()
    test_vertex_histogram()
    test_edge_histogram()
    test_graph_hopper()
    test_graph_hopper_pd()
    test_graph_hopper_isolated_vertex()
    test_graph_hopper_single_vertex()
    test_graph_hopper_kernel_type_callable()
    test_graph_hopper_node_kernel_matches_reference()
    test_graph_hopper_node_kernel_mixed_labels_and_attributes()
    test_graph_hopper_node_kernel_discrete_labels()
    test_graph_hopper_node_kernel_validation()
    test_graph_hopper_node_kernel_sklearn()
    test_refit_transform_normalized()
    test_core_framework()
