import numpy as np
import pytest
from multilabel_graphcut_annotation.gmm import solve


@pytest.mark.parametrize('pts, gmm_center_ini, gmm_cov_ini, gmm_center_exp, gmm_cov_exp', [
    (
        np.vstack((
            0  + np.random.normal(size=(1000, 1)),
            10 + np.random.normal(size=(1000, 1)),
        )),
        ((-1,), (3,),),
        (((1,),), ((2,),)),
        ((0,), (10,),),
        (((1,),), ((1,),),),
    ),
    (
        np.vstack((
            np.random.normal(size=(100000, 2)),
            10 + np.random.normal(size=(100000, 2)),
        )),
        ((-1, -1), (100, 100,)),
        (((1, 0), (0, 1)), ((1e4, 0,), (0, 1e4,))),
        ((0, 0), (10, 10)),
        (((1, 0), (0, 1)), ((1, 0), (0, 1))),
    ),
])
def test_gmm(pts, gmm_center_ini, gmm_cov_ini, gmm_center_exp, gmm_cov_exp,):
    gmm_center_ini = np.array(gmm_center_ini)
    gmm_cov_ini = np.array(gmm_cov_ini)
    gmm_center_exp = np.array(gmm_center_exp)
    gmm_cov_exp = np.array(gmm_cov_exp)

    n_data = pts.shape[0]
    n_dim = pts.shape[-1]
    n_gmm = gmm_center_ini.shape[0]
    assert n_gmm == gmm_center_ini.shape[0], "test wrong"
    assert n_gmm == gmm_cov_ini.shape[0], "test wrong"
    assert n_dim == gmm_center_ini.shape[1], "test wrong"
    assert n_dim == gmm_cov_ini.shape[1], "test wrong"
    assert n_dim == gmm_cov_ini.shape[2], "test wrong"

    solver = solve(
        pts,
        n_gmm=n_gmm,
        gmm_center_ini=gmm_center_ini,
        gmm_cov_ini=gmm_cov_ini,
    )
    _, _ = next(solver)
    _gmm_centers, _gmm_cov = next(solver)
    while True:
        _, _ = next(solver)
        gmm_centers, gmm_cov = next(solver)
        dev = gmm_centers - _gmm_centers
        if np.sqrt(
            np.linalg.norm(
                dev[:, None, :] @
                np.linalg.inv(gmm_cov[:, :, :]) @
                dev[:, :, None]
            )
        ).min(axis=0) < 1e-4 and np.sqrt(((gmm_cov - _gmm_cov) ** 2).sum()) < 1e-4:
            break
        _gmm_centers, _gmm_cov = gmm_centers, gmm_cov

    np.testing.assert_allclose(gmm_centers, gmm_center_exp, atol=1e-2)
    np.testing.assert_allclose(gmm_cov, gmm_cov_exp, atol=1e-2)
