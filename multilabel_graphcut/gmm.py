"""Gaussian mixture model"""
from typing import List, Optional, Tuple
import numpy as np
import scipy.stats
from scipy.special import logsumexp


BGRMean = np.ndarray
BGRCovarianceMatrix = np.ndarray
Model = Tuple[List[Tuple[BGRMean, ...]], List[Tuple[BGRCovarianceMatrix, ...]]]


def fit_model(
    vs: np.ndarray,
    ls: np.ndarray,
    n_class: int,
    model: Model,
) -> Optional[Model]:
    """Fit gaussian mixture using EM-like momentum method.

    Args:
        vs (np.ndarray): values of shape (n_data, n_dimension)
        ls (np.ndarray): known labels of shape (n_data,), whose values are in range [0, n_class - 1]
        model (Model): model to use as a starting point

    Returns:
        None if fitting failed for whatever reason.
        Otherwise, return means, covariance matrices, mix coefficients,
            whose shapes are (n_gmm, C), (n_gmm, C, C), (n_gmm,) respectively.
    """
    # pylint: disable=too-many-locals,invalid-name
    n_gmm = 4  # hardcode :)

    # Interpret input params
    if model is not None:
        center_ini, cov_ini, _ = model
        assert center_ini.shape[1] == cov_ini.shape[1]
        n_gmm = center_ini.shape[1]
    else:
        if n_gmm <= 8:
            center_ini = np.array([
                (
                    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0,),
                )[:n_gmm]
                for _ in range(n_class)
            ], dtype=np.float32)
        else:
            raise NotImplementedError(n_gmm)
        cov_ini = [
            tuple(128 * 128 * np.identity(vs.shape[-1]) for _ in range(n_gmm))
            for _ in range(n_class)
        ]
    # Interpret input params done

    label_means = []
    label_covars = []
    mixs = []
    for i in range(n_class):
        bgrs = vs[ls == i]
        if len(bgrs) <= 2:
            print(f"Need more label on class {i}")
            return None

        solver = solve(
            bgrs,
            n_gmm=n_gmm,
            gmm_center_ini=center_ini[i],
            gmm_cov_ini=cov_ini[i],
        )
        for _ in range(10):
            _, _ = next(solver)
            gmm_centers, gmm_covs = next(solver)
        _, z = next(solver)
        label_means += [gmm_centers]
        label_covars += [gmm_covs]
        mixs += [z.sum(axis=1)]
    label_means = np.array(label_means)
    label_covars = np.array(label_covars)
    mixs = np.array(mixs)
    return label_means, label_covars, mixs


def pixelwise_likelihood(vs: np.ndarray, weights: np.ndarray, model: Model) -> np.ndarray:
    """Return pixelwise likelihood

    Args:
        vs (np.ndarray):      values of shape (n_data, n_dimension)
        weights (np.ndarray): sample size used to derive each value of given values.
                              e.g. Area of superpixels
        model (Model):        model to be used for calculating likelihood

    Returns:
        np.ndarray: N x label
    """
    label_means, label_covars, mixs = model
    dev = vs[:, None, None, :] - label_means[None, :, :, :]  # (sites, label, gmm, channel)
    unary = 0.5 * (
        dev[:, :, :, None, :] @
        np.linalg.inv(label_covars)[None, :, :, :, :] @
        dev[:, :, :, :, None]
    )[:, :, :, 0, 0] * weights[:, None, None]
    unary = - logsumexp(- unary, b=mixs[None, :, :], axis=(2,))  # pylint: disable=invalid-unary-operand-type
    return unary


def solve(
    rgbs: np.ndarray,
    n_gmm: int,
    gmm_center_ini: np.ndarray,
    gmm_cov_ini: np.ndarray,
):
    """_summary_

    Args:
        rgbs (np.ndarray): of shape (n_data, n_dim)
        n_gmm (int):
        gmm_center_ini (np.ndarray): (n_gmm, n_dim)
        gmm_cov_ini (np.ndarray): (n_gmm, n_dim, n_dim)

    Yields:
        _type_: _description_
    """
    # pylint: disable=invalid-name
    gmm_center = np.array(gmm_center_ini, dtype=np.float32)
    gmm_cov = np.array(gmm_cov_ini, dtype=np.float32)
    del gmm_center_ini, gmm_cov_ini

    n_dim = rgbs.shape[-1]
    assert n_gmm == gmm_center.shape[0], "wrong input"
    assert n_gmm == gmm_cov.shape[0], "wrong input"
    assert n_dim == gmm_center.shape[1], "wrong input"
    assert n_dim == gmm_cov.shape[1], "wrong input"
    assert n_dim == gmm_cov.shape[2], "wrong input"

    nll = np.zeros((n_gmm, len(rgbs)))

    while True:
        for i_gmm in range(n_gmm):
            nll[i_gmm, :] = - scipy.stats.multivariate_normal.logpdf(
                rgbs[0],
                mean=gmm_center[i_gmm],
                cov=gmm_cov[i_gmm]
            )

        z = np.exp(- nll)
        z /= z.sum(axis=0)
        yield nll, z
        z_sum = z.sum(axis=1)

        # debug start
        # total = z_sum.sum()
        # mixing_coeff = z_sum / total
        # print("ratio:", '\t'.join([f"{_z:.1%}" for _z in mixing_coeff]))
        # print("average nll:", - np.log((np.exp(- nll) * mixing_coeff[:, None]).sum(axis=0)).mean())
        # debug end
        for i_gmm in range(n_gmm):
            if z_sum[i_gmm] < 1e-7:
                raise RuntimeError("Overfit")
            mean = (rgbs * z[i_gmm, :, None]).sum(axis=0) / z_sum[i_gmm]
            dev = rgbs - mean
            cov = dev.T @ (dev * z[i_gmm, :, None]) / z_sum[i_gmm]

            # slightly slower convergence
            gmm_center[i_gmm] = 0.8 * gmm_center[i_gmm] + 0.2 * mean
            gmm_cov[i_gmm] = 0.8 * gmm_cov[i_gmm] + 0.2 * cov
        if np.any(np.linalg.eigvals(cov) <= 0):
            # numerical error; indicates that overfitting is happening.
            raise RuntimeError("Numerical Error")

        # debug start
        # for i in range(i_gmm):
        #     c = gmm_center[i].tolist()
        #     max_eig = np.sqrt(np.linalg.eigvals(gmm_cov[i])).max()
        #     print(f"{i}: {c}, {max_eig}")
        # debug end
        yield gmm_center, gmm_cov
