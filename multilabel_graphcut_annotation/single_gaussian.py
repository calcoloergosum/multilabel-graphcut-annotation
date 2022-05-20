"""Single gaussian fitting. Mainly used for testing"""
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

Model = Tuple[
    npt.NDArray[np.float64],  # mean
    npt.NDArray[np.float64],  # covariance
]


def fit_model(
    vs: npt.NDArray[np.float64],
    ls: npt.NDArray[np.float64],
    n_class: int
) -> Optional[Model]:
    """Calculate parameters in MLE manner.
    vs: values of shape (n_data, n_dimension)
    ls: known labels of shape (n_data,), whose values are in range [0, n_class - 1]
    """
    label_means = []
    label_covars = []
    for i in range(n_class):
        bgrs = vs[ls == i]
        if len(bgrs) <= 2:
            print("Need more label")
            return None
        mean = bgrs.mean(axis=0)
        dev = bgrs - mean
        covar = dev.T @ dev / len(bgrs)
        label_means.append(mean)
        label_covars.append(covar)
    return np.array(label_means), np.array(label_covars)


def pixelwise_likelihood(
    image: npt.NDArray[np.float64],
    model: Model
) -> npt.NDArray[np.float64]:
    """Return pixelwise likelihood

    Args:
        image (np.ndarray): of shape (N, C)
        model (Model): model

    Returns:
        np.ndarray: N x label
    """
    label_means, label_covars = model
    dev = image[:, :, None, :] - label_means[None, None, :, :]  # type: ignore
    unary = 0.5 * (
        dev[:, :, :, None, :] @
        np.linalg.inv(label_covars)[None, None, :, :, :] @  # type: ignore
        dev[:, :, :, :, None]
    )[:, :, :, 0, 0]
    return unary
