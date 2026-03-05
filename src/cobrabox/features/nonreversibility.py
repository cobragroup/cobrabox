"""Non-reversibility feature: deviation from normality (d_norm)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData


def _fit_var1(X: np.ndarray) -> np.ndarray:
    """Fit VAR(1) coefficient matrix A for forward model X_t = A @ X_{t-1}.

    Args:
        X: Array of shape (n_channels, n_time).

    Returns:
        Coefficient matrix A of shape (n_channels, n_channels).
    """
    X_t = X[:, 1:]  # current:  (n, T-1)
    X_tm1 = X[:, :-1]  # lag-1:    (n, T-1)
    # Solve X_t.T = X_tm1.T @ A.T in least-squares sense
    A_T, _, _, _ = np.linalg.lstsq(X_tm1.T, X_t.T, rcond=None)
    return A_T.T  # (n, n)


def _rescale_to_unit_spectral_radius(M: np.ndarray) -> np.ndarray:
    """Scale M so its largest eigenvalue magnitude is strictly less than 1."""
    sr = np.max(np.abs(np.linalg.eigvals(M)))
    if sr >= 1.0:
        M = M / sr
    return M


def _compute_d_norm(X: np.ndarray) -> float:
    """Compute d_norm for a single (n_channels, n_time) segment.

    d_norm = ||A A^T - A^T A||_F / ||A A^T + A^T A||_F

    Measures how far the VAR(1) coefficient matrix A deviates from normality.
    A normal matrix satisfies A A^T = A^T A. A d_norm of 0 means A is normal;
    larger values indicate greater deviation.

    Args:
        X: Array of shape (n_channels, n_time).

    Returns:
        Scalar d_norm value in [0, inf). Returns 0.0 if A A^T + A^T A is zero.

    Raises:
        ValueError: If n_time < 2.
    """
    if X.shape[1] < 2:
        raise ValueError("Need at least 2 timepoints to fit VAR(1).")

    A = _rescale_to_unit_spectral_radius(_fit_var1(X))

    AAT = A @ A.T
    ATA = A.T @ A

    diff = AAT - ATA
    total = AAT + ATA
    denom = np.linalg.norm(total, "fro")
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(diff, "fro") / denom)


@dataclass
class Nonreversibility(BaseFeature[SignalData]):
    """Compute d_norm: deviation from normality of the VAR(1) coefficient matrix.

    Fits a VAR(1) model in the forward time direction:

        X_t = A @ X_{t-1} + noise

    A is rescaled so its spectral radius is strictly less than 1. A matrix is
    normal when it commutes with its transpose (A A^T = A^T A). The d_norm
    metric quantifies how far A deviates from normality:

        d_norm = ||A A^T - A^T A||_F / ||A A^T + A^T A||_F

    A d_norm of 0 indicates A is a normal matrix; larger values indicate
    greater deviation from normality (stronger time-irreversibility).

    Args:
        None

    Returns:
        xarray DataArray with a single spatial coordinate ('d_norm').
        The time dimension is removed.

    Raises:
        ValueError: If 'time' dimension is missing or has fewer than 2 timepoints.
        ValueError: If 'space' dimension has fewer than 2 channels.

    Example:
        >>> result = cb.feature.Nonreversibility().apply(data)
    """

    output_type: ClassVar[type[Data]] = Data

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data  # time is always last (Data.__init__ transposes it)

        if "time" not in xr_data.dims:
            raise ValueError("data must have 'time' dimension")

        if xr_data.sizes["space"] < 2:
            raise ValueError(
                "nonreversibility requires at least 2 time series (space dimension >= 2)"
            )

        X = xr_data.values  # shape (n_space, n_time)
        d = _compute_d_norm(X)
        return xr.DataArray(
            [d],
            dims=["space"],
            coords={"space": ["d_norm"]},
        )
