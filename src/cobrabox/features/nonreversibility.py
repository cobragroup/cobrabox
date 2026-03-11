"""Non-reversibility feature: normalised deviation from causal normality (dc_norm)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import xarray as xr

from ..base_feature import BaseFeature
from ..data import Data, SignalData


@dataclass
class Nonreversibility(BaseFeature[SignalData]):
    """Compute dc_norm: normalised deviation from causal normality (time-irreversibility).

    Fits a VAR(1) model in forward and reverse time directions:

        Forward:  X_t = A @ X_{t-1} + noise
        Reverse:  X_t = B @ X_{t+1} + noise

    Both A and B are rescaled so their spectral radius is strictly less than 1.
    dc_norm quantifies the asymmetry between forward and reverse dynamics:

        dc_norm = ||A - B^T||_F / (||A + B^T||_F + ||A - B^T||_F)

    Result is bounded in [0, 1). A dc_norm of 0 indicates perfect
    time-reversibility; larger values indicate stronger irreversibility.

    Args:
        None

    Returns:
        xarray DataArray with a single spatial coordinate ('dc_norm').
        The time dimension is removed.

    Raises:
        ValueError: If 'time' dimension is missing or has fewer than 2 timepoints.
        ValueError: If 'space' dimension has fewer than 2 channels.

    Example:
        >>> result = cb.feature.Nonreversibility().apply(data)

    References:
        TODO: Add citation for the dc_norm / time-irreversibility VAR(1) measure.
    """

    output_type: ClassVar[type[Data]] = Data

    @staticmethod
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

    @staticmethod
    def _rescale_to_unit_spectral_radius(M: np.ndarray) -> np.ndarray:
        """Scale M so its largest eigenvalue magnitude is strictly less than 1."""
        sr = np.max(np.abs(np.linalg.eigvals(M)))
        if sr >= 1.0:
            M = M / sr
        return M

    def _compute_dc_norm(self, X: np.ndarray) -> float:
        """Compute dc_norm for a single (n_channels, n_time) segment.

        dc_norm = ||A - B^T||_F / (||A + B^T||_F + ||A - B^T||_F)

        where A is the forward VAR(1) coefficient matrix and B is the reverse
        VAR(1) coefficient matrix, both rescaled to have spectral radius < 1.
        Result is bounded in [0, 1).

        Args:
            X: Array of shape (n_channels, n_time).

        Returns:
            Scalar dc_norm value in [0, 1). Returns 0.0 if denominator is zero.

        Raises:
            ValueError: If n_time < 2.
        """
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 timepoints to fit VAR(1).")

        A = self._rescale_to_unit_spectral_radius(self._fit_var1(X))
        B = self._rescale_to_unit_spectral_radius(self._fit_var1(X[:, ::-1]))

        diff_norm = np.linalg.norm(A - B.T, "fro")
        sum_norm = np.linalg.norm(A + B.T, "fro")
        denom = sum_norm + diff_norm
        if denom < 1e-12:
            return 0.0
        return float(diff_norm / denom)

    def __call__(self, data: SignalData) -> xr.DataArray:
        xr_data = data.data  # time is always last (Data.__init__ transposes it)

        if "space" not in xr_data.dims:
            raise ValueError("data must have a 'space' dimension")

        if xr_data.sizes["space"] < 2:
            raise ValueError(
                "nonreversibility requires at least 2 time series (space dimension >= 2)"
            )

        X = xr_data.values  # shape (n_space, n_time)
        dc = self._compute_dc_norm(X)
        return xr.DataArray([dc], dims=["space"], coords={"space": ["dc_norm"]})
