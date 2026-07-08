"""Shared multivariate-autoregressive (MVAR) utilities.

Used by GrangerCausality, PartialDirectedCoherence, DirectedTransferFunction,
and (indirectly) Nonreversibility. Centralizes VAR fitting and frequency-domain
derivations so the connectivity domain has one canonical implementation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from statsmodels.tsa.api import VAR


def fit_mvar(
    X: np.ndarray,
    order: int | None = None,
    method: Literal["auto", "lstsq", "statsmodels"] = "auto",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit a vector-autoregressive (VAR) model of order *p* to multichannel data.

    Args:
        X: Array of shape ``(K, T)`` — K channels by T samples — or ``(T, K)``
            if K < T. The function auto-orients so the longer axis is treated
            as time.
        order: Lag order p. If ``None``, statsmodels selects the optimal lag
            using AIC.
        method: Which fitting routine to use.

            * ``"auto"`` — pick statsmodels for K > 1 (multivariate) or for
              auto order selection; pick lstsq for K == 1 with explicit order.
            * ``"statsmodels"`` — always use :class:`statsmodels.tsa.api.VAR`.
            * ``"lstsq"`` — use a stacked-OLS solve (requires ``order`` to be
              an int).

    Returns:
        ``(coefs, resid_cov, order)`` where ``coefs`` has shape ``(p, K, K)``
        with ``coefs[k, i, j] = A_{k+1}[i, j]``, ``resid_cov`` is the
        ``(K, K)`` residual covariance, and ``order`` is the lag order used.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"fit_mvar requires a 2-D array, got shape {X.shape}")

    if X.shape[0] > X.shape[1]:
        X = X.T
    K, T = X.shape

    use_statsmodels = (method == "statsmodels") or (method == "auto" and (K > 1 or order is None))

    if use_statsmodels:
        var_model = VAR(X.T)
        if order is None:
            fit = var_model.fit(ic="aic")
        else:
            fit = var_model.fit(maxlags=order)
        return fit.coefs, np.asarray(fit.sigma_u), fit.coefs.shape[0]

    if order is None:
        raise ValueError("method='lstsq' requires an explicit order")

    p = int(order)
    if p < 1:
        raise ValueError(f"order must be >= 1, got {p}")

    Y = X[:, p:].T  # (T - p, K)
    Z = np.column_stack([X[:, p - k - 1 : T - k - 1].T for k in range(p)])  # (T-p, p*K)

    coefs_flat, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    coefs = coefs_flat.reshape(p, K, K).transpose(0, 2, 1)
    resid = Y - Z @ coefs_flat
    resid_cov = (resid.T @ resid) / (T - p)
    return coefs, resid_cov, p


def frequency_response(coefs: np.ndarray, n_freqs: int, sr: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``A(f) = I - sum_k A_k exp(-2πi k f / sr)`` for a VAR model.

    Args:
        coefs: VAR coefficient tensor of shape ``(p, K, K)``.
        n_freqs: Number of frequency bins, linearly spaced over ``[0, sr/2]``.
        sr: Sampling rate (Hz).

    Returns:
        ``(A_f, freqs)`` where ``A_f`` has shape ``(n_freqs, K, K)`` and ``freqs``
        is a 1-D array of length ``n_freqs``.
    """
    p, K, _ = coefs.shape
    freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    np.einsum("fii->fi", A_f)[:] = 1.0
    for k in range(1, p + 1):
        phase = np.exp(-2j * np.pi * k * freqs / sr)
        A_f -= coefs[k - 1][np.newaxis, :, :] * phase[:, np.newaxis, np.newaxis]
    return A_f, freqs


def pdc_from_A(A_f: np.ndarray) -> np.ndarray:
    """Column-normalize ``A(f)`` to obtain partial directed coherence.

    ``PDC[i, j, f] = |A(f)[i, j]| / sqrt(Σ_m |A(f)[m, j]|²)``.
    """
    A_abs = np.abs(A_f)
    col_norm = np.sqrt((A_abs**2).sum(axis=1, keepdims=True))
    col_norm = np.where(col_norm == 0.0, 1.0, col_norm)
    return A_abs / col_norm


def dtf_from_A(A_f: np.ndarray) -> np.ndarray:
    """Row-normalize ``H(f) = A(f)⁻¹`` to obtain the directed transfer function.

    ``DTF[i, j, f] = |H(f)[i, j]| / sqrt(Σ_m |H(f)[i, m]|²)``.
    """
    H_f = np.linalg.inv(A_f)
    H_abs = np.abs(H_f)
    row_norm = np.sqrt((H_abs**2).sum(axis=2, keepdims=True))
    row_norm = np.where(row_norm == 0.0, 1.0, row_norm)
    return H_abs / row_norm


def granger_log_ratio(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float, float, float]:
    """Compute log-ratio Granger causality from y to x at the given lag.

    Returns ``(GC, ssr_restricted, ssr_unrestricted, variance_ratio)`` where
    ``GC = log(ssr_r / ssr_u)``.
    """
    n = len(x)
    X_restricted = np.column_stack([x[lag - i : n - i] for i in range(1, lag + 1)])
    X_unrestricted = np.column_stack(
        [x[lag - i : n - i] for i in range(1, lag + 1)]
        + [y[lag - i : n - i] for i in range(1, lag + 1)]
    )
    y_target = x[lag:]

    coeffs_r, _, _, _ = np.linalg.lstsq(X_restricted, y_target, rcond=None)
    residuals_r = y_target - X_restricted @ coeffs_r
    ssr_r = float(np.sum(residuals_r**2))

    coeffs_u, _, _, _ = np.linalg.lstsq(X_unrestricted, y_target, rcond=None)
    residuals_u = y_target - X_unrestricted @ coeffs_u
    ssr_u = float(np.sum(residuals_u**2))

    variance_ratio = ssr_r / ssr_u
    return float(np.log(variance_ratio)), ssr_r, ssr_u, variance_ratio
