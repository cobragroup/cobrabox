# src/cobrabox/features/svd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import xarray as xr

from cobrabox.base_feature import BaseFeature
from cobrabox.data import Data

_SVDOutputMode = Literal["V", "U"]


@dataclass
class SVD(BaseFeature[Data]):
    """Compute truncated SVD over one dimension.

    The input `data.data` may be N-D (fMRI, EEG, etc.). We treat `dim` as the
    sample axis and stack all other dims into a single feature axis internally.

    Args:
        dim: Sample dimension (e.g. "time", "trial").
        n_components: Number of components to return.
        center:
            If True, subtract each feature's baseline (mean over `dim`) before SVD.
            Intuition: removes offsets/mean image so components describe
            fluctuations rather than the average level.
        zscore:
            If True, after centering, also normalize each feature by its
            variability (divide by std over `dim`).
            Intuition: prevents a few "loud"/high-variance features from
            dominating; components reflect shared patterns rather than amplitude.
            Implies centering.
        mask:
            Optional boolean selector over non-`dim` dims.
            True keeps a feature (voxel/channel/etc.), False removes it.
            Use mask to restrict SVD to a subset of features, for instance to
            exclude irrelevant/noisy features (e.g. non-brain voxels, bad EEG
            channels), speed up computation, and focus on a region/sensor set.
        return_unstacked_V:
            If True, return V reshaped back to the original non-`dim` dims.
        output:
            Which primary output to return as xr.DataArray:
            - "V" (default): feature patterns/maps, shape (component, ...)
            - "U": sample scores/timecourses, shape (dim, component)

    Returns:
        xr.DataArray:
            If output="V": (component, ...) feature patterns/maps
            If output="U": (dim, component) sample scores/timecourses

        Extra outputs are stored in `result.attrs["svd"]`:
            S, Vh, mean, std, and whichever of U/V is not returned.

    Example:
        # fMRI (time,x,y,z) + mask -> returns V(component,x,y,z)
        >>> result = cb.feature.SVD(dim="time", n_components=10, mask=brain_mask).apply(data)
        # EEG (time,channel) -> returns V(component,channel)
        >>> result = cb.feature.SVD(dim="time", n_components=10).apply(eeg_data)
        # EEG time-frequency (time,channel,freq) + zscore -> V(component,channel,freq)
        >>> result = cb.feature.SVD(dim="time", n_components=10, zscore=True).apply(tfr_data)
        # Trial-wise (trial,time,channel) -> V(component,time,channel), U stored in attrs
        >>> result = cb.feature.SVD(dim="trial", n_components=10).apply(epochs_data)
        # No unstacking -> main output is Vh(component,features)
        >>> out = cb.feature.SVD(dim="time", n_components=10, return_unstacked_V=False).apply(data)
        >>> svd = out.data.attrs["svd"]
        >>>         U = svd["U"]  # sample scores / timecourses (if output="V")
        >>> S = svd["S"]  # singular values

    Raises:
        ValueError: If ``dim`` is not found in the input data dimensions.
        ValueError: If ``n_components`` is not positive.
        ValueError: If ``output`` is not one of ``{"V", "U"}``.

    References:
        Golub, G. H., & Kahan, W. (1965). Calculating the singular values and pseudo-inverse
        of a matrix. *Journal of the Society for Industrial and Applied Mathematics: Series B,
        Numerical Analysis*, 2(2), 205-224.
    """

    dim: str
    n_components: int = 10
    center: bool = True
    zscore: bool = False
    mask: xr.DataArray | None = None
    return_unstacked_V: bool = True
    output: _SVDOutputMode = "V"

    def __call__(self, data: Data) -> xr.DataArray | Data:
        da = data.data

        if self.dim not in da.dims:
            raise ValueError(f"dim '{self.dim}' not found in {da.dims}")
        if self.n_components <= 0:
            raise ValueError("n_components must be > 0")
        if self.output not in get_args(_SVDOutputMode):
            raise ValueError(
                f"output must be one of {get_args(_SVDOutputMode)}, got {self.output!r}"
            )

        # Stack all non-dim dims into "features" -> (dim, features)
        other_dims = [d for d in da.dims if d != self.dim]
        stacked = da.stack(features=other_dims) if other_dims else da.expand_dims(features=[0])
        X = stacked.transpose(self.dim, "features")

        # Optional mask (robust): broadcast to non-dim template, then stack
        if self.mask is not None:
            m = self.mask
            if self.dim in m.dims:
                m = m.any(self.dim)

            if other_dims:
                template = da.isel({self.dim: 0}, drop=True)
                m = m.broadcast_like(template).stack(features=other_dims)
            else:
                m = m.expand_dims(features=[0])

            keep = np.asarray(m.data).astype(bool)
            X = X.isel(features=keep)

        X_np = np.asarray(X.data)

        # Center / z-score over dim (per feature)
        mean_da: xr.DataArray | None = None
        std_da: xr.DataArray | None = None

        if self.zscore:
            mean_np = X_np.mean(axis=0, keepdims=True)
            std_np = X_np.std(axis=0, ddof=0, keepdims=True)
            std_np = np.where(std_np == 0, 1.0, std_np)
            X_np = (X_np - mean_np) / std_np

            mean_da = xr.DataArray(
                mean_np.squeeze(0), dims=("features",), coords={"features": X.features}, name="mean"
            )
            std_da = xr.DataArray(
                std_np.squeeze(0), dims=("features",), coords={"features": X.features}, name="std"
            )
        elif self.center:
            mean_np = X_np.mean(axis=0, keepdims=True)
            X_np = X_np - mean_np
            mean_da = xr.DataArray(
                mean_np.squeeze(0), dims=("features",), coords={"features": X.features}, name="mean"
            )

        # SVD (NumPy full then slice)
        U, S, Vh = np.linalg.svd(X_np, full_matrices=False)
        k = min(self.n_components, S.shape[0])
        comp = np.arange(k)

        U_da = xr.DataArray(
            U[:, :k],
            dims=(self.dim, "component"),
            coords={self.dim: X.coords[self.dim], "component": comp},
            name="U",
        )
        S_da = xr.DataArray(S[:k], dims=("component",), coords={"component": comp}, name="S")
        Vh_da = xr.DataArray(
            Vh[:k, :],
            dims=("component", "features"),
            coords={"component": comp, "features": X.features},
            name="Vh",
        )

        # Build V (either unstacked or flattened)
        if self.return_unstacked_V and other_dims:
            V_da = Vh_da.unstack("features")
            V_da.name = "V"
            if mean_da is not None:
                mean_da = mean_da.unstack("features")
            if std_da is not None:
                std_da = std_da.unstack("features")
        else:
            V_da = Vh_da.copy()
            V_da.name = "Vh"

        # Choose primary output
        primary = V_da if self.output == "V" else U_da

        # Store extra results in attrs
        primary.attrs["svd"] = {
            "U": U_da if self.output != "U" else None,
            "S": S_da,
            "Vh": Vh_da,
            "mean": mean_da,
            "std": std_da,
            "svd_dim": self.dim,
            "n_components": int(k),
            "center": bool(self.center),
            "zscore": bool(self.zscore),
            "masked": self.mask is not None,
        }

        return primary
