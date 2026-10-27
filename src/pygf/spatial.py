"""CRS-aware environmental raster loading and point extraction."""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from rioxarray.exceptions import NoDataInBounds


@dataclass(eq=False)  # DataFrame.attrs comparisons must not compare GeoDataFrames elementwise.
class _SpatialContext:
    directory: str
    points: gpd.GeoDataFrame
    buffer: float


def _validate_points(points):
    if not isinstance(points, gpd.GeoDataFrame) or points.crs is None:
        raise ValueError("Points must be a GeoDataFrame with a CRS")
    if points.index.has_duplicates:
        raise ValueError("Point sample indices must be unique")
    if points.empty or points.geometry.isna().any() or points.geometry.is_empty.any():
        raise ValueError("Points must be nonempty and have valid geometries")
    if not points.geom_type.eq("Point").all():
        raise ValueError("Only Point geometries can be sampled")


def _buffered_bounds(points, raster, buffer):
    _validate_points(points)
    if not np.isfinite(buffer) or buffer < 0:
        raise ValueError("buffer must be nonnegative and finite")
    projected = points.to_crs(raster.rio.crs)
    left, bottom, right, top = projected.total_bounds
    if not np.isfinite([left, bottom, right, top]).all():
        raise ValueError("Points must have finite coordinates in the raster CRS")
    dx, dy = raster.rio.resolution()
    # A single point or a line of sites still needs a usable raster window.
    pad_x = (right - left) * buffer if right > left else abs(dx)
    pad_y = (top - bottom) * buffer if top > bottom else abs(dy)
    return left - pad_x, bottom - pad_y, right + pad_x, top + pad_y


def open_bioclim(directory, *, variables=None, bounds=None, points=None,
                 buffer=0.2, chunks="auto") -> xr.Dataset:
    """Open aligned WorldClim ``*bio_N.tif`` layers lazily as bio1 ... bio19.

    points automatically clips to their bounding box plus buffer (default 20%
    of the width/height on each side), after reprojection to the raster CRS.
    Collapsed axes get one cell of padding. Alternatively, explicit
    bounds=(left, bottom, right, top) is in the raster CRS and is applied before
    any computation. Rasters must have identical CRS, coordinates and shape;
    differing grids raise rather than silently resample. Numeric scale/offset
    metadata and nodata are decoded by rioxarray. Close the returned Dataset
    when finished, or use it as a context manager.
    """
    if points is not None and bounds is not None:
        raise ValueError("Supply points or bounds, not both")
    auto_bounds = points is not None
    files = {}
    for path in Path(directory).glob("*.tif"):
        match = re.search(r"bio[_-]?(\d+)\.tif$", path.name, re.IGNORECASE)
        if match:
            name = f"bio{int(match[1])}"
            if name in files:
                raise ValueError(f"Multiple rasters found for {name}")
            files[name] = path
    if not files:
        raise FileNotFoundError(f"No bioclim GeoTIFFs in {directory}")
    names = list(variables) if variables is not None else sorted(files, key=lambda n: int(n[3:]))
    if not names or len(names) != len(set(names)):
        raise ValueError("variables must be nonempty and unique")
    missing = set(names) - set(files)
    if missing:
        raise ValueError(f"Missing bioclim rasters: {sorted(missing)}")
    handles = []
    arrays = {}
    reference = None
    try:
        for name in names:
            raster = rioxarray.open_rasterio(files[name], masked=True, mask_and_scale=True,
                                            chunks=chunks)
            handles.append(raster)
            if raster.sizes.get("band") != 1 or raster.rio.crs is None:
                raise ValueError(f"{files[name]} must have one band and a CRS")
            layer = raster.squeeze("band", drop=True)
            if reference is not None:
                if (layer.rio.crs != reference.rio.crs
                        or not layer.x.equals(reference.x) or not layer.y.equals(reference.y)):
                    raise ValueError(f"{files[name]} does not share the other rasters' grid/CRS")
            reference = layer
            if auto_bounds and bounds is None:
                bounds = _buffered_bounds(points, layer, buffer)
            if bounds is not None:
                layer = layer.rio.clip_box(*bounds, auto_expand=auto_bounds)
            arrays[name] = layer.rename(name)
        dataset = xr.Dataset(arrays)
        dataset.set_close(lambda: [handle.close() for handle in handles])
        return dataset
    except Exception:
        for handle in handles:
            handle.close()
        raise


@contextmanager
def _environment_region(environment, points, *, variables=None, buffer=0.2,
                         stride=1, chunks=None):
    """Own directory-backed handles; caller-owned Datasets stay untouched."""
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if isinstance(environment, xr.Dataset):
        yield environment
    else:
        if points is None:
            raise ValueError("Sample points are required to open an environment directory")
        with open_bioclim(environment, variables=variables, points=points,
                           buffer=buffer, chunks=chunks or {"x": 512, "y": 512}) as raster:
            yield raster.isel(x=slice(None, None, stride), y=slice(None, None, stride))


def sample_environment(environment, points: gpd.GeoDataFrame, *, buffer=0.2,
                       variables=None, chunks=None, drop_missing=False) -> pd.DataFrame:
    """Extract nearest raster cells at GeoPandas points in any declared CRS.

    environment is a directory of bioclim GeoTIFFs. It is opened and clipped
    to the site bounding box plus 20% per side in the raster CRS, and closed
    after extraction. buffer changes that fraction; variables selects layers.
    An existing xarray Dataset is also accepted and used as supplied.

    Vectorized xarray selection computes only requested cells, even for global
    Dask-backed rasters. Outside-grid points and nodata produce NaN. Index and
    order are preserved; drop_missing=True explicitly removes incomplete rows.

    Directory-backed results remember their source and locations in DataFrame
    metadata, allowing a fitted GradientForest to plot maps with no path setup.
    """
    _validate_points(points)
    directory = not isinstance(environment, xr.Dataset)
    try:
        with _environment_region(environment, points, variables=variables, buffer=buffer,
                                  chunks=chunks) as raster:
            result = _sample_cells(raster, points)
    except NoDataInBounds:
        # A directory whose extent misses every site still returns NaN rows,
        # just like sampling an already-open Dataset outside its coverage.
        with open_bioclim(environment, variables=variables,
                           chunks=chunks or {"x": 512, "y": 512}) as raster:
            result = _sample_cells(raster, points)
    if drop_missing:
        result = result.dropna()
    if directory:
        result.attrs["_pygf_spatial_context"] = _SpatialContext(
            str(Path(environment).resolve()), points.loc[result.index].copy(), buffer)
    return result


def _sample_cells(environment, points):
    if environment.rio.crs is None:
        raise ValueError("Both points and environment must have a CRS")
    if not environment.data_vars:
        raise ValueError("Environment must have raster variables")
    for name, layer in environment.data_vars.items():
        if set(layer.dims) != {"x", "y"}:
            raise ValueError(f"{name} must have exactly x and y dimensions")
    projected = points.to_crs(environment.rio.crs)
    xs, ys = projected.geometry.x.to_numpy(), projected.geometry.y.to_numpy()
    left, bottom, right, top = environment.rio.bounds()
    inside = (np.isfinite(xs) & np.isfinite(ys) & (xs >= left) & (xs < right)
              & (ys > bottom) & (ys <= top))
    # Safe replacement permits .sel even if a CRS transform produced infinity.
    xs = np.where(inside, xs, float(environment.x[0]))
    ys = np.where(inside, ys, float(environment.y[0]))
    sampled = environment.sel(x=xr.DataArray(xs, dims="sample"),
                              y=xr.DataArray(ys, dims="sample"), method="nearest")
    result = pd.DataFrame({name: sampled[name].compute().values
                           for name in environment.data_vars}, index=points.index)
    result.loc[~inside, :] = np.nan
    return result
