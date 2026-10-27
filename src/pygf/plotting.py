"""PCA biplots of GF turnover, with matching spatial colours."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

from .spatial import _environment_region, _validate_points


@dataclass
class TurnoverBiplot:
    """Figure and numerical results returned by GradientForest.plot_biplot.

    scores/colors are indexed like the finite rows of X. vectors contains PCA
    covariance loadings, their length in PC1/2, and GF predictor importance.
    rgb is a georeferenced RGBA DataArray when a raster was supplied, else None.
    axes has 'biplot' and, when spatial information is available, 'map' entries.
    """

    figure: object
    axes: dict
    pca: PCA
    scores: pd.DataFrame
    vectors: pd.DataFrame
    colors: pd.DataFrame
    rgb: object
    color_limits: np.ndarray
    basis: str
    response: object


@dataclass
class TurnoverMap:
    """Single-gradient map and computed, georeferenced turnover layers."""

    figure: object
    ax: object
    turnover: xr.Dataset
    variable: str
    response: object


def plot_turnover(forest, variable=None, *, environment=None, points=None,
                  response=None, buffer=None, stride=4, cmap="viridis",
                  ax=None, figsize=(7, 4)):
    """Map one predictor's turnover and overlay sample locations.

    variable defaults to the most important predictor, or names a fitted
    predictor such as 'bio1'. environment is a bioclim directory; points is a
    GeoDataFrame. Both default to the source remembered when fitting on the
    result of sample_environment(directory, sites). buffer defaults to that
    extraction's buffer (20% per side unless changed). Directory maps use
    every fourth cell by default; stride=1 retains native resolution.

    Existing xarray Datasets are used as supplied, without cropping or striding.
    response optionally selects a retained SNP, as in transform_raster.
    Returns TurnoverMap with figure, ax, all computed turnover layers,
    selected variable and response. Source raster handles close after computing.
    """
    import matplotlib.pyplot as plt

    forest._curves(response)
    importance = forest.importance_ if response is None else forest.response_importances_.loc[response]
    variable = importance.idxmax() if variable is None else variable
    if variable not in forest.feature_names_in_:
        raise ValueError(f"Unknown predictor: {variable!r}")
    environment = environment if environment is not None else getattr(forest, "environment_", None)
    points = points if points is not None else getattr(forest, "sites_", None)
    if environment is None:
        raise ValueError("Supply environment, or fit on sample_environment(directory, sites)")
    if buffer is None:
        buffer = getattr(forest, "environment_buffer_", 0.2)
    if points is not None:
        _validate_points(points)
    with _environment_region(environment, points, variables=list(forest.feature_names_in_),
                              buffer=buffer, stride=stride) as region:
        turnover = forest.transform_raster(region, response=response).compute()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    turnover[variable].plot(ax=ax, cmap=cmap)
    if points is not None:
        points.to_crs(turnover.rio.crs).plot(ax=ax, color="black", markersize=10)
    ax.set_title(f"Turnover along {variable}")
    return TurnoverMap(ax.figure, ax, turnover, variable, response)


def _project(pca, values):
    scores = np.zeros((len(values), 3))
    scores[:, :pca.n_components_] = pca.transform(values)
    return scores


def _colors(scores, limits):
    span = limits[1] - limits[0]
    # An axis with no variation contributes a neutral channel, including
    # rank-one SNP models, which still have a useful one-dimensional biplot.
    active = span > np.finfo(float).eps * max(float(np.abs(limits).max()), 1e-30)
    rgb = np.full(scores.shape, 0.5)
    rgb[:, active] = np.clip((scores[:, active] - limits[0, active]) / span[active], 0, 1)
    return rgb


def _draw_vectors(ax, vectors, n_vectors, arrow_scale):
    selected = vectors.loc[vectors.length > 0].head(n_vectors)
    tips = selected[["PC1", "PC2"]].to_numpy() * arrow_scale
    if not len(tips):
        return
    ax.update_datalim(tips * 1.15)
    ax.autoscale_view()
    for tip in tips:
        ax.annotate("", xy=tip, xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.3})
    # Separate labels in display space without altering the vector endpoints.
    ax.figure.canvas.draw()
    screen = ax.transData.transform(tips)
    previous = -np.inf
    for i in np.argsort(screen[:, 1]):
        y = max(screen[i, 1], previous + 16)
        previous = y
        offset = (y - screen[i, 1]) * 72 / ax.figure.dpi
        side = 1 if tips[i, 0] >= 0 else -1
        ax.annotate(str(selected.index[i]), xy=tips[i], xytext=(7 * side, offset),
                    textcoords="offset points", ha="left" if side > 0 else "right",
                    va="center", fontsize=9,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.3"})


def plot_biplot(forest, X, points=None, *, environment=None, response=None,
                n_vectors=5, arrow_scale=2.0, label_sites=False,
                inset=False, figsize=None, buffer=None, stride=4):
    """Plot a GF turnover ordination and, optionally, its geographic projection.

    Parameters
    ----------
    forest : fitted GradientForest
    X : DataFrame or 2-D array
        Environmental predictors at samples or populations. DataFrame sample
        IDs must match the GeoDataFrame index; rows missing predictors are omitted.
    points : GeoDataFrame, optional
        Locations of rows of X. Reprojected to the raster CRS when needed.
    environment : path or xarray.Dataset, optional
        Directory of bioclim rasters. It is clipped to the sample bounding box
        plus buffer and downsampled by stride internally. An existing Dataset
        is used as supplied. PCA is fitted to all complete transformed raster
        cells, or to transformed X when no raster is given. The resulting
        region defines the PCA basis and RGB colour calibration.
    buffer : float, optional
        Fraction of the site width/height added on each side, default 0.2 (or
        the extraction's buffer when available). Sites are reprojected first.
    stride : int, default 4
        Use every stride-th cell for directory maps; 1 uses native resolution.
    response : response name, optional
        None averages all contributing SNPs. A name selects that SNP's own
        density-corrected turnover curves; fit with keep_response_curves=True.
        Excluded/unknown SNPs raise rather than displaying an aggregate result.
    n_vectors : int, default 5
        Label the largest covariance-loading vectors in the displayed PC1/2
        plane. Zero hides arrows. These are GF-transformed environmental
        variables: arrows point towards increasing environmental values.
    arrow_scale : float, default 2
        Common display multiplier for arrows, preserving relative lengths.
    label_sites : bool, default False
        Label sample/population IDs on both panels.
    inset : bool, default False
        Place the biplot inside the map instead of beside it.
    figsize : tuple, optional

    Returns
    -------
    TurnoverBiplot
        Figure, axes, PCA, sample scores/colours, environmental vectors and
        optional georeferenced raster colours. Save with result.figure.savefig.

    Notes
    -----
    PCA is centred but NOT standardized, preserving GF importance weights.
    Arrows are components.T * sqrt(explained_variance), i.e. covariance loadings,
    ranked by their PC1/2 length. They describe contributions to the displayed
    turnover pattern, not raw-environment regression slopes or causal effects.
    Overall GF importance is provided separately in result.vectors.importance.

    PC1/2 are plotted; PC1/2/3 map to red/green/blue, scaled using the basis'
    2nd/98th percentiles. Map cells and site points use the SAME calibration.
    At most 5,000 raster points are drawn in the biplot (all cells fit PCA).
    Each SNP gets its own PCA and colours; colours across separate fits are
    not a common genetic-distance scale. This depicts turnover coordinates,
    not allele-frequency predictions. For population-level interpretation,
    fit the forest to population allele frequencies and population environments.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Biplots require matplotlib; install pygf[plot]") from exc

    if not isinstance(n_vectors, int) or n_vectors < 0:
        raise ValueError("n_vectors must be a nonnegative integer")
    if not np.isfinite(arrow_scale) or arrow_scale <= 0:
        raise ValueError("arrow_scale must be positive and finite")
    if points is None:
        points = getattr(forest, "sites_", None)
    if buffer is None:
        buffer = getattr(forest, "environment_buffer_", 0.2)
    transformed = forest.transform(X, response=response).dropna()
    if transformed.empty:
        raise ValueError("X must have at least one complete row for plotting")
    names = list(forest.feature_names_in_)
    raster = None
    rgb = None
    if environment is not None:
        with _environment_region(environment, points, variables=names,
                                  buffer=buffer, stride=stride) as region:
            if region.rio.crs is None:
                raise ValueError("The environmental raster must have a CRS")
            raster = forest.transform_raster(region, response=response).compute()
        map_crs = raster.rio.crs
        raster = raster[names].to_array("predictor").transpose("y", "x", "predictor")
        valid = np.isfinite(raster.values).all(axis=-1)
        basis = raster.values[valid]
    else:
        basis = transformed.to_numpy()
    if len(basis) < 2 or not np.ptp(basis, axis=0).any():
        raise ValueError("A biplot requires at least two rows and nonzero turnover variation")
    pca = PCA(n_components=min(3, *basis.shape), svd_solver="full").fit(basis)
    basis_scores = _project(pca, basis)
    site_scores = _project(pca, transformed.to_numpy())
    limits = np.percentile(basis_scores, [2, 98], axis=0)
    site_colors = _colors(site_scores, limits)
    if raster is not None:
        rgba = np.zeros((*valid.shape, 4))
        rgba[valid, :3] = _colors(basis_scores, limits)
        rgba[valid, 3] = 1
        rgb = xr.DataArray(
            rgba, dims=("y", "x", "channel"),
            coords={"y": raster.y, "x": raster.x,
                    "channel": ["red", "green", "blue", "alpha"]},
            name="turnover_rgb",
        ).rio.write_crs(map_crs)
    loadings = np.zeros((len(names), 3))
    loadings[:, :pca.n_components_] = pca.components_.T * np.sqrt(pca.explained_variance_)
    vectors = pd.DataFrame(loadings, index=names, columns=["PC1", "PC2", "PC3"])
    vectors["length"] = np.linalg.norm(loadings[:, :2], axis=1)
    importance = forest.importance_ if response is None else forest.response_importances_.loc[response]
    vectors["importance"] = importance.reindex(names)
    vectors = vectors.sort_values("length", ascending=False)

    locations = None
    if points is not None:
        if not isinstance(points, gpd.GeoDataFrame) or points.crs is None:
            raise ValueError("points must be a GeoDataFrame with a CRS")
        if points.index.has_duplicates or not transformed.index.isin(points.index).all():
            raise ValueError("points must have unique sample IDs covering the rows of X")
        locations = points.loc[transformed.index]
        if (locations.geometry.isna().any() or locations.geometry.is_empty.any()
                or not locations.geom_type.eq("Point").all()):
            raise ValueError("Locations must have nonempty Point geometries")
        if environment is not None:
            locations = locations.to_crs(map_crs)
        if not np.isfinite(np.c_[locations.geometry.x, locations.geometry.y]).all():
            raise ValueError("Locations must have finite coordinates in the map CRS")
    spatial = environment is not None or locations is not None
    if spatial and inset:
        fig, map_ax = plt.subplots(figsize=figsize or (9, 6))
        biplot_ax = map_ax.inset_axes([0.03, 0.04, 0.40, 0.40])
        biplot_ax.patch.set_alpha(0.94)
    elif spatial:
        fig, (map_ax, biplot_ax) = plt.subplots(
            1, 2, figsize=figsize or (12, 5.5), layout="constrained")
    else:
        fig, biplot_ax = plt.subplots(figsize=figsize or (6, 5.5), layout="constrained")
        map_ax = None
    title = "All SNPs" if response is None else f"SNP: {response}"
    if map_ax is not None:
        if rgb is not None:
            rgb.plot.imshow(ax=map_ax, rgb="channel", add_colorbar=False)
        if locations is not None:
            map_ax.scatter(locations.geometry.x, locations.geometry.y, c=site_colors,
                           edgecolors="black", linewidths=0.7, s=45, zorder=3)
            if label_sites:
                for name, point in locations.geometry.items():
                    map_ax.annotate(str(name), (point.x, point.y), xytext=(4, 4),
                                    textcoords="offset points", fontsize=7)
        crs = map_crs if environment is not None else locations.crs
        if crs.is_geographic:
            latitude = np.clip(np.mean(map_ax.get_ylim()), -89, 89)
            map_ax.set_aspect(1 / np.cos(np.deg2rad(latitude)))
            map_ax.set(xlabel="Longitude", ylabel="Latitude")
        else:
            map_ax.set_aspect("equal")
            map_ax.set(xlabel="Easting", ylabel="Northing")
        map_ax.set_title(f"{title}: spatial turnover")
    if raster is not None:
        rows = np.linspace(0, len(basis_scores) - 1, min(5000, len(basis_scores)), dtype=int)
        biplot_ax.scatter(basis_scores[rows, 0], basis_scores[rows, 1],
                          c=_colors(basis_scores[rows], limits), s=3, alpha=0.5,
                          rasterized=True, linewidths=0)
    biplot_ax.scatter(site_scores[:, 0], site_scores[:, 1], c=site_colors,
                      edgecolors="black", linewidths=0.7, s=45, zorder=3)
    if label_sites:
        for name, score in zip(transformed.index, site_scores):
            biplot_ax.annotate(str(name), score[:2], xytext=(4, 4),
                               textcoords="offset points", fontsize=7)
    ratios = np.zeros(3)
    ratios[:pca.n_components_] = pca.explained_variance_ratio_
    biplot_ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    biplot_ax.axvline(0, color="0.8", lw=0.5, zorder=0)
    biplot_ax.set(xlabel=f"PC1 ({ratios[0]:.1%})", ylabel=f"PC2 ({ratios[1]:.1%})",
                   title=f"{title}: turnover biplot")
    biplot_ax.set_aspect("equal", adjustable="datalim")
    _draw_vectors(biplot_ax, vectors, n_vectors, arrow_scale)
    axes = {"biplot": biplot_ax}
    if map_ax is not None:
        axes["map"] = map_ax
    return TurnoverBiplot(
        figure=fig, axes=axes, pca=pca,
        scores=pd.DataFrame(site_scores, index=transformed.index, columns=["PC1", "PC2", "PC3"]),
        vectors=vectors,
        colors=pd.DataFrame(site_colors, index=transformed.index, columns=["red", "green", "blue"]),
        rgb=rgb, color_limits=limits, basis="raster" if raster is not None else "sites",
        response=response,
    )
