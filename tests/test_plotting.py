import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pygf import GradientForest, open_bioclim, sample_environment


@pytest.fixture
def fitted():
    rng = np.random.default_rng(10)
    X = pd.DataFrame(rng.uniform(-2, 2, (100, 3)), columns=["bio1", "bio2", "bio3"])
    y = pd.DataFrame({"snp_a": X.bio1 > 0, "snp_b": X.bio2 > 0,
                      "constant": np.ones(len(X))})
    gf = GradientForest(n_estimators=60, max_features=1.0, random_state=4,
                        keep_response_curves=True).fit(X, y)
    return gf, X


def test_single_response_curves_and_aggregate_agree(fitted):
    gf, X = fitted
    retained = gf.response_scores_.query("retained").index
    assert set(retained) == {"snp_a", "snp_b"}
    individual = [gf.transform(X, response=name) for name in retained]
    np.testing.assert_allclose(gf.transform(X), np.mean(individual, axis=0))
    assert not np.allclose(individual[0], individual[1])
    for name in retained:
        endpoints = gf.response_turnover_[name][:, -1]
        np.testing.assert_allclose(endpoints, gf.response_importances_.loc[name])
    assert gf.forests_ == {}
    with pytest.raises(ValueError, match="does not contribute"):
        gf.transform(X, response="constant")
    with pytest.raises(ValueError, match="Unknown response"):
        gf.transform(X, response="absent")
    gf.response_turnover_ = {}
    with pytest.raises(ValueError, match="keep_response_curves=True"):
        gf.transform(X, response="snp_a")


def test_biplot_scores_vectors_and_single_snp(fitted):
    gf, X = fitted
    result = gf.plot_biplot(X, n_vectors=2)
    np.testing.assert_allclose(result.scores.iloc[:, :3], result.pca.transform(gf.transform(X).to_numpy()))
    expected = result.pca.components_.T * np.sqrt(result.pca.explained_variance_)
    np.testing.assert_allclose(result.vectors.loc[X.columns, ["PC1", "PC2", "PC3"]], expected)
    assert result.vectors.importance.to_dict() == gf.importance_.to_dict()
    assert result.basis == "sites"
    assert result.scores.index.equals(X.index)
    assert len(result.axes["biplot"].texts) == 4  # two arrows and two labels
    snp = gf.plot_biplot(X, response="snp_a")
    assert snp.response == "snp_a"
    assert snp.vectors.importance.idxmax() == "bio1"
    np.testing.assert_allclose(snp.pca.mean_, gf.transform(X, response="snp_a").mean())
    assert not np.allclose(snp.pca.mean_, result.pca.mean_)
    plt.close("all")


def test_spatial_colors_crs_mask_and_site_alignment(fitted):
    gf, X = fitted
    # Each pixel corresponds to one of the first twenty sample environments.
    env = xr.Dataset({name: (("y", "x"), X[name].iloc[:20].values.reshape(4, 5))
                      for name in X}, coords={"y": [3.5, 2.5, 1.5, .5],
                                             "x": [.5, 1.5, 2.5, 3.5, 4.5]}).rio.write_crs(4326)
    env["bio1"][0, 0] = np.nan
    sites = gpd.GeoDataFrame(index=X.index[:20],
        geometry=gpd.points_from_xy(np.tile(env.x.values, 4), np.repeat(env.y.values, 5)),
        crs=4326).iloc[::-1].to_crs(3857)
    for response in (None, "snp_a"):
        result = gf.plot_biplot(X.iloc[:20], sites, environment=env, response=response,
                               n_vectors=2, inset=response is not None)
        assert result.basis == "raster"
        assert result.rgb.rio.crs.to_epsg() == 4326
        assert result.rgb.sel(channel="alpha").values[0, 0] == 0
        assert result.rgb.sel(channel="alpha").values.sum() == 19
        # All complete cells share their corresponding sample's exact colour.
        np.testing.assert_allclose(result.rgb.values.reshape(-1, 4)[1:, :3],
                                   result.colors.iloc[1:])
        artist = result.axes["map"].collections[0]
        np.testing.assert_allclose(artist.get_offsets(),
            np.c_[np.tile(env.x.values, 4), np.repeat(env.y.values, 5)])
        grid = gf.transform_raster(env, response=response).compute()
        for j, name in enumerate(X):
            expected = gf.transform(X.iloc[:20], response=response)[name].to_numpy(copy=True).reshape(4, 5)
            expected[0, 0] = np.nan
            np.testing.assert_allclose(grid[name], expected, equal_nan=True)
    plt.close("all")


def test_rank_one_and_missing_points():
    X = pd.DataFrame({"bio1": np.linspace(-2, 2, 60)})
    gf = GradientForest(n_estimators=40, random_state=6).fit(X, X.bio1 > 0)
    result = gf.plot_biplot(X)
    assert np.isfinite(result.colors).all().all()
    assert (result.scores.PC2 == 0).all()
    np.testing.assert_allclose(result.colors[["green", "blue"]], .5)
    sites = gpd.GeoDataFrame(index=["unknown"], geometry=gpd.points_from_xy([0], [0]), crs=4326)
    with pytest.raises(ValueError, match="sample IDs"):
        gf.plot_biplot(X, sites)
    with pytest.raises(ValueError, match="nonzero turnover"):
        gf.plot_biplot(pd.DataFrame({"bio1": np.ones(10)}))
    plt.close("all")


def test_directory_maps_use_remembered_sites_and_buffer(tmp_path):
    y_coords = np.arange(19.5, 0, -1)
    x_coords = np.arange(.5, 20)
    for name, values in (("bio1", np.tile(x_coords, (20, 1))),
                         ("bio2", np.tile(y_coords[:, None], (1, 20)))):
        xr.DataArray(values, dims=("y", "x"), coords={"x": x_coords, "y": y_coords}).rio.write_crs(
            4326).rio.to_raster(tmp_path / f"{name}.tif")
    points = gpd.GeoDataFrame(index=[f"s{i}" for i in range(30)],
        geometry=gpd.points_from_xy(np.linspace(5, 15, 30), np.tile([6., 14.], 15)), crs=4326)
    X = sample_environment(tmp_path, points)
    gf = GradientForest(n_estimators=50, max_features=1.0, random_state=8,
                         keep_response_curves=True).fit(X, X.bio1 > 10)
    assert gf.sites_.index.equals(X.index)
    assert gf.environment_ == str(tmp_path)
    for variable, response in ((None, None), ("bio2", "bio1")):
        result = gf.plot_turnover(variable, response=response, stride=1)
        assert result.variable == (gf.importance_.idxmax() if variable is None else variable)
        assert result.turnover.rio.crs.to_epsg() == 4326
        with open_bioclim(tmp_path, points=points) as expected:
            np.testing.assert_array_equal(result.turnover.x, expected.x)
            np.testing.assert_array_equal(result.turnover.y, expected.y)
            reference = gf.transform_raster(expected, response=response).compute()
            xr.testing.assert_allclose(result.turnover, reference)
    # The directory entry point produces the same PCA as the equivalent region.
    biplot = gf.plot_biplot(X, environment=tmp_path, stride=1)
    with open_bioclim(tmp_path, points=points) as region:
        explicit = gf.plot_biplot(X, points, environment=region)
    np.testing.assert_allclose(biplot.scores, explicit.scores)
    np.testing.assert_allclose(biplot.rgb, explicit.rgb)
    default_map = gf.plot_turnover()
    assert default_map.turnover.sizes["x"] < result.turnover.sizes["x"]
    with pytest.raises(ValueError, match="Unknown predictor"):
        gf.plot_turnover("bio20")
    plain = GradientForest(n_estimators=30, random_state=8).fit(
        pd.DataFrame(X.to_numpy(), columns=X.columns, index=X.index), X.bio1 > 10)
    with pytest.raises(ValueError, match="Supply environment"):
        plain.plot_turnover()
    # Refitting on plain data clears old geospatial context.
    gf.fit(pd.DataFrame(X.to_numpy(), columns=X.columns, index=X.index), X.bio1 > 10)
    assert gf.environment_ is None and gf.sites_ is None
    plt.close("all")
