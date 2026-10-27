import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import rioxarray  # noqa: F401

from pygf import GradientForest, open_bioclim, sample_environment


def raster(path, values, crs="EPSG:4326", x=None):
    data = xr.DataArray(np.asarray(values, dtype=np.float32), dims=("y", "x"),
                        coords={"y": [1.5, .5], "x": [.5, 1.5] if x is None else x})
    data.rio.write_crs(crs).rio.write_nodata(-9999).rio.to_raster(path)


def test_point_extraction_nodata_crs_bounds_and_lazy_projection(tmp_path):
    raster(tmp_path / "wc2.1_bio_1.tif", [[1, 2], [3, -9999]])
    raster(tmp_path / "wc2.1_bio_2.tif", [[4, 5], [6, 7]])
    points = gpd.GeoDataFrame(index=["west", "masked", "outside"],
        geometry=gpd.points_from_xy([.5, 1.5, 20], [1.5, .5, 20]), crs=4326)
    with open_bioclim(tmp_path, chunks={"x": 1, "y": 1}) as env:
        assert env.bio1.chunks is not None
        extracted = sample_environment(env, points.to_crs(3857))
        assert extracted.loc["west", "bio1"] == 1
        assert np.isnan(extracted.loc["masked", "bio1"])
        assert extracted.loc["outside"].isna().all()
        assert len(sample_environment(env, points, drop_missing=True)) == 1
        rng = np.random.default_rng(4)
        X = pd.DataFrame(rng.uniform(0, 8, (80, 2)), columns=["bio1", "bio2"])
        gf = GradientForest(n_estimators=40, random_state=2).fit(X, X.bio1 ** 2)
        projected = gf.transform_raster(env)
        assert projected.bio1.chunks is not None
        assert projected.rio.crs == env.rio.crs
        assert projected.sizes == env.sizes
        output = projected.compute()
        assert np.isnan(output.bio2.sel(x=1.5, y=.5))
        expected = gf.transform(pd.DataFrame({"bio1": [1.], "bio2": [4.]}))
        assert output.bio1.sel(x=.5, y=1.5) == pytest.approx(expected.bio1[0])
        assert output.bio1.attrs["units"] == "R2 importance"


def test_grid_mismatch_rejected(tmp_path):
    raster(tmp_path / "bio_1.tif", [[1, 2], [3, 4]])
    raster(tmp_path / "bio_2.tif", [[1, 2], [3, 4]], x=[1.5, 2.5])
    with pytest.raises(ValueError, match="grid/CRS"):
        open_bioclim(tmp_path)


def test_directory_sampling_buffer_order_crs_and_single_point(tmp_path):
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    layer = xr.DataArray(values, dims=("y", "x"),
        coords={"y": np.arange(9.5, 0, -1), "x": np.arange(.5, 10)})
    layer.rio.write_crs(4326).rio.to_raster(tmp_path / "bio_1.tif")
    sites = gpd.GeoDataFrame(index=["east", "west"],
        geometry=gpd.points_from_xy([7.1, 3.1], [7.1, 3.1]), crs=4326)
    with open_bioclim(tmp_path, points=sites.to_crs(3857)) as region:
        # Site width/height 4 + 0.8 per side selects cell centres 2.5..7.5.
        np.testing.assert_array_equal(region.x, np.arange(2.5, 8))
        np.testing.assert_array_equal(region.y, np.arange(7.5, 2, -1))
        expected = sample_environment(region, sites)
    actual = sample_environment(tmp_path, sites.to_crs(3857))
    np.testing.assert_allclose(actual, expected)
    assert actual.index.tolist() == ["east", "west"]
    outside = gpd.GeoDataFrame(index=["outside"],
        geometry=gpd.points_from_xy([20], [20]), crs=4326)
    assert sample_environment(tmp_path, outside).isna().all().all()
    assert sample_environment(tmp_path, outside, drop_missing=True).empty
    with open_bioclim(tmp_path, points=sites.iloc[:1]) as region:
        assert region.sizes["x"] >= 2 and region.sizes["y"] >= 2
    with pytest.raises(ValueError, match="buffer"):
        sample_environment(tmp_path, sites, buffer=-.1)
    with pytest.raises(ValueError, match="points or bounds"):
        open_bioclim(tmp_path, points=sites, bounds=(0, 0, 10, 10))
