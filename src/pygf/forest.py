"""Response-specific random forests and density-corrected turnover curves.

Independent implementation of the method of Ellis, Smith & Pitcher (2012),
doi:10.1890/11-0252.1. See README for numerical differences from R gradientForest.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted


def _conditioning_groups(tree, X, feature, correlations, max_level, threshold):
    """Grid of at most max_level shallow, correlated splits in this tree."""
    groups = np.zeros(len(X), dtype=np.int64)
    used = 0
    queue = [0]
    while queue and used < max_level:
        node = queue.pop(0)
        other = tree.feature[node]
        if other < 0:
            continue
        if other != feature and abs(correlations[feature, other]) > threshold:
            groups = 2 * groups + (X[:, other] > tree.threshold[node])
            used += 1
        queue.extend([tree.children_left[node], tree.children_right[node]])
    return groups


def _oob_statistics(forest, X, y, rng, n_repeats, max_level, corr_threshold):
    """Score only observations with OOB votes; permute within each tree's OOB set."""
    n, p = X.shape
    predictions = np.zeros(n)
    votes = np.zeros(n, dtype=int)
    deltas = np.zeros(p)
    counts = np.zeros(p, dtype=int)
    if max_level:
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = np.nan_to_num(np.corrcoef(X, rowvar=False))
        correlations = np.atleast_2d(correlations)
    for estimator, inbag in zip(forest.estimators_, forest.estimators_samples_):
        oob = np.flatnonzero(np.bincount(inbag, minlength=n) == 0)
        if not len(oob):
            continue
        inputs = X[oob]
        baseline = estimator.predict(inputs, check_input=False)
        predictions[oob] += baseline
        votes[oob] += 1
        if len(oob) < 2:
            continue
        baseline_mse = np.mean((y[oob] - baseline) ** 2)
        for j in np.unique(estimator.tree_.feature):
            if j < 0:
                continue
            groups = (
                _conditioning_groups(estimator.tree_, inputs, j, correlations,
                                     max_level, corr_threshold)
                if max_level else np.zeros(len(inputs), dtype=int)
            )
            partitions = [np.flatnonzero(groups == group) for group in np.unique(groups)]
            permuted = inputs.copy()
            for _ in range(n_repeats):
                for rows in partitions:
                    permuted[rows, j] = inputs[rng.permutation(rows), j]
                pred = estimator.predict(permuted, check_input=False)
                deltas[j] += np.mean((y[oob] - pred) ** 2) - baseline_mse
            counts[j] += n_repeats
        # An unused variable has exactly zero permutation delta in this tree.
        unused = np.ones(p, dtype=bool)
        used_features = estimator.tree_.feature[estimator.tree_.feature >= 0]
        unused[used_features] = False
        counts[unused] += n_repeats
    valid = votes > 0
    result = np.full(n, np.nan)
    result[valid] = predictions[valid] / votes[valid]
    denominator = np.sum((y[valid] - y[valid].mean()) ** 2)
    r2 = 1 - np.sum((y[valid] - result[valid]) ** 2) / denominator if denominator > 0 else np.nan
    return r2, np.divide(deltas, counts, out=np.zeros(p), where=counts > 0), result


def _split_histograms(forest, edges):
    histograms = [np.zeros(len(e) - 1) for e in edges]
    for estimator in forest.estimators_:
        tree = estimator.tree_
        nodes = np.flatnonzero(tree.feature >= 0)
        left, right = tree.children_left[nodes], tree.children_right[nodes]
        weights = tree.weighted_n_node_samples
        gain = (weights[nodes] * tree.impurity[nodes]
                - weights[left] * tree.impurity[left]
                - weights[right] * tree.impurity[right])
        for j in np.unique(tree.feature[nodes]):
            selected = tree.feature[nodes] == j
            histograms[j] += np.histogram(
                tree.threshold[nodes[selected]], bins=edges[j],
                weights=np.maximum(gain[selected], 0),
            )[0]
    return histograms


def _turnover_mass(splits, observations, edges, density_correction):
    """Smooth weighted split density; divide by smoothed sampling density."""
    width = edges[1] - edges[0]
    # Silverman-style bandwidth, shared by both histograms, at least one bin.
    sd = np.std(observations)
    iqr = np.subtract(*np.percentile(observations, [75, 25])) / 1.349
    scale = min(sd, iqr) if iqr > 0 else sd
    sigma = max(0.9 * scale * len(observations) ** (-0.2) / width, 1.0)
    mass = gaussian_filter1d(splits, sigma, mode="constant")
    if density_correction:
        density = gaussian_filter1d(
            np.histogram(observations, bins=edges)[0].astype(float),
            sigma, mode="constant",
        )
        mass /= np.maximum(density, density.max() * 1e-6)
    return mass / mass.sum() if mass.sum() > 0 else mass


class GradientForest(TransformerMixin, BaseEstimator):
    """Fit a forest for each response and learn environmental turnover.

    X and y have samples in rows. DataFrame indices must match exactly; use
    explicit ``.loc`` alignment rather than silently mixing individuals.
    Responses may contain NaNs; each forest uses only observed responses.
    Predictors must be finite, numeric, and contain no missing values.

    Only responses with OOB R² > max(0, min_oob_r2) and positive permutation
    importance contribute. Per-response positive permutation importances are
    normalized to sum to R². Smoothed split improvements are divided by the
    predictor sampling density and scaled to these importances. Curves are
    averaged over contributing responses. Their endpoints sum to mean retained
    R². Constant predictors have zero turnover.

    ``transform`` (and ``predict``) returns cumulative turnover coordinates,
    NOT fitted response values. Values outside training ranges are clamped.

    max_level=0 gives marginal OOB permutation importance. Positive max_level
    conditions permutations on a grid of at most max_level shallow splits in
    each tree on other predictors with |r| > corr_threshold. This bounded grid
    is an approximation to extendedForest's conditional partitions.

    n_bins controls compact split storage. keep_forests=False avoids retaining
    all trees for genomic data. n_jobs parallelizes trees within each response.
    keep_response_curves=True retains compact curves for individual-response
    transformation and biplots, without retaining the underlying forests.
    """

    def __init__(self, n_estimators=200, *, max_features=1 / 3,
                 min_samples_leaf=2, max_depth=None, n_bins=256,
                 min_oob_r2=0.0, n_permutations=1, density_correction=True,
                 max_level=0, corr_threshold=0.5, random_state=None,
                 n_jobs=None, keep_forests=False, keep_response_curves=False):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.min_oob_r2 = min_oob_r2
        self.n_permutations = n_permutations
        self.density_correction = density_correction
        self.max_level = max_level
        self.corr_threshold = corr_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.keep_forests = keep_forests
        self.keep_response_curves = keep_response_curves

    def fit(self, X, y):
        """Learn turnover curves and retain OOB diagnostics for every response."""
        if not isinstance(self.n_bins, int) or self.n_bins < 8:
            raise ValueError("n_bins must be an integer >= 8")
        if not isinstance(self.n_permutations, int) or self.n_permutations < 1:
            raise ValueError("n_permutations must be a positive integer")
        if not isinstance(self.max_level, int) or not 0 <= self.max_level <= 16:
            raise ValueError("max_level must be an integer between 0 and 16")
        if not 0 <= self.corr_threshold <= 1:
            raise ValueError("corr_threshold must be between 0 and 1")
        if not np.isfinite(self.min_oob_r2) or not 0 <= self.min_oob_r2 < 1:
            raise ValueError("min_oob_r2 must be in [0, 1)")
        frame = self._frame(X)
        if isinstance(y, pd.Series):
            y = y.to_frame()
        if isinstance(y, pd.DataFrame):
            if not frame.index.equals(y.index):
                raise ValueError("X and y sample indices/order must match exactly")
            responses = y.copy()
        else:
            values = np.asarray(y, dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2 or len(values) != len(frame):
                raise ValueError("y must have the same number of rows as X")
            responses = pd.DataFrame(values, index=frame.index)
        if responses.columns.has_duplicates or responses.shape[1] == 0:
            raise ValueError("Response names must be unique and nonempty")
        if len(frame) < 5:
            raise ValueError("At least five samples are required")
        values = frame.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite numeric predictors")
        targets = responses.to_numpy(dtype=float)
        if np.isinf(targets).any():
            raise ValueError("Responses may contain NaN but not infinity")
        context = frame.attrs.get("_pygf_spatial_context")
        self.environment_ = context.directory if context is not None else None
        self.sites_ = context.points.loc[frame.index].copy() if context is not None else None
        self.environment_buffer_ = context.buffer if context is not None else 0.2
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.response_names_in_ = np.asarray(responses.columns, dtype=object)
        self.feature_ranges_ = pd.DataFrame(
            {"min": frame.min(), "max": frame.max()})
        edges = []
        for low, high in self.feature_ranges_.itertuples(index=False, name=None):
            edges.append(np.linspace(low, high if high > low else low + 1, self.n_bins + 1))
        self.forests_ = {}
        self.response_turnover_ = {}
        raw = np.zeros((responses.shape[1], self.n_features_in_))
        allocated = np.zeros_like(raw)
        cumulative = [np.zeros(self.n_bins + 1) for _ in edges]
        diagnostics = []
        oob = np.full(targets.shape, np.nan)
        rng = np.random.default_rng(self.random_state)
        retained = 0
        for k, name in enumerate(responses.columns):
            observed = np.isfinite(targets[:, k])
            response = targets[observed, k]
            row = {"response": name, "n_observed": len(response), "oob_r2": np.nan,
                   "oob_coverage": 0.0, "retained": False, "reason": "too_few_observations"}
            if len(response) < 5:
                diagnostics.append(row)
                continue
            if np.ptp(response) == 0:
                row["reason"] = "constant"
                diagnostics.append(row)
                continue
            forest = RandomForestRegressor(
                n_estimators=self.n_estimators, max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth,
                bootstrap=True, random_state=int(rng.integers(0, 2**31 - 1)),
                n_jobs=self.n_jobs,
            ).fit(values[observed], response)
            score, importance, prediction = _oob_statistics(
                forest, values[observed], response, rng, self.n_permutations,
                self.max_level, self.corr_threshold,
            )
            oob[observed, k] = prediction
            raw[k] = importance
            row.update(oob_r2=score, oob_coverage=float(np.isfinite(prediction).mean()),
                       reason="nonpositive_oob_skill")
            if self.keep_forests:
                self.forests_[name] = forest
            if not np.isfinite(score) or score <= self.min_oob_r2:
                diagnostics.append(row)
                continue
            positive = np.maximum(importance, 0)
            split_mass = _split_histograms(forest, edges)
            positive[[mass.sum() <= 0 for mass in split_mass]] = 0
            if positive.sum() == 0:
                row["reason"] = "no_positive_importance"
                diagnostics.append(row)
                continue
            allocated[k] = positive / positive.sum() * score
            response_curves = []
            for j, weight in enumerate(allocated[k]):
                curve = np.zeros(self.n_bins + 1)
                if weight > 0:
                    mass = _turnover_mass(split_mass[j], values[observed, j], edges[j],
                                          self.density_correction)
                    curve = np.r_[0, mass.cumsum()] * weight
                    cumulative[j] += curve
                if self.keep_response_curves:
                    response_curves.append(curve)
            if self.keep_response_curves:
                self.response_turnover_[name] = np.asarray(response_curves)
            retained += 1
            row.update(retained=True, reason="retained")
            diagnostics.append(row)
        self.response_scores_ = pd.DataFrame(diagnostics).set_index("response")
        self.oob_predictions_ = pd.DataFrame(oob, index=frame.index, columns=responses.columns)
        self.permutation_importances_ = pd.DataFrame(raw, index=responses.columns, columns=frame.columns)
        self.response_importances_ = pd.DataFrame(allocated, index=responses.columns, columns=frame.columns)
        self.n_responses_retained_ = retained
        self.feature_importances_ = allocated.sum(axis=0) / max(retained, 1)
        self.importance_ = pd.Series(self.feature_importances_, index=frame.columns,
                                     name="importance").sort_values(ascending=False)
        self.turnover_ = {}
        for j, name in enumerate(frame.columns):
            curve = cumulative[j] / max(retained, 1)
            self.turnover_[name] = pd.DataFrame({"value": edges[j], "cumulative_importance": curve})
        if retained == 0:
            warnings.warn("No responses have positive OOB skill and importance; turnover is zero.",
                          RuntimeWarning, stacklevel=2)
        return self

    @staticmethod
    def _frame(X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("X must be a two-dimensional table")
            frame = pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])])
        if frame.shape[1] == 0 or frame.columns.has_duplicates or frame.index.has_duplicates:
            raise ValueError("X must have predictors and unique column and sample names")
        return frame

    def _curves(self, response=None):
        check_is_fitted(self, "turnover_")
        if response is None:
            return self.turnover_
        if response not in self.response_scores_.index:
            raise ValueError(f"Unknown response: {response!r}")
        if not self.response_scores_.loc[response, "retained"]:
            reason = self.response_scores_.loc[response, "reason"]
            raise ValueError(f"Response {response!r} does not contribute to turnover ({reason})")
        if response not in getattr(self, "response_turnover_", {}):
            raise ValueError("Refit with keep_response_curves=True to use a single response")
        return {
            name: pd.DataFrame({"value": self.turnover_[name].value,
                                "cumulative_importance": self.response_turnover_[response][j]})
            for j, name in enumerate(self.feature_names_in_)
        }

    def transform(self, X, *, response=None):
        """Return turnover coordinates, preserving index and NaNs.

        response=None averages all retained responses. A response name selects
        that response's own curves (fit with keep_response_curves=True).
        """
        curves = self._curves(response)
        frame = self._frame(X)
        if not isinstance(X, pd.DataFrame):
            if frame.shape[1] != self.n_features_in_:
                raise ValueError("X has the wrong number of predictors")
            frame.columns = self.feature_names_in_
        missing = set(self.feature_names_in_) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing predictors: {sorted(missing)}")
        result = pd.DataFrame(index=frame.index)
        for name in self.feature_names_in_:
            values = frame[name].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError("Predictors cannot contain infinity")
            curve = curves[name]
            result[name] = np.interp(values, curve.value, curve.cumulative_importance)
        return result

    def predict(self, X, *, response=None):
        """Alias for transform: predicts turnover coordinates, not responses."""
        return self.transform(X, response=response)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()

    def transform_raster(self, environment: xr.Dataset, *, response=None) -> xr.Dataset:
        """Transform each predictor lazily, retaining dimensions, CRS and nodata.

        response selects one retained response, or None for their mean.
        All required environmental layers must be present. A pixel missing any
        predictor is masked across all output variables. Dask chunks stay lazy.
        """
        curves = self._curves(response)
        missing = set(self.feature_names_in_) - set(environment.data_vars)
        if missing:
            raise ValueError(f"Missing raster predictors: {sorted(missing)}")
        required = environment[list(self.feature_names_in_)]
        valid = np.isfinite(required.to_array("predictor")).all("predictor")
        result = xr.Dataset(attrs=environment.attrs)
        for name in self.feature_names_in_:
            curve = curves[name]
            result[name] = xr.apply_ufunc(
                np.interp, required[name],
                kwargs={"xp": curve.value.to_numpy(),
                        "fp": curve.cumulative_importance.to_numpy()},
                dask="parallelized", output_dtypes=[float], keep_attrs=False,
            ).where(valid)
            result[name].attrs = {"long_name": f"Cumulative turnover along {name}",
                                  "units": "R2 importance"}
        if environment.rio.crs is not None:
            result = result.rio.write_crs(environment.rio.crs)
        return result

    def plot_biplot(self, X, points=None, *, environment=None, response=None,
                    n_vectors=5, arrow_scale=2.0, label_sites=False,
                    inset=False, figsize=None, buffer=None, stride=4):
        """Plot spatial turnover and a PCA biplot with environmental vectors.

        See pygf.plotting.plot_biplot for scaling, returned data and options.
        response=None uses all retained SNPs; a response name uses that SNP.
        """
        from .plotting import plot_biplot

        return plot_biplot(
            self, X, points, environment=environment, response=response,
            n_vectors=n_vectors, arrow_scale=arrow_scale, label_sites=label_sites,
            inset=inset, figsize=figsize,
            buffer=buffer, stride=stride,
        )

    def plot_turnover(self, variable=None, *, environment=None, points=None,
                      response=None, buffer=None, stride=4, cmap="viridis",
                      ax=None, figsize=(7, 4)):
        """Map one turnover gradient, defaulting to the most important predictor.

        When X came from sample_environment(directory, sites), no other
        arguments are needed. Directory opening, site bounds plus 20% padding,
        downsampling and point overlay are handled internally. See
        pygf.plotting.plot_turnover for advanced options and the result object.
        """
        from .plotting import plot_turnover

        return plot_turnover(
            self, variable, environment=environment, points=points, response=response,
            buffer=buffer, stride=stride, cmap=cmap, ax=ax, figsize=figsize,
        )
