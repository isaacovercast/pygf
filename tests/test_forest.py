import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from pygf import GradientForest
from pygf.forest import _conditioning_groups, _oob_statistics
from sklearn.ensemble import RandomForestRegressor


@pytest.fixture
def data():
    rng = np.random.default_rng(23)
    x = rng.uniform(-2, 2, (180, 3))
    X = pd.DataFrame(x, columns=["signal", "noise", "other"])
    X["constant"] = 1.0
    y = pd.DataFrame({"step": (x[:, 0] > 0) + rng.normal(0, .04, len(x)),
                      "smooth": np.sin(x[:, 0]) + rng.normal(0, .04, len(x)),
                      "constant": np.ones(len(x)), "missing": np.nan})
    y.loc[::7, "step"] = np.nan
    return X, y


def test_turnover_recovers_signal_and_conserves_skill(data):
    X, y = data
    gf = GradientForest(n_estimators=80, max_features=1.0, random_state=3).fit(X, y)
    assert gf.n_responses_retained_ == 2
    assert gf.importance_.index[0] == "signal"
    assert gf.importance_["signal"] > 10 * gf.importance_["noise"]
    assert gf.importance_["constant"] == 0
    retained = gf.response_scores_.retained
    np.testing.assert_allclose(gf.response_importances_.sum(axis=1)[retained],
                               gf.response_scores_.oob_r2[retained])
    assert gf.importance_.sum() == pytest.approx(gf.response_scores_.oob_r2[retained].mean())
    for name, curve in gf.turnover_.items():
        assert curve.cumulative_importance.iloc[0] == 0
        assert np.all(np.diff(curve.cumulative_importance) >= -1e-15)
        assert curve.cumulative_importance.iloc[-1] == pytest.approx(gf.importance_[name])
    assert gf.oob_predictions_.loc[::7, "step"].isna().all()
    # The sharp response produces faster turnover near its threshold.
    curve = gf.turnover_["signal"]
    central = np.interp([-.4, .4], curve.value, curve.cumulative_importance)
    edge = np.interp([-2, -1.2], curve.value, curve.cumulative_importance)
    assert np.diff(central)[0] > np.diff(edge)[0]


def test_transform_alignment_clamping_and_nan(data):
    X, y = data
    gf = GradientForest(n_estimators=30, random_state=1).fit(X, y)
    values = X.iloc[:3].copy()
    values.loc[0, "signal"] = -100
    values.loc[1, "signal"] = 100
    values.loc[2, "signal"] = np.nan
    transformed = gf.transform(values[values.columns[::-1]])
    assert transformed.index.equals(values.index)
    assert list(transformed) == list(X)
    assert transformed.signal.iloc[0] == 0
    assert transformed.signal.iloc[1] == pytest.approx(gf.importance_["signal"])
    assert np.isnan(transformed.signal.iloc[2])
    np.testing.assert_allclose(gf.predict(X), gf.transform(X))
    with pytest.raises(ValueError, match="indices/order"):
        gf.fit(X, y.iloc[::-1])
    with pytest.raises(ValueError, match="Missing predictors"):
        gf.transform(X.drop(columns="signal"))
    with pytest.raises(NotFittedError):
        GradientForest().transform(X)


def test_zero_skill_and_invalid_inputs(data):
    X, y = data
    with pytest.warns(RuntimeWarning, match="No responses"):
        gf = GradientForest().fit(X, y[["constant", "missing"]])
    assert not gf.transform(X).to_numpy().any()
    invalid = X.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        gf.fit(invalid, y)
    with pytest.raises(ValueError, match="n_bins"):
        GradientForest(n_bins=1).fit(X, y)


def test_reproducibility_and_conditional_mode(data):
    X, y = data
    X["correlated"] = X.signal + .01 * X.noise
    kwargs = dict(n_estimators=30, max_level=2, random_state=83)
    a = GradientForest(**kwargs).fit(X, y)
    b = GradientForest(**kwargs).fit(X, y)
    pd.testing.assert_frame_equal(a.response_scores_, b.response_scores_)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


def test_oob_predictions_use_only_unsampled_rows():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 2)).astype(np.float32)
    y = X[:, 0] ** 2
    forest = RandomForestRegressor(n_estimators=20, random_state=8).fit(X, y)
    _, _, actual = _oob_statistics(forest, X, y, rng, 1, 0, .5)
    expected = np.zeros(len(y))
    counts = np.zeros(len(y))
    for tree, bag in zip(forest.estimators_, forest.estimators_samples_):
        rows = np.setdiff1d(np.arange(len(y)), bag)
        expected[rows] += tree.predict(X[rows])
        counts[rows] += 1
    np.testing.assert_allclose(actual, expected / counts)


def test_conditioning_partitions_correlated_covariate():
    X = np.array([[i, i] for i in range(20)], dtype=np.float32)
    forest = RandomForestRegressor(n_estimators=1, max_features=1.0, random_state=0).fit(X, X[:, 0])
    tree = forest.estimators_[0].tree_
    feature = 1 - tree.feature[0]
    groups = _conditioning_groups(tree, X, feature, np.ones((2, 2)), 1, .5)
    np.testing.assert_array_equal(groups, X[:, tree.feature[0]] > tree.threshold[0])
