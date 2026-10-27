# pyGF

A Python gradient forest for ecological and genomic turnover, with GeoPandas
points, xarray genotype/environmental data, and rioxarray GeoTIFF support.

The worked example is
[`notebooks/gradient_forest_Pguttatus.ipynb`](notebooks/gradient_forest_Pguttatus.ipynb).
It reads `examples/Pguttatus.vcf` and the headerless
`examples/Pguttatus_coords.txt`, samples all 19 bioclim rasters in `envs/`,
fits SNP-specific forests, plots importance and cumulative turnover, and projects
turnover onto a regional raster. The provided genotype `.npy` is not required.

## Install and run

Python 3.9 or later (pip selects dependency versions compatible with your Python):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook,test]'
python -m ipykernel install --prefix .venv --name pygf --display-name 'Python (pyGF)'
jupyter lab notebooks/gradient_forest_Pguttatus.ipynb
```

Choose the **Python (pyGF)** kernel and run the notebook from `notebooks/`;
its data paths are relative to that directory. On systems where geospatial native libraries
cannot build, install prebuilt wheels with `pip install --only-binary=:all:` or
use a conda-forge environment for GeoPandas/rioxarray before installing pyGF.

Run tests with `python -m pytest`. The notebook defaults to up to 256 SNPs,
one per RAD contig, and 100 trees per SNP. Increase `max_variants` and
`n_estimators` in the example calls to scale up. Raster operations remain lazy
until computing/extracting/exporting. The
notebook saves derived files under `notebooks/outputs/`.

## Spatial biplots: all SNPs or one SNP

```python
gf = GradientForest(
    n_estimators=100, random_state=42, keep_response_curves=True,
).fit(X, y)
result = gf.plot_biplot(X, sites, environment='envs', n_vectors=3)
result.figure.savefig('turnover_biplot.png', dpi=200)

snp = gf.response_scores_.oob_r2.idxmax()
single = gf.plot_biplot(X, sites, environment='envs', response=snp, n_vectors=3)
```

`response=None` averages all contributing SNPs. A response name uses that SNP's
own density-corrected curves; unknown or excluded responses raise. Retaining
these compact curves requires `keep_response_curves=True` when fitting, without
keeping the forests. The same `response` option is available on `transform`,
`predict`, and `transform_raster`. The aggregate result is unchanged by retaining
individual curves.

The plot pairs a geographic map with a PCA biplot, using identical RGB colours
for locations and raster cells. PCA is fitted to complete transformed raster
cells, or to transformed `X` when no raster is supplied. Predictors are centred
without standardization, preserving GF importance weights. PC1/2 form the biplot;
PC1/2/3 supply red/green/blue, scaled to the basis' 2nd–98th percentiles. Pass a
bioclim directory: site bounds are reprojected to the raster CRS, buffered by
20% of width/height on each side, and clipped automatically. Directory maps use
every fourth cell by default; `buffer` and `stride` control these choices.
Existing xarray Datasets are used as supplied. Background
biplot points are limited to 5,000 for rendering; all complete cells fit PCA.

Environmental arrows are PCA covariance loadings
(`components_.T * sqrt(explained_variance_)`), ranked by length in PC1/2.
They point towards increasing GF-transformed environmental values, and their
relative lengths show contributions to the displayed turnover pattern.
`arrow_scale` applies one common display multiplier (default 2), preserving
relative lengths. These are not raw-environment regression coefficients or
causal effects; overall GF importance is also returned in `result.vectors`.

`n_vectors` sets the number of labelled arrows; `label_sites=True` labels sample
IDs; `inset=True` places the biplot inside the map. `points` and `environment`
are optional: without spatial information, the result is a standalone biplot.
Locations are matched by sample ID and reprojected to the map CRS. Missing
raster pixels are transparent; rank-one models have a zero second PC.

The return object exposes `figure`, `axes` (keyed by `map`/`biplot`), `pca`,
`scores`, `vectors`, `colors`, and a georeferenced RGBA `rgb` DataArray. It also
records `response`, `basis` and `color_limits`. Each response has its own PCA
and colour calibration; colours across separate SNP plots are not a shared
genetic-distance scale. PCA vectors summarize GF turnover coordinates, not
predicted allele frequencies. For **population-level** inference, fit the forest
to population allele frequencies and population environments, and pass population
locations. The supplied example retains individual sample locations.

This follows the transformed-environment ordination approach in the
[original GF vignette](https://gradientforest.r-forge.r-project.org/biodiversity-survey.pdf),
using [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
Install plotting alone with `pip install -e '.[plot]'`, or use the notebook extra.

## Python API

```python
from pygf import GradientForest, read_coordinates, read_vcf
from pygf import sample_environment

sites = read_coordinates('examples/Pguttatus_coords.txt')
genotypes = read_vcf(
    'examples/Pguttatus.vcf', samples=sites,
    min_call_rate=0.8, min_maf=0.05,
    max_variants=256, one_per_contig=True, random_state=42,
)

X = sample_environment('envs', sites)
y = genotypes.dosage.to_pandas()  # already in coordinate-file order
gf = GradientForest(n_estimators=100, random_state=42).fit(X, y)
coordinates = gf.transform(X)
mapped = gf.plot_turnover()        # most important predictor
gf.plot_turnover('bio1')           # select a predictor
turnover_raster = mapped.turnover  # computed xarray Dataset; ready to export

print(gf.importance_)          # predictor importance, descending
print(gf.response_scores_)     # OOB R², sample coverage, inclusion reason
curve = gf.turnover_['bio1']    # value and cumulative_importance columns
```

`sample_environment` accepts the bioclim directory and automatically opens the
site bounding box with 20% padding on each side. It samples at native resolution,
preserves site order, closes the rasters, and records the source/locations in
DataFrame metadata. `fit` remembers this context as `environment_`, `sites_` and
`environment_buffer_`, so `plot_turnover()` needs no further setup. Its result
contains `figure`, `ax`, `turnover`, `variable` and `response`. Override
`environment`, `points`, `buffer`, `stride` or `response` for a different map.

`samples=sites` uses the DataFrame index to order VCF genotypes to match the
coordinate file (`read_vcf(path, sites)` also works). Climate
rows preserve that order. With `drop_missing=True`, align sites/genotypes to the
remaining climate IDs once before fitting. No inputs are reordered silently.

`open_bioclim` and `transform_raster` remain available for custom xarray workflows;
compute outputs before closing their source Dataset. `predict` aliases `transform`: both return
environmental **turnover coordinates**, not predicted responses. Arrays are
accepted, but DataFrames provide explicit sample and predictor names. DataFrame
X/y indices must match exactly; mismatches raise. During transformation,
predictors are matched by name and returned in training order.

## Method

This is an independent implementation of the gradient forest framework in
[Ellis, Smith & Pitcher (2012), *Ecology* 93:156–168](https://doi.org/10.1890/11-0252.1).
The [original project's vignette](https://gradientforest.r-forge.r-project.org/biodiversity-survey.pdf)
describes the progression from individual forests to environmental turnover.

1. Fit a bootstrapped scikit-learn regression forest to each numeric response.
   A response may be a species abundance, allele frequency, or individual SNP
   dosage. Missing responses are excluded per forest, without imputation;
   predictors must be finite. Constant and sparsely observed responses are
   recorded and skipped.
2. Aggregate predictions only from trees where a sample was out of bag. Compute
   OOB R² on samples with OOB votes. For each tree, permute a predictor within
   its OOB observations and measure the increase in MSE, then average deltas
   across trees and permutation repeats. Retain responses above `min_oob_r2`
   (default zero) with positive importance.
3. Clip negative permutation importances to zero. Normalize them across
   predictors and multiply by that response's OOB R². This allocates predictive
   skill to predictors while preserving its total.
4. Collect tree split thresholds and weighted decreases in squared-error
   impurity into `n_bins` bins along each predictor. Smooth split mass with a
   Gaussian kernel. When `density_correction=True`, divide by the smoothed
   histogram of predictor values observed for that response. Both histograms
   share a Silverman-style bandwidth; a small denominator floor avoids division
   by zero. Normalize the corrected mass, integrate, and scale to the allocated
   predictor importance.
5. Average curves over contributing responses. Each curve starts at zero and
   ends at its mean allocated importance; all endpoints sum to mean retained
   OOB R². Interpolate curves linearly for new environments, clamping outside
   training ranges. NaNs remain NaN. Raster pixels missing any predictor are
   masked across every output layer.

`max_level=0` uses marginal permutation. Positive `max_level` uses a bounded
conditional grid: find up to that many shallow splits in each tree on other
predictors with absolute Pearson correlation above `corr_threshold`, and permute
only among OOB observations with the same split signature. This preserves those
coarse associations; it is **an approximation**, not the recursive conditional
partition algorithm of R `extendedForest`.

This implementation is not numerically interchangeable with R `gradientForest`:
it uses scikit-learn CART trees, configurable compact histograms, Gaussian
smoothing rather than R's density machinery, the conditional grid above, and
explicit averaging over responses with positive skill and importance. No R
parity validation is claimed. Regression responses are supported; categorical
classification responses and combined forests from different surveys are not.

`keep_forests=False` (default) discards trees after each response, keeping memory
bounded by one forest plus compact curves and diagnostics. With
`keep_forests=True`, fitted per-response forests are in `forests_`.
`permutation_importances_` contains raw OOB MSE increases;
`response_importances_` contains allocated R² (zero for excluded responses).
`feature_importances_` is in input order; `importance_` is sorted.

## Data and interpretation

- `read_coordinates` expects sample ID, longitude, latitude in WGS84; pass
  `header=0` for named `sample,longitude,latitude` columns. The helper returns a
  GeoDataFrame indexed by sample ID.
- `read_vcf` supports plain or gzip VCF with diploid, biallelic SNP GT fields.
  Phase separators and GT position in FORMAT are respected. Missing/partial GTs
  are NaN, multiallelic/non-SNP and failed FILTER records are skipped, and called
  non-diploid/invalid GTs raise. It does not read BCF or apply depth/quality
  filters beyond FILTER. Requested sample IDs must be present and unique.
- Call rate and MAF are computed on selected samples. A bounded uniform reservoir
  samples all eligible records, not a prefix. `one_per_contig=True` takes the
  first eligible SNP per contig before reservoir selection: suitable for this
  RAD example, **not** whole-chromosome data or a substitute for LD pruning.
  `max_variants=None` loads every eligible record into memory.
- `open_bioclim` recognizes `*bio_N.tif` filenames, opens one-band rasters,
  applies encoded nodata/scale/offset, and rejects mismatched CRS or grids.
  Bounds are in the raster CRS; `points=sites` computes the buffered bounds
  automatically. Collapsed site extents receive at least one cell of padding.
  `sample_environment` accepts a directory or an existing Dataset and reprojects GeoPandas points,
  selects nearest cells, and returns NaN for outside-grid or nodata locations.
- The notebook models individual genotypes. Population allele frequencies need
  population assignments and an appropriate population-level environmental
  matrix, neither of which is inferred from coordinates.
- OOB scores do not address spatial autocorrelation, neutral population
  structure, or transferability. For inference, use spatially blocked validation,
  repeated seeds, more trees/loci, and ecological/genetic controls. Turnover and
  environmental offset alone do not establish adaptation or predict fitness.

## Validation

Tests check recovery of a simulated threshold response, conservation of allocated
OOB R², monotone curves, correct OOB sample exclusion, deterministic fits,
conditional grouping, sample alignment, VCF filtering/missingness and reservoir
selection, CRS reprojection, nodata and outside-grid masks, raster grid mismatch
rejection, and agreement between point and lazy raster transformations. The
example notebook demonstrates the API and exports the real-data outputs.
Additional tests verify per-SNP curves against their aggregate, PCA covariance
loadings, identical map/site colour calibration, spatial ID alignment, raster
masking and reprojection, and rank-one ordinations. Directory tests verify 20%
buffered bounds in the raster CRS, remembered map context, automatic predictor
selection, single-point windows, and sites entirely outside raster coverage.

The supplied example was executed end to end with Python 3.9.19: 36 samples,
19 bioclim predictors, and 256 SNPs sampled from 21,358 eligible RAD contigs
(229,322 VCF records scanned). With the notebook's seed and settings, 136 SNPs
contributed to turnover. Automated tests passed on Python 3.9 and 3.11. The executed notebook
retains its figures and outputs; exported GeoTIFFs use EPSG:4326 and preserve
the common nodata mask.

Editable installation was also verified with Python 3.9.19, using pip-selected
compatible dependency versions.
