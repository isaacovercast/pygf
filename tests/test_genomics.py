import gzip

import numpy as np
import pandas as pd
import pytest

from pygf import read_coordinates, read_vcf


def write_vcf(path):
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC\tD\n"
        "locus1\t1\tv1\tA\tG\t.\tPASS\t.\tDP:GT\t2:0/0\t2:0|1\t2:1/1\t2:./1\n"
        "locus1\t2\tv2\tA\tT\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t0/0\n"
        "locus2\t1\tv3\tA\tC,G\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t0/0\n"
        "locus3\t1\tv4\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/0\t0/0\t0/0\n"
        "locus4\t1\tv5\tA\tC\t.\tq10\t.\tGT\t0/0\t0/1\t1/1\t0/0\n"
        "locus5\t1\tv6\tA\tC\t.\tPASS\t.\tGT\t1/1\t0/1\t0/0\t0/0\n")


def test_vcf_dosages_filters_order_and_contig_thinning(tmp_path):
    path = tmp_path / "tiny.vcf"
    write_vcf(path)
    ds = read_vcf(path, samples=["C", "A", "D", "B"], min_call_rate=.75, one_per_contig=True)
    assert ds.sizes == {"sample": 4, "variant": 2}
    np.testing.assert_allclose(ds.dosage[:, 0], [2, 0, np.nan, 1], equal_nan=True)
    assert ds.maf[0] == .5
    assert ds.call_rate[0] == .75
    assert ds.attrs["eligible_variants"] == 2
    assert list(read_vcf(path).vcf_id.values) == ["v2", "v6"]
    with pytest.raises(ValueError, match="absent"):
        read_vcf(path, samples=["UNKNOWN"])
    compressed = tmp_path / "tiny.vcf.gz"
    with gzip.open(compressed, "wt") as stream:
        stream.write(path.read_text())
    xr_plain = read_vcf(path).dosage
    xr_compressed = read_vcf(compressed).dosage
    assert xr_compressed.identical(xr_plain)


def test_reservoir_reproducible_and_not_prefix(tmp_path):
    path = tmp_path / "many.vcf"
    write_vcf(path)
    header = "\n".join(path.read_text().splitlines()[:2])
    rows = [f"loc{i}\t1\tv{i}\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t0/0"
            for i in range(100)]
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    a = read_vcf(path, max_variants=8, random_state=42)
    b = read_vcf(path, max_variants=8, random_state=42)
    assert a.equals(b)
    assert a.attrs["eligible_variants"] == 100
    assert any(int(chrom[3:]) > 20 for chrom in a.chrom.values)


def test_coordinates(tmp_path):
    path = tmp_path / "coords.txt"
    path.write_text("B,-93,30\nA,-95,34\n")
    points = read_coordinates(path)
    assert list(points.index) == ["B", "A"]
    assert points.crs.to_epsg() == 4326
    assert points.geometry.x.iloc[0] == -93
    path.write_text("A,200,30\n")
    with pytest.raises(ValueError, match="longitude/latitude"):
        read_coordinates(path)


@pytest.mark.parametrize("geospatial", [False, True])
def test_vcf_accepts_coordinate_dataframe(tmp_path, geospatial):
    path = tmp_path / "tiny.vcf"
    write_vcf(path)
    coords = tmp_path / "coords.txt"
    coords.write_text("C,-93,30\nA,-95,34\nD,-96,35\nB,-94,33\n")
    sites = read_coordinates(coords)
    if not geospatial:
        sites = pd.DataFrame(sites.drop(columns="geometry"))
    expected = read_vcf(path, samples=sites.index, min_call_rate=.75)
    assert read_vcf(path, samples=sites, min_call_rate=.75).identical(expected)
    assert read_vcf(path, sites, min_call_rate=.75).identical(expected)
    assert read_vcf(path, sites).dosage.to_pandas().index.equals(sites.index)
    with pytest.raises(ValueError, match="unique"):
        read_vcf(path, pd.concat([sites, sites]))
