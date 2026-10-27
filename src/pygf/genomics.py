"""Small streaming VCF reader for diploid, biallelic SNP dosages."""

from __future__ import annotations

import gzip
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def read_coordinates(path, *, header=None, sep=",") -> gpd.GeoDataFrame:
    """Read sample, longitude, latitude columns as WGS84 points.

    Defaults match examples/Pguttatus_coords.txt, which has no header.
    With header=0, the input must have sample, longitude, latitude columns.
    """
    frame = pd.read_csv(path, sep=sep, header=header,
                        names=["sample", "longitude", "latitude"] if header is None else None,
                        dtype={"sample": str})
    if not {"sample", "longitude", "latitude"}.issubset(frame.columns):
        raise ValueError("Coordinates require sample, longitude, latitude columns")
    if frame["sample"].isna().any() or frame["sample"].duplicated().any():
        raise ValueError("Sample IDs must be present and unique")
    xy = frame[["longitude", "latitude"]].to_numpy(dtype=float)
    if (not np.isfinite(xy).all() or (np.abs(xy[:, 0]) > 180).any()
            or (np.abs(xy[:, 1]) > 90).any()):
        raise ValueError("Coordinates must be finite longitude/latitude in degrees")
    frame = frame.set_index("sample")
    return gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(frame.longitude, frame.latitude),
                            crs="EPSG:4326")


def read_vcf(path, samples=None, *, min_call_rate=0.8, min_maf=0.05,
             max_variants=None, one_per_contig=False, random_state=None) -> xr.Dataset:
    """Stream a VCF (.vcf or .vcf.gz) to a sample × variant dosage Dataset.

    samples may be a coordinate DataFrame/GeoDataFrame (its index supplies
    sample IDs and order), an iterable of IDs, or None for VCF order.

    Only PASS/unfiltered diploid biallelic SNPs are considered. Phased and
    unphased GTs are supported. A partial/missing GT is NaN, never reference.
    Call rate and MAF are computed on the selected samples before selection.
    Constant dosage loci are discarded. Non-diploid called GTs raise an error.

    max_variants uses uniform reservoir sampling over ALL eligible records,
    rather than taking the beginning of the file. With one_per_contig=True,
    only the first eligible SNP on each contig enters that reservoir (useful
    for RAD loci; this is not LD pruning). No imputation is performed.
    """
    if not 0 <= min_call_rate <= 1 or not 0 <= min_maf <= 0.5:
        raise ValueError("min_call_rate must be in [0, 1] and min_maf in [0, 0.5]")
    if max_variants is not None and (not isinstance(max_variants, int) or max_variants < 1):
        raise ValueError("max_variants must be a positive integer or None")
    if isinstance(samples, pd.DataFrame):
        samples = samples.index
    rng = np.random.default_rng(random_state)
    opener = gzip.open if str(path).endswith(".gz") else open
    records = []
    chosen_samples = None
    seen_contigs = set()
    eligible = 0
    scanned = 0
    with opener(Path(path), "rt") as stream:
        for line in stream:
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip().split("\t")
                available = fields[9:]
                if not available or len(available) != len(set(available)):
                    raise ValueError("VCF must have unique sample IDs and genotype columns")
                chosen_samples = list(samples) if samples is not None else available
                if not chosen_samples or len(chosen_samples) != len(set(chosen_samples)):
                    raise ValueError("Requested sample IDs must be unique and nonempty")
                missing = set(chosen_samples) - set(available)
                if missing:
                    raise ValueError(f"Samples absent from VCF: {sorted(missing)}")
                columns = [9 + available.index(sample) for sample in chosen_samples]
                n_fields = len(fields)
                continue
            if line.startswith("#"):
                continue
            if chosen_samples is None:
                raise ValueError("Missing #CHROM VCF header")
            fields = line.rstrip().split("\t")
            scanned += 1
            if len(fields) != n_fields:
                raise ValueError(f"Wrong number of VCF columns at record {scanned}")
            chrom, position, identifier, ref, alt = fields[:5]
            if (len(ref) != 1 or len(alt) != 1 or ref.upper() not in "ACGT"
                    or alt.upper() not in "ACGT" or ref == alt
                    or fields[6] not in ("PASS", ".")):
                continue
            if one_per_contig and chrom in seen_contigs:
                continue
            formats = fields[8].split(":")
            if "GT" not in formats:
                continue
            gt_column = formats.index("GT")
            dosage = np.full(len(columns), np.nan, dtype=np.float32)
            for i, column in enumerate(columns):
                parts = fields[column].split(":")
                genotype = parts[gt_column] if len(parts) > gt_column else "."
                alleles = genotype.replace("|", "/").split("/")
                if "." in alleles:
                    continue
                if len(alleles) != 2 or any(a not in ("0", "1") for a in alleles):
                    raise ValueError(f"Expected diploid biallelic GT at {chrom}:{position}, "
                                     f"sample {chosen_samples[i]}: {genotype}")
                dosage[i] = int(alleles[0]) + int(alleles[1])
            observed = dosage[np.isfinite(dosage)]
            call_rate = len(observed) / len(dosage)
            if not len(observed) or call_rate < min_call_rate or np.ptp(observed) == 0:
                continue
            frequency = float(observed.mean() / 2)
            maf = min(frequency, 1 - frequency)
            if maf < min_maf:
                continue
            seen_contigs.add(chrom)
            eligible += 1
            record = (scanned, chrom, int(position), identifier, ref, alt,
                      call_rate, maf, dosage)
            if max_variants is None or len(records) < max_variants:
                records.append(record)
            else:
                index = int(rng.integers(eligible))
                if index < max_variants:
                    records[index] = record
    if chosen_samples is None:
        raise ValueError("Missing #CHROM VCF header")
    if not records:
        raise ValueError("No variable biallelic SNPs pass the VCF filters")
    records.sort(key=lambda record: record[0])
    names = [f"{r[1]}:{r[2]}:{r[4]}:{r[5]}" for r in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate chromosome:position:ref:alt variant identifiers")
    return xr.Dataset(
        {"dosage": (("sample", "variant"), np.column_stack([r[8] for r in records])),
         "chrom": ("variant", [r[1] for r in records]),
         "position": ("variant", [r[2] for r in records]),
         "vcf_id": ("variant", [r[3] for r in records]),
         "ref": ("variant", [r[4] for r in records]),
         "alt": ("variant", [r[5] for r in records]),
         "call_rate": ("variant", [r[6] for r in records]),
         "maf": ("variant", [r[7] for r in records])},
        coords={"sample": chosen_samples, "variant": names},
        attrs={"source": str(path), "records_scanned": scanned,
               "eligible_variants": eligible, "min_call_rate": min_call_rate,
               "min_maf": min_maf, "one_per_contig": int(one_per_contig),
               "dosage_encoding": "alternate-allele count (0, 1, 2); missing=NaN"},
    )
