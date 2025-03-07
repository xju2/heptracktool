"""Utilities for reading raw athena files dumped from the DumpObject."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def read_track_candidates(fname):
    with open(fname) as f:
        lines = f.readlines()
        track_candidates = [[int(x) for x in line.strip().split(",")] for line in lines]
    return track_candidates


def read_spacepoints(filename):
    spacepoints = pd.read_csv(
        filename,
        header=None,
        engine="python",
        names=["hit_id", "x", "y", "z", "cluster_index_1", "cluster_index_2"],
    )
    # pixel_hits = spacepoints[pd.isna(spacepoints["cluster_index_2"])]
    # strip_hits = spacepoints[~pd.isna(spacepoints["cluster_index_2"])]
    return spacepoints


def split_spacepoints(spacepoints):
    pixel_hits = spacepoints[pd.isna(spacepoints["cluster_index_2"])]
    strip_hits = spacepoints[~pd.isna(spacepoints["cluster_index_2"])]
    return pixel_hits, strip_hits


def read_particles(filename):
    field_names = [
        "subevent",
        "barcode",
        "px",
        "py",
        "pz",
        "pt",
        "eta",
        "vx",
        "vy",
        "vz",
        "radius",
        "status",
        "charge",
        "pdgId",
        "pass",
        "vProdNIn",
        "vProdNOut",
        "vProdStatus",
        "vProdBarcode",
    ]
    particles = pd.read_csv(filename, header=None, engine="python", sep=r",#")

    particles = particles[0].str.split(",", expand=True)
    particles.columns = field_names
    particles = particles.astype(
        {
            "subevent": int,
            "barcode": int,
            "px": float,
            "py": float,
            "pz": float,
            "pt": float,
            "eta": float,
            "vx": float,
            "vy": float,
            "vz": float,
            "radius": float,
            "status": int,
            "charge": float,
            "pdgId": int,
            "pass": str,
            "vProdNIn": int,
            "vProdNOut": int,
            "vProdStatus": int,
            "vProdBarcode": int,
        }
    )

    particle_ids = get_particle_ids(particles)
    particles.insert(0, "particle_id", particle_ids)

    return particles


def get_particle_ids(df) -> pd.Series:
    barcode = df.barcode.astype(str)
    subevent = df.subevent.astype(str)

    # convert barcode to 7 digits
    max_length = 7
    particle_ids = subevent + barcode.str.pad(width=max_length, fillchar="0")
    return particle_ids


def get_particle_ids_int64(df) -> pd.Series:
    barcode = df.barcode.astype(np.int64)
    subevent = df.subevent.astype(np.int64)

    # convert barcode to 7 digits
    # max_length = 7
    particle_ids = subevent * 10_000_000 + barcode
    return particle_ids


def read_true_track(filename):
    """Read fitted tracks information from a file."""
    true_track = pd.read_csv(filename, header=None, engine="python", sep=r",#")

    # first block on track indices
    trk_index = true_track[0].str.split(",", expand=True)
    trk_index.columns = ["trkid", "fitter", "material"]
    trk_index = trk_index.astype(
        {"trkid": "int32", "fitter": "int32", "material": "int32"}
    )

    # second block on track info
    trk_info = true_track[3].str.split(",", expand=True).drop(0, axis=1)
    trk_info.columns = [
        "nDoF",
        "chi2",
        "charge",
        "x",
        "y",
        "z",
        "px",
        "py",
        "pz",
        "mot",
        "oot",
        "trkid",
        "subevent",
        "barcode",
        "probability",
        "pdgId",
        "status",
    ]
    trk_info = trk_info.astype(
        {
            "nDoF": "int32",
            "chi2": "float32",
            "charge": "int32",
            "x": "float32",
            "y": "float32",
            "z": "float32",
            "px": "float32",
            "py": "float32",
            "pz": "float32",
            "mot": "int32",
            "oot": "int32",
            "trkid": "int32",
            "subevent": "int32",
            "barcode": "int32",
            "probability": "float32",
            "pdgId": "int32",
            "status": "int32",
        }
    )
    trk_info = trk_info.drop("trkid", axis=1)

    particle_id = get_particle_ids(trk_info)
    trk_info = trk_info.assign(
        particle_id=particle_id,
        pt=np.sqrt(trk_info.px**2 + trk_info.py**2),
    )

    # merge the two blocks
    tracks = pd.concat([trk_index, trk_info], axis=1)
    return tracks


def read_detailed_matching(filename):
    detailed_true_tracks = pd.read_csv(
        filename, header=None, engine="python", sep=r",#"
    )
    num_matched = detailed_true_tracks[0].str.split(",", expand=True)
    num_matched.columns = ["trkid", "num_matched"]
    num_matched = num_matched.astype({"trkid": "int32", "num_matched": "int32"})

    primary_matched = detailed_true_tracks[1].str.split(",", expand=True)[[1, 2]]
    primary_matched.columns = ["subevent", "barcode"]
    primary_matched = primary_matched.astype({"subevent": "int32", "barcode": "int32"})

    true_hits = detailed_true_tracks[2].str.split(",", expand=True)[[1, 2]]
    true_hits.columns = ["true_pixel_hits", "true_sct_hits"]
    true_hits = true_hits.astype({"true_pixel_hits": "int32", "true_sct_hits": "int32"})

    reco_hits = detailed_true_tracks[3].str.split(",", expand=True)[[1, 2]]
    reco_hits.columns = ["reco_pixel_hits", "reco_sct_hits"]
    reco_hits = reco_hits.astype({"reco_pixel_hits": "int32", "reco_sct_hits": "int32"})

    common_hits = detailed_true_tracks[4].str.split(",", expand=True)[[1, 2]]
    common_hits.columns = ["common_pixel_hits", "common_sct_hits"]
    common_hits = common_hits.astype(
        {"common_pixel_hits": "int32", "common_sct_hits": "int32"}
    )

    detailed_matching = pd.concat(
        [num_matched, primary_matched, true_hits, reco_hits, common_hits], axis=1
    )

    particle_id = get_particle_ids(detailed_matching)
    detailed_matching = detailed_matching.assign(particle_id=particle_id)

    return detailed_matching


def truth_match_clusters(pixel_hits, strip_hits, clusters):
    """
    Here we handle the case where a pixel spacepoint belongs to exactly one cluster, but
    a strip spacepoint belongs to 0, 1, or 2 clusters, and we only accept the case of 2 clusters
    with shared truth particle_id
    """
    pixel_clusters = pixel_hits.merge(
        clusters[["cluster_id", "particle_id"]],
        left_on="cluster_index_1",
        right_on="cluster_id",
        how="left",
    ).drop("cluster_id", axis=1)

    strip_clusters = strip_hits.merge(
        clusters[["cluster_id", "particle_id"]],
        left_on="cluster_index_1",
        right_on="cluster_id",
        how="left",
    )
    strip_clusters = strip_clusters.merge(
        clusters[["cluster_id", "particle_id"]],
        left_on="cluster_index_2",
        right_on="cluster_id",
        how="left",
        suffixes=("_1", "_2"),
    ).drop(["cluster_id_1", "cluster_id_2"], axis=1)

    # Get clusters that share particle ID
    matching_clusters = strip_clusters.particle_id_1 == strip_clusters.particle_id_2
    strip_clusters["particle_id"] = strip_clusters["particle_id_1"].where(
        matching_clusters, other=0
    )
    strip_clusters = strip_clusters.drop(["particle_id_1", "particle_id_2"], axis=1)
    truth_spacepoints = pd.concat([pixel_clusters, strip_clusters], ignore_index=True)
    truth_spacepoints = truth_spacepoints.astype({"particle_id": "str"})
    return truth_spacepoints


def add_region_labels(
    hits: pd.DataFrame, region_labels: dict[int, dict[str, Any]]
) -> pd.DataFrame:
    """Label the detector regions (forward-endcap pixel, forward-endcap strip, etc.)."""
    for region_label, conditions in region_labels.items():
        condition_mask = np.logical_and.reduce(
            [
                hits[condition_column] == condition
                for condition_column, condition in conditions.items()
            ]
        )
        hits.loc[condition_mask, "region"] = region_label

    assert (
        hits.region.isna()
    ).sum() == 0, "There are hits that do not belong to any region!"
    return hits


def merge_spacepoints_clusters(spacepoints, clusters):
    """
    Finally, we merge the features of each cluster with the spacepoints - where a spacepoint may
    own 1 or 2 signal clusters, and thus we give the suffixes _1, _2
    """

    spacepoints = spacepoints.merge(
        clusters.drop(["particle_id", "side"], axis=1),
        left_on="cluster_index_1",
        right_on="cluster_id",
        how="left",
    ).drop("cluster_id", axis=1)

    unique_cluster_fields = [
        "cluster_id",
        "cluster_x",
        "cluster_y",
        "cluster_z",
        # 'eta_angle', 'phi_angle',
    ]  # These are fields that is unique to each cluster (therefore they need the _1, _2 suffix)

    spacepoints = spacepoints.merge(
        clusters[unique_cluster_fields],
        left_on="cluster_index_2",
        right_on="cluster_id",
        how="left",
        suffixes=("_1", "_2"),
    ).drop("cluster_id", axis=1)

    # Ignore duplicate entries (possible if a particle has duplicate hits in the same clusters)
    spacepoints = spacepoints.drop_duplicates(
        ["hit_id", "cluster_index_1", "cluster_index_2", "particle_id"]
    ).fillna(-1)

    return spacepoints


def particles_of_interest(particles):
    """
    Keep only particles of interest.
    """
    results = particles[
        (particles.eta.abs() < 4)
        & (particles.radius < 260)
        & (particles.charge.abs() > 0)
    ]

    return results


def add_module_id(hits, module_lookup):
    """
    Add the module ID to the hits dataframe
    """
    if "module_id" in hits:
        return hits
    cols_to_merge = [
        "hardware",
        "barrel_endcap",
        "layer_disk",
        "eta_module",
        "phi_module",
        "ID",
    ]
    merged_hits = hits.merge(
        module_lookup[cols_to_merge + ["ID"]], on=cols_to_merge, how="left"
    )
    merged_hits = merged_hits.rename(columns={"ID": "module_id"})

    assert (
        hits.shape[0] == merged_hits.shape[0]
    ), "Merged hits dataframe has different number of rows - possibly missing modules from lookup"
    assert (
        hits.shape[1] + 1 == merged_hits.shape[1]
    ), "Merged hits dataframe has different number of columns"

    return merged_hits
