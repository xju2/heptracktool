from typing import List
import itertools

import numpy as np
import pandas as pd


class TrackingData:
    r"""Data for machine learning-based particle tracking.

    The data is stored in two pandas dataframes: particles and measurements.
    The particles dataframe contains the particle information, while the
    measurements dataframe contains both the measurement information and the
    matched particle information.
    """

    def __init__(
        self,
        particles: pd.DataFrame,
        measurements: pd.DataFrame,
        detector_feature_names: List[str],
        name: str = "TrackingData",
    ):
        self.particles = particles
        self.hits = measurements
        self.name = name
        self.check_input_data()

    def check_input_data(self):
        required_measurement_features = [
            "hit_id",
            "x",
            "y",
            "z",
            "particle_id",
            "vx",
            "vy",
            "vz",
        ]
        # check if input data contain all the required features
        # the vx, vy, vz are normally obtained from the particle ataframe
        assert all([f in self.hits.columns for f in required_measurement_features]), (
            "The measurements dataframe does not contain all the required features."
            "Required features are:"
            ",".join(required_measurement_features)
        )

        # check if the particle dataframe contains all the required columns
        required_particle_columns = [
            "particle_id",
            "px",
            "py",
            "pz",
            "vx",
            "vy",
            "vz",
            "radius",
            "status",
            "charge",
            "pdg_id",
        ]
        assert all([f in self.particles.columns for f in required_particle_columns]), (
            "The particles dataframe does not contain all the required columns."
            "Required columns are:"
            ",".join(required_particle_columns)
        )

    def build_true_edges(self, detector_feature_names: List[str]):
        r"""Build the true edges from the input data."""
        if len(detector_feature_names) == 0:
            raise ValueError("The detector_feature_names should not be empty.")

        # Sort by increasing distance from production
        hits = self.hits.assign(
            R=np.sqrt(
                (hits.x - hits.vx) ** 2
                + (hits.y - hits.vy) ** 2
                + (hits.z - hits.vz) ** 2
            )
        )
        # remove noise hits
        signal = hits[hits.particle_id != 0]
        signal = signal.sort_values("R").reset_index(drop=False)

        signal_index_list = (
            signal.groupby(["particle_id"] + detector_feature_names, sort=False)[
                "index"
            ]
            .agg(lambda x: list(x))
            .groupby(level=0)
            .agg(lambda x: list(x))
        )

        e = []
        for row in signal_index_list.values:
            for i, j in zip(row[0:-1], row[1:]):
                e.extend(list(itertools.product(i, j)))

        true_edges = np.array(e).T
        self.true_edges = true_edges
        return true_edges

    def __str__(self):
        return "TrackingData with {} particles and {} measurements.".format(
            len(self.particles), len(self.measurements)
        )
