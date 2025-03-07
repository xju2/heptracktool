"""Compare tracking algorithms represented by two AthenaRawDataReader.
Each tracking algorithm is run in Athena frameework and produces a TrackCollection.
The two TrackCollections are dumped to text files and read by the reader.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrackAlgComparator:
    def __init__(
        self, reader, other_reader, min_reco_clusters=5, name="TrackAlgComparator"
    ) -> None:
        super().__init__()
        self.reader = reader
        self.other_reader = other_reader
        self.min_reco_clusters = min_reco_clusters
        self.name = name

        # common tracks
        self.common_track_indices: list[tuple[int, list[int], int]] = None
        self.common_info: pd.DataFrame = None
        self.common_track: pd.DataFrame = None
        self.common_other_track: pd.DataFrame = None

        # disjoint tracks
        self.disjoint_track_indices: list[tuple[int, list[int]]] = None
        self.disjoint_info: pd.DataFrame = None
        self.disjoint_track: pd.DataFrame = None
        self.disjoint_other_track: pd.DataFrame = None

        # unmatched tracks
        self.unmatched_tracks: list[tuple[int, list[int]]] = None

        # optional to reverse the comparison in the run time.
        self._reverse_comparison = False
        self.redo_comparison = False

    def readers(self):
        return (
            (self.reader, self.other_reader)
            if not self.reverse_comparison
            else (self.other_reader, self.reader)
        )

    @property
    def reverse_comparison(self) -> bool:
        """Return the reverse comparison flag."""
        return self._reverse_comparison

    @reverse_comparison.setter
    def reverse_comparison(self, do_reverse: bool = False) -> None:
        """Set the reverse comparison flag."""
        if self._reverse_comparison != do_reverse:
            # changing the comparison direction
            # so we need to redo the comparison
            self.redo_comparison = True
            self._reverse_comparison = do_reverse

    def compare_track_contents(self) -> tuple[Any, Any, Any]:
        """Compares each track in the first track collection
        with all tracks in the other track collection. Each track is labeled as
        common (also means matched), disjoint, or unmatched.
        common:
          the contents of the track are exactly the same as
          another track in the other collection (matched).
        disjoint:
          the contents of the track do not exist in any track in the other collection.
        unmatched:
          track do not match to any track in the other collection.

        Returns
        -------
            common_tracks: list of tuples of (track_id, track, cluster ids, other_track_id)
            unmatched_tracks: list of tuples of (track_id, cluster ids)
            disjoint_tracks: list of tuples of (track_id, cluster ids)

        Raises
        ------
            ValueError: If the track contents are empty.
        """
        if not self.redo_comparison and self.common_info is not None:
            return (
                self.common_track_indices,
                self.unmatched_tracks,
                self.disjoint_track_indices,
            )

        track_reader, other_track_reader = self.readers()
        min_num_clusters = self.min_reco_clusters

        label, other_label = track_reader.name, other_track_reader.name
        tracks, other_tracks = (
            track_reader.clusters_on_track,
            other_track_reader.clusters_on_track,
        )
        if tracks is None or other_tracks is None:
            raise ValueError("Track contents are empty.")

        num_matched = 0
        num_issubset = 0
        num_other_issubset = 0
        num_disjoints = 0

        unmatched_tracks = []
        disjoint_tracks = []
        common_tracks = []

        tot_tracks, tot_other_tracks = len(tracks), len(other_tracks)
        print(
            f"{tot_tracks} {label} tracks compared to {tot_other_tracks} {other_label} tracks.\n"
            f"Require min_num_clusters = {min_num_clusters} only for {label} tracks."
        )

        tot_filtered_tracks = 0
        for trkid, track in enumerate(tracks):
            if len(track) < min_num_clusters:
                continue
            tot_filtered_tracks += 1
            found_a_match = False
            all_disjoint = True
            found_as_subset = False
            found_other_as_subset = False
            ckf_idx = -1

            for idx, other_track in enumerate(other_tracks):
                track_set = set(track)
                other_track_set = set(other_track)
                if not track_set.isdisjoint(other_track_set):
                    all_disjoint = False
                if track_set == other_track_set:
                    found_a_match = True
                    ckf_idx = idx
                if track_set < other_track_set:
                    found_as_subset = True
                if track_set > other_track_set:
                    found_other_as_subset = True

            if found_a_match:
                num_matched += 1
                common_tracks.append((trkid, track, ckf_idx))
            else:
                unmatched_tracks.append((trkid, track))

            if all_disjoint:
                num_disjoints += 1
                disjoint_tracks.append((trkid, track))
            if found_as_subset:
                num_issubset += 1
            if found_other_as_subset:
                num_other_issubset += 1

        print(
            f"Total # of {label} tracks: {tot_tracks}. After filtering, "
            f"# of {label} tracks: {tot_filtered_tracks} ({tot_filtered_tracks / tot_tracks:.3%})"
        )
        print(
            f"Matched: {num_matched}, {tot_filtered_tracks}, {num_matched / tot_filtered_tracks:.4%}"
        )
        print(
            f"{label} is a subset: {num_issubset}, {tot_filtered_tracks}, "
            f"{num_issubset / tot_filtered_tracks:.4%}"
        )
        print(
            f"{other_label} is a subset: {num_other_issubset}, {tot_filtered_tracks}, "
            f"{num_other_issubset / tot_filtered_tracks:.4%}"
        )
        print(
            f"Disjoint:  {num_disjoints}, {tot_filtered_tracks}, {num_disjoints / tot_filtered_tracks:.4%}"
        )

        (
            self.common_track_indices,
            self.unmatched_tracks,
            self.disjoint_track_indices,
        ) = (common_tracks, unmatched_tracks, disjoint_tracks)

        # if we have already done the comparison, set the redo flag to False
        if self.redo_comparison:
            self.redo_comparison = False

        return (common_tracks, unmatched_tracks, disjoint_tracks)

    def analyse_common_track(self) -> pd.DataFrame:
        """Return a common track object.

        Returns
        -------
            df_common: pd.DataFrame
        """
        self.compare_track_contents()

        common_tracks = self.common_track_indices
        reader, other_reader = self.readers()

        trk_id = np.array([x[0] for x in common_tracks])
        other_trk_id = np.array([x[2] for x in common_tracks])
        nclusters = np.array([len(x[1]) for x in common_tracks])
        df_common = pd.DataFrame({
            f"{reader.name}_trkid": trk_id,
            f"{other_reader.name}_trkid": other_trk_id,
            "nclusters": nclusters,
        })
        self.common_info = df_common

        # `isin` does not preserve the order of the original array
        # but the dataframe index is the same as trk_id.
        self.common_track = reader.tracks.loc[trk_id]
        self.other_common_track = other_reader.tracks.loc[other_trk_id]

        return df_common

    def plot_common_tracks(self) -> tuple[np.array, np.array]:
        """Analyze the common tracks. Compare their chi2 and other metrics."""
        df = self.analyse_common_track()
        reader, other_reader = self.readers()
        label, other_label = reader.name, other_reader.name

        num_common_tracks = len(df)

        # number of clusters
        plt.title("Common tracks")
        plt.hist(
            df.nclusters.values,
            bins=31,
            range=(-0.5, 30.5),
            label=f"Total {num_common_tracks}",
            alpha=0.5,
        )
        plt.legend()
        plt.xlabel("number of clusters")
        plt.show()

        # chi2 / ndof
        chi2_hist_config = {
            "bins": 50,
            "range": (0, 4),
            "alpha": 0.5,
            "histtype": "step",
            "lw": 2,
        }
        chi2 = self.common_track.chi2.to_numpy() / self.common_track.nDoF.to_numpy()
        other_chi2 = (
            self.other_common_track.chi2.to_numpy() / self.other_common_track.nDoF.to_numpy()
        )
        plt.title("Common Tracks")
        plt.hist(chi2, **chi2_hist_config, label=label)
        plt.hist(other_chi2, **chi2_hist_config, label=other_label)
        plt.xlim(0, 4)
        plt.xlabel(r"$\chi^2$/ndof")
        plt.legend()
        plt.show()

        # scatter plot for chi2 / ndof
        plt.title("Common Tracks")
        config = {"s": 10, "alpha": 0.5}
        plt.scatter(chi2, other_chi2, **config)
        plt.plot([0, 4], [0, 4], color="red", linestyle="--")
        plt.xlim(0, 4)
        plt.ylim(0, 4)
        plt.xlabel(rf"{label} $\chi^2$/ndof")
        plt.ylabel(rf"{other_label} $\chi^2$/ndof")
        plt.show()

        # difference in chi2 / ndof
        delta_chi2 = chi2 - other_chi2
        plt.title("Common Tracks")
        bin_values, _, _ = plt.hist(delta_chi2, bins=50, range=(-2, 2), alpha=0.5)
        max_bin_value, min_bin_value = np.max(bin_values), np.min(bin_values)
        y_start = max_bin_value * 0.5
        delta = (max_bin_value - min_bin_value) * 0.08
        plt.text(-1.5, y_start, f"Mean: {np.mean(delta_chi2):8.4f}", fontsize=12)
        plt.text(-1.5, y_start - delta, f"Std:  {np.std(delta_chi2):8.4f}", fontsize=12)
        plt.xlabel(rf"({label} - {other_label}) $\chi^2$/ndof")
        plt.plot([0, 0], [0, max_bin_value], color="red", linestyle="--")
        plt.show()

        return chi2, other_chi2

    def match_to_truth(self) -> None:
        """Match a track to the truth track."""
        reader, other_reader = self.readers()
        reader.match_to_truth()
        other_reader.match_to_truth()

    def plot_disjoint_tracks(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze the disjoint tracks. We have to the comparison twice.
        Once looping over the tracks in the first reader,
        and once looping over the tracks in the second reader.
        """
        reader, other_reader = self.readers()
        self.match_to_truth()  # will be used to quanlify the disjoint tracks
        label, other_label = reader.name, other_reader.name

        # perform the comparison to find disjoint tracks
        # for reader and other_reader
        self.reverse_comparison = True
        _, _, disjoint_other_tracks = self.compare_track_contents()

        # reset the flag
        self.reverse_comparison = False
        _, _, disjoint_tracks = self.compare_track_contents()

        # compare the two disjoint track lists
        num_disjoints = len(disjoint_tracks)
        num_other_disjoints = len(disjoint_other_tracks)

        cluster_config = {"bins": 31, "range": (-0.5, 30.5), "alpha": 0.5}
        plt.title("Disjoint tracks")
        plt.hist(
            [len(x[1]) for x in disjoint_tracks],
            **cluster_config,
            label=f"{label} {num_disjoints}",
        )
        plt.hist(
            [len(x[1]) for x in disjoint_other_tracks],
            **cluster_config,
            label=f"{other_label} {num_other_disjoints}",
        )
        plt.legend()
        plt.xlabel("number of clusters")
        plt.show()

        # good disjoint tracks
        tracks_matched_to_truth = reader.tracks_matched_to_truth
        disjoint_track_indices = np.array([x[0] for x in disjoint_tracks], dtype=int)
        good_disjoints = tracks_matched_to_truth[
            tracks_matched_to_truth.trkid.isin(disjoint_track_indices)
        ]
        num_good_disjoints = len(good_disjoints)

        # other disjoint tracks
        other_tracks_matched_to_truth = other_reader.tracks_matched_to_truth
        disjoint_other_track_indices = np.array([x[0] for x in disjoint_other_tracks], dtype=int)
        other_good_disjoints = other_tracks_matched_to_truth[
            other_tracks_matched_to_truth.trkid.isin(disjoint_other_track_indices)
        ]
        num_other_good_disjoints = len(other_good_disjoints)

        print(f"Number of good disjoint {label} tracks: {num_good_disjoints} / {num_disjoints}")
        print(
            f"Number of good disjoint {other_label} tracks: {num_other_good_disjoints} / {num_other_disjoints}"
        )

        # plot the number of clusters for the good disjoint tracks
        good_disjoints_tot_hits = good_disjoints.reco_pixel_hits + good_disjoints.reco_sct_hits
        other_good_disjoints_tot_hits = (
            other_good_disjoints.reco_pixel_hits + other_good_disjoints.reco_sct_hits
        )

        plt.title("Good Disjoint Tracks")
        plt.hist(
            good_disjoints_tot_hits,
            **cluster_config,
            label=f"{label} {num_good_disjoints}/{num_disjoints}",
        )
        plt.hist(
            other_good_disjoints_tot_hits,
            **cluster_config,
            label=f"{other_label} {num_other_good_disjoints}/{num_other_disjoints}",
        )
        plt.legend()
        plt.xlabel("Number of clusters")
        plt.show()

        # plot particle PT of the matched good disjoint tracks
        plt.title("Good Disjoint Tracks")
        particles = reader.particles

        matched_particles = particles[
            particles.particle_id.isin(good_disjoints.particle_id.values)
        ]
        other_matched_particles = particles[
            particles.particle_id.isin(other_good_disjoints.particle_id.values)
        ]

        pt = matched_particles.pt
        pt_other = other_matched_particles.pt
        num_high_pt = len(pt[pt > 1000])
        num_high_pt_other = len(pt_other[pt_other > 1000])

        pt_config = dict(bins=50, range=(0, 5000), alpha=0.5)
        plt.title("Good disjoint tracks")
        plt.hist(pt, label=f"{label} {len(pt)}", **pt_config)
        plt.text(2000, 40, f"{label} pT > 1 GeV: {num_high_pt:5}", fontsize=12)
        plt.text(2000, 30, f"{other_label} pT > 1 GeV: {num_high_pt_other:5}", fontsize=12)
        plt.hist(pt_other, label=f"CKF {len(pt_other)}", **pt_config)
        plt.legend()
        plt.xlabel("particle pT [MeV]")
        plt.show()

        merged = matched_particles.merge(good_disjoints, on="particle_id")
        merged_other = other_matched_particles.merge(other_good_disjoints, on="particle_id")
        plt.title("Good Disjoint Tracks")
        config = dict(s=10.0, alpha=0.5)
        plt.scatter(
            merged.pt,
            merged.reco_pixel_hits + merged.reco_sct_hits,
            label=f"{label}",
            **config,
        )
        plt.scatter(
            merged_other.pt,
            merged_other.reco_pixel_hits + merged_other.reco_sct_hits,
            label=f"{other_label}",
            **config,
        )
        plt.xlim(0, 5000)
        plt.ylim(0, 30)
        plt.xlabel("particle pT [MeV]")
        plt.ylabel("number of clusters")
        plt.legend()
        plt.show()

        return (
            disjoint_other_tracks,
            disjoint_other_tracks,
            good_disjoints,
            other_good_disjoints,
            matched_particles,
            other_matched_particles,
        )
