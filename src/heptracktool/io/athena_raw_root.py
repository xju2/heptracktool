from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import awkward
import numpy as np
import pandas as pd
import uproot

from heptracktool.io import utils_athena_raw as utils_raw_csv
from heptracktool.io import utils_athena_raw_root as utils_raw_root
from heptracktool.io.base import BaseTrackDataReader


class AthenaRawRootReader(BaseTrackDataReader):
    """Read Raw ROOT files created from RDO files with the dumping file.

    The code that creates the ROOT file can be found
    [link](https://gitlab.cern.ch/atlas/athena/-/tree/main/InnerDetector/InDetGNNTracking?ref_type=heads)
    """

    def __init__(
        self,
        inputdir,
        output_dir=None,
        overwrite=True,
        name="AthenaRawRootReader",
        debug=False,
    ):
        super().__init__(inputdir, output_dir, overwrite, name)
        self.debug = debug

        # find all files in inputdir
        root_files = sorted(self.inputdir.glob("*.root"))

        self.tree_name = "GNN4ITk"

        # create a map: event number -> file index and entry index
        self.event_map: dict[int, tuple[int, int]] = {}
        file_idx = 0
        self.root_files = []
        self.tot_evts = 0
        for filename in root_files:
            file_handle = uproot.open(filename)
            if self.tree_name not in file_handle:
                print(f"Tree {self.tree_name} is not in {filename}")
                continue

            events = uproot.open(f"{filename}:{self.tree_name}")
            event_numbers = events.arrays(["event_number"])["event_number"]
            num_entries = len(event_numbers)
            self.event_map.update(
                {
                    event_number: (file_idx, entry)
                    for event_number, entry in zip(event_numbers, range(num_entries))
                }
            )
            self.root_files.append(filename)
            self.tot_evts += num_entries
            file_idx += 1

        self.num_files = len(self.root_files)
        print(
            f"Directory: {self.inputdir} contains  {self.num_files} files and total {self.tot_evts} events."
        )

    def read_file(self, file_idx: int = 0, max_evts: int = -1) -> list[int]:
        """Read all events from the ROOT file. Each event is saved in parquet format.

        Args
        ----
            file_idx: int, index of the ROOT file to read in the input directory.
            max_evts: int, maximum number of events to read. If -1, read all events.

        Returns
        -------
            list of event numbers: list[int]. Return None if file_idx is out of range.
        """
        if file_idx >= self.num_files:
            print(
                f"File index {file_idx} is out of range. Max index is {self.num_files - 1}"
            )
            return None
        filename = self.root_files[file_idx]
        print(f"Reading file: {filename}")
        tree = uproot.open(f"{filename}:{self.tree_name}")

        existing_branches = set(tree.keys())
        requested_branches = set(utils_raw_root.all_branches)
        missing_branches = requested_branches - existing_branches
        if missing_branches:
            print(f"Missing branches: {missing_branches}")
            requested_branches -= missing_branches

        all_event_info = tree.arrays(requested_branches)
        num_events = len(all_event_info["event_number"])
        print(f"Number of events: {num_events} in file {filename}.")
        if max_evts > 0 and max_evts < num_events:
            print(f"Reading only {max_evts} events.")
            num_events = max_evts

        event_numbers = [
            self.process_one_event(all_event_info[evtid]) for evtid in range(num_events)
        ]
        return event_numbers

    def process_one_event(self, tracks_info: awkward.highlevel.Array) -> int:
        """Process one event and save the data in parquet format.

        Args
        ----
            tracks_info: awkward.highlevel.Array, the data of one event.

        Returns
        -------
            event_number: int, the event number of the processed event.
        """
        event_number = tracks_info["event_number"]

        self._read_particles(tracks_info)
        clusters = self._read_clusters(tracks_info)
        self._read_spacepoints(tracks_info, clusters)

        detailed_matching = self._read_detailed_matching(tracks_info)
        tracks = self._read_tracks(tracks_info)
        if detailed_matching is not None and tracks is not None:
            tracks = tracks.merge(
                detailed_matching,
                on=(
                    "trkid",
                    "subevent",
                    "barcode",
                ),
                how="left",
            )
            self._save("tracks", tracks, event_number)

        # read track contents
        sp_on_tracks = self._read_sp_on_tracks(tracks_info)
        cluster_on_tracks = self._read_cluster_on_tracks(tracks_info)

        if sp_on_tracks is not None and cluster_on_tracks is not None:
            cluster_on_tracks_array = np.array(
                [x for item in cluster_on_tracks for x in item], dtype=np.int64
            )
            sp_on_tracks["clusterIdxOnTrack"] = cluster_on_tracks_array
            self._save("trackcontents", sp_on_tracks, event_number)

        return event_number

    def _read_particles(self, tracks_info: awkward.highlevel.Array) -> pd.DataFrame:
        event_number = tracks_info["event_number"]

        if utils_raw_root.particle_branch_names[0] not in tracks_info.fields:
            return None

        # read particles
        particle_arrays = [tracks_info[x] for x in utils_raw_root.particle_branch_names]
        particles = pd.DataFrame(
            dict(zip(utils_raw_root.particle_col_names, particle_arrays))
        )
        # convert barcode to 7 digits
        particle_ids = utils_raw_csv.get_particle_ids(particles)
        particles.insert(0, "particle_id", particle_ids)
        particles = utils_raw_csv.particles_of_interest(particles)
        self._save("particles", particles, event_number)
        return particles

    def _read_clusters(self, tracks_info: awkward.highlevel.Array) -> pd.DataFrame:
        event_number = tracks_info["event_number"]

        if utils_raw_root.cluster_branch_names[0] not in tracks_info.fields:
            return None

        cluster_arrays = [
            tracks_info[x]
            for x in utils_raw_root.cluster_branch_names
            if x
            not in {"CLhardware", "CLparticleLink_barcode", "CLparticleLink_eventIndex"}
        ]
        # hardware is a std::vector, need special treatment
        cluster_hardware = np.array(tracks_info["CLhardware"].tolist(), dtype=str)
        cluster_columns = [*utils_raw_root.cluster_col_names, "hardware"]
        cluster_arrays.append(cluster_hardware)
        clusters = pd.DataFrame(dict(zip(cluster_columns, cluster_arrays)))

        clusters = clusters.astype({"hardware": "str", "barrel_endcap": "int32"})
        # read truth links for each cluster
        subevent_name, barcode_name = utils_raw_root.cluster_link_branch_names
        matched_subevents = tracks_info[subevent_name].tolist()
        matched_barcodes = tracks_info[barcode_name].tolist()
        max_matched = max(len(x) for x in matched_subevents)

        # loop over clusters matched particles
        matched_info = []
        for entry_idx in range(max_matched):
            matched_info += [
                (cluster_id, subevent[entry_idx], barcode[entry_idx])
                for cluster_id, subevent, barcode in zip(
                    clusters["cluster_id"].values,
                    matched_subevents,
                    matched_barcodes,
                )
                if len(subevent) > entry_idx
            ]
        cluster_matched = pd.DataFrame(
            matched_info, columns=["cluster_id", "subevent", "barcode"]
        )
        cluster_matched["particle_id"] = utils_raw_csv.get_particle_ids(cluster_matched)
        clusters = clusters.merge(cluster_matched, on="cluster_id", how="left")

        self._save("clusters", clusters, event_number)
        return clusters

    def _read_spacepoints(
        self, tracks_info: awkward.highlevel.Array, clusters: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        event_number = tracks_info["event_number"]

        if utils_raw_root.spacepoint_branch_names[0] not in tracks_info.fields:
            return None

        spacepoint_arrays = [
            tracks_info[x] for x in utils_raw_root.spacepoint_branch_names
        ]
        spacepoints = pd.DataFrame(
            dict(zip(utils_raw_root.spacepoint_col_names, spacepoint_arrays))
        )
        self._save("spacepoints", spacepoints, event_number)

        pixel_hits = spacepoints[spacepoints["cluster_index_2"] == -1]
        strip_hits = spacepoints[spacepoints["cluster_index_2"] != -1]

        if clusters is not None:
            # matching spacepoints to particles through clusters
            truth = utils_raw_csv.truth_match_clusters(pixel_hits, strip_hits, clusters)
            truth = utils_raw_csv.merge_spacepoints_clusters(truth, clusters)
        else:
            truth = spacepoints
        self._save("truth", truth, event_number)
        return truth

    def _read_detailed_matching(
        self, tracks_info: awkward.highlevel.Array
    ) -> pd.DataFrame:
        """Read detailed truth tracks (dtt).

        A reoc track may be matched to multiple truth tracks.
        Detailed truth tracks contains all the truth tracks that are matched to a reco track.
        but we don't need all the info. Instead, we will only keep the most probable truth track.

        Args
        ----
            tracks_info: awkward.highlevel.Array, the data of one event.

        Returns
        -------
            detailed_matching: pd.DataFrame, the detailed truth track info.
        """
        event_number = tracks_info["event_number"]

        if utils_raw_root.detailed_truth_branch_names[0] not in tracks_info.fields:
            return None

        dtt_arrays = [
            tracks_info[x].to_numpy()
            for x in utils_raw_root.detailed_truth_branch_names
            if "trajectory" not in x
        ]
        dtt_particles = [
            tracks_info[x].to_list()
            for x in utils_raw_root.detailed_truth_branch_names
            if "trajectory" in x
        ]
        # only keep the first matched particle info.
        dtt_particle_arrays = [[min(x) for x in pp] for pp in dtt_particles]
        detailed_matching = pd.DataFrame(
            np.array(
                [
                    dtt_arrays[0],
                    dtt_arrays[1],
                    dtt_particle_arrays[0],
                    dtt_particle_arrays[1],
                    dtt_arrays[2][:, 0],
                    dtt_arrays[2][:, 1],
                    dtt_arrays[3][:, 0],
                    dtt_arrays[3][:, 1],
                    dtt_arrays[4][:, 0],
                    dtt_arrays[4][:, 1],
                ]
            ).T,
            columns=[
                "trkid",
                "num_matched",
                "subevent",
                "barcode",
                "true_pixel_hits",
                "true_sct_hits",
                "reco_pixel_hits",
                "reco_sct_hits",
                "common_pixel_hits",
                "common_sct_hits",
            ],
        )
        self._save("detailed_matching", detailed_matching, event_number)
        return detailed_matching

    def _read_tracks(self, tracks_info: awkward.highlevel.Array) -> pd.DataFrame:
        event_number = tracks_info["event_number"]

        if utils_raw_root.reco_track_branch_names[0] not in tracks_info.fields:
            return None

        track_arrays = [tracks_info[x] for x in utils_raw_root.reco_track_branch_names]
        track_info = pd.DataFrame(
            dict(zip(utils_raw_root.reco_track_col_names, track_arrays))
        )
        track_info = track_info.astype(utils_raw_root.reco_track_col_types)
        self._save("tracks", track_info, event_number)
        return track_info

    def _read_sp_on_tracks(self, tracks_info: awkward.highlevel.Array) -> pd.DataFrame:
        event_number = tracks_info["event_number"]

        if utils_raw_root.reco_track_sp_branch_names[0] not in tracks_info.fields:
            if self.debug:
                print("No track content info (space points).")
            return None

        sp_on_track_arrays = [
            tracks_info[x] for x in utils_raw_root.reco_track_sp_branch_names
        ]
        sp_on_track_info = pd.DataFrame(
            dict(zip(utils_raw_root.reco_track_sp_col_names, sp_on_track_arrays))
        )
        sp_on_track_info = sp_on_track_info.astype(
            utils_raw_root.reco_track_sp_col_types
        )

        # groups = sp_on_track_info.groupby("trkid")
        # # for each group (track), we put the space point indices to a list without sorting.
        # # and return a list of lists.
        # tracking_contents = groups["spIdxOnTrack"].apply(lambda x: x.tolist()).tolist()

        self._save("sps_on_track", sp_on_track_info, event_number)
        return sp_on_track_info

    def _read_cluster_on_tracks(
        self, tracks_info: awkward.highlevel.Array
    ) -> list[list[int]]:
        event_number = tracks_info["event_number"]
        branch_name = "TRKmeasurementsOnTrack_pixcl_sctcl_index"
        if branch_name not in tracks_info.fields:
            if self.debug:
                print("No track content info (clusters).")
            return None

        cluster_on_track_list = tracks_info[branch_name]
        self._save("clusters_on_track", cluster_on_track_list, event_number)
        return cluster_on_track_list

    def _save(self, outname: str, df: Any, evtid: int) -> bool:
        outname = self.get_outname(outname, evtid)
        if outname.exists() and not self.overwrite:
            return True
        if df is not None:
            if isinstance(df, pd.DataFrame):
                df.to_parquet(outname, compression="gzip")
            else:
                # use pickle for the rest data objects
                outname = outname.with_suffix(".pkl")
                with open(outname, "wb") as f:
                    pickle.dump(df, f)
        return False

    def _read(self, outname: str, evtid: int) -> pd.DataFrame:
        outname = self.get_outname(outname, evtid)
        if outname.exists():
            return pd.read_parquet(outname)

        if outname.with_suffix(".pkl").exists():
            with open(outname.with_suffix(".pkl"), "rb") as f:
                return pickle.load(f)  # noqa: S301
        return None

    def get_outname(self, outname: str, evtid: int) -> Path:
        return self.outdir / f"event{evtid:06d}-{outname}.parquet"

    def read(self, evtid: int = 0) -> bool:
        self.clusters = self._read("clusters", evtid)
        self.particles = self._read("particles", evtid)
        self.spacepoints = self._read("spacepoints", evtid)
        self.truth = self._read("truth", evtid)
        self.detailed_matching = self._read("detailed_matching", evtid)
        self.tracks = self._read("tracks", evtid)
        self.sps_on_track = self._read("sps_on_track", evtid)
        self.clusters_on_track = self._read("clusters_on_track", evtid)
        self.track_contents = self._read("trackcontents", evtid)

        return all(
            [
                self.clusters is not None,
                self.particles is not None,
                self.spacepoints is not None,
                self.truth is not None,
                self.detailed_matching is not None,
                self.tracks is not None,
                self.sps_on_track is not None,
                self.clusters_on_track is not None,
            ]
        )

    def get_event_info(self, file_idx: int = 0) -> pd.DataFrame:
        if file_idx >= self.num_files:
            print(
                f"File index {file_idx} is out of range. Max index is {self.num_files - 1}"
            )
            return None

        filename = self.root_files[file_idx]
        print("reading event info from", filename)
        with uproot.open(filename) as f:
            tree = f[self.tree_name]
            event_info = tree.arrays(utils_raw_root.event_branch_names, library="pd")
            return event_info

    def find_event(self, event_numbers: list[int]):
        # we loop over all availabel root files
        # check if the requrested event number is in the file
        # if yes, we write down the file name and the event number
        # if no, we continue to the next file

        # event_number_map: Dict[int, str] = dict([(x, "") for x in event_numbers])
        event_number_map: dict[int, str] = {}

        for root_file_idx in range(len(self.root_files)):
            event_info = self.get_event_info(root_file_idx)
            for event_number in event_numbers:
                if event_number in event_info["event_number"].to_numpy():
                    print(
                        f"Event {event_number} is in file {self.root_files[root_file_idx]}"
                    )
                    event_number_map[event_number] = self.root_files[root_file_idx]

                    self.read_file(root_file_idx)
                    self.truth.to_csv(f"event{event_number:09d}-truth.csv")

        return event_number_map

    def match_to_truth(self) -> pd.DataFrame:
        """Match a reco track to a truth track
        only if all reco track contents are from the truth track.
        """
        detailed = self.detailed_matching
        all_matched_to_truth = detailed[
            (detailed.reco_pixel_hits == detailed.common_pixel_hits)
            & (detailed.reco_sct_hits == detailed.common_sct_hits)
        ]
        num_all_matched_to_truth = len(all_matched_to_truth)

        frac_all_matched_to_truth = num_all_matched_to_truth / len(detailed)
        print(
            f"All matched to truth {self.name}: ",
            num_all_matched_to_truth,
            len(detailed),
            frac_all_matched_to_truth,
        )

        # Check that each reco track is matched to only one truth track
        _, counts = np.unique(all_matched_to_truth.trkid.values, return_counts=True)
        assert np.all(
            counts == 1
        ), "Some reco tracks are matched to multiple truth tracks"

        self.tracks_matched_to_truth = all_matched_to_truth

        return all_matched_to_truth
