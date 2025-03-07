"""AthenaRawDataReader is to read the data dumped by the DumpObjects algorithm in Athena.
See: https://gitlab.cern.ch/xju/athena/-/tree/xju/exatrkx-rel21.9/Tracking/TrkDumpAlgs.

The dumped data includes:
* Data Information
    * spacepoints_*
    * clusters_*
    * trackcandidates_*
        * track candidates are represented as indices of spacepoints
    * trackcandidates_clusters_*
        * track candidates are represented as indices of clusters

* Truth Information
    * particles
    * detailedtracktruth -> each reconstructed track may be matched to more than one truth track
    * tracks_evt -> each reconstructed track is matched to one truth track
    * subevents -> event index used in each event to indicate if the event is from hard process.

They are add the same postfix: "{}_evt{index}-{run_number}_{event_number}.txt"

"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from heptracktool.io import utils as io_utils
from heptracktool.io import utils_athena_raw as reader_utils
from heptracktool.io.base import BaseTrackDataReader


class AthenaRawDataReader(BaseTrackDataReader):
    def __init__(
        self, inputdir, output_dir=None, overwrite=False, name="AthenaRawDataReader"
    ):
        super().__init__(inputdir, output_dir, overwrite, name)

        # find number of events in the directory
        # and extract the event id, run number, and event number.
        all_evts = list(self.inputdir.glob("spacepoints_*.txt"))
        pattern = "spacepoints_evt([0-9]*)-([0-9]*)_([0-9]*).*.txt"
        regrex = re.compile(pattern)

        def find_evt_info(x):
            matched = regrex.search(x.name)
            if matched is None:
                return None
            evtid = int(matched.group(1).strip())
            run_number = int(matched.group(2).strip())
            event_number = int(matched.group(3).strip())
            return (evtid, run_number, event_number)

        self.all_evtids = [find_evt_info(x) for x in all_evts]
        self.all_evtids = [x for x in self.all_evtids if x is not None]
        self.nevts = len(self.all_evtids)
        print(f"Total {self.nevts} events in directory: {self.inputdir}")

    def getnamepatch(self, evtid, run_number, event_number):
        """Get the postfix for the given event id or event number."""
        if evtid is None and run_number is None and event_number is None:
            event_info = self.all_evtids[0]
        elif event_number is not None:
            event_info = [x for x in self.all_evtids if x[2] == event_number]
            if len(event_info) == 0:
                raise ValueError(
                    f"Cannot find the event with event number: {event_number}"
                )
            event_info = event_info[0]
        else:
            raise ValueError("Please provide either evtid or event_number or nothing")

        evtid, run_number, event_number = event_info
        return "_evt{}-{}_{}".format(evtid, run_number, event_number)

    def get_filename(
        self, prefix, evtid, run_number, event_number, suffix="txt"
    ) -> str:
        """Get the input filename for the given event id or event number."""
        namepatch = self.getnamepatch(evtid, run_number, event_number)
        return self.inputdir / "{}{}.{}".format(prefix, namepatch, suffix)

    def get_outname(
        self, prefix, evtid, run_number, event_number, suffix="parquet", **kwargs
    ) -> str:
        """Get the filename for the given event id or event number."""
        namepatch = self.getnamepatch(evtid, run_number, event_number)
        return self.outdir / "{}{}.{}".format(prefix, namepatch, suffix)

    def read_wrap(
        self,
        read_fn: Callable[[str], pd.DataFrame | Any],
        prefix,
        evtid=None,
        run_number=None,
        event_number=None,
        **kwargs,
    ):
        """Read the data from the input directory."""
        outname = Path(
            self.get_outname(prefix, evtid, run_number, event_number, **kwargs)
        )
        if outname.exists() and not self.overwrite:
            return io_utils.load_data(outname)

        filename = self.get_filename(prefix, evtid, run_number, event_number)
        df = read_fn(filename)
        io_utils.save_data(df, outname)
        return df

    def read_track_candidates(
        self, evtid=None, run_number=None, event_number=None
    ) -> list[list[int]]:
        """Read track candidates from the input directory.

        Return
        ------
            List[List[int]]: each element is a list of spacepoint indices
        """
        self.tracks = self.read_wrap(
            reader_utils.read_track_candidates,
            "trackcandidates",
            evtid,
            run_number,
            event_number,
            suffix="pkl",
        )
        return self.tracks

    def read_track_candidates_clusters(
        self, evtid=None, run_number=None, event_number=None
    ) -> list[list[int]]:
        """Read track candidates from the input directory.

        Return
        ------
            list[list[int]]: each element is a list of cluster indices
        """
        self.clusters_on_track = self.read_wrap(
            reader_utils.read_track_candidates,
            "trackcandidates_clusters",
            evtid,
            run_number,
            event_number,
            suffix="pkl",
        )
        return self.clusters_on_track

    def read_spacepoints(
        self, evtid=None, run_number=None, event_number=None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read spacepoints from the input directory.

        Return
        ------
            pd.DataFrame: spacepoints
        """
        self.spacepoints = self.read_wrap(
            reader_utils.read_spacepoints,
            "spacepoints",
            evtid,
            run_number,
            event_number,
        )
        return self.spacepoints

    def read_clusters(
        self, evtid=None, run_number=None, event_number=None
    ) -> pd.DataFrame:
        """Read clusters from the input directory.

        Return
        ------
            pd.DataFrame: clusters
        """
        filename = self.get_filename("clusters", evtid, run_number, event_number)
        cell_split_df = pd.read_csv(
            filename, header=None, engine="python", sep=r",#,|#,|,#"
        )

        cell_df = pd.DataFrame()

        # First read the co-ordinates of each cluster
        cell_df[
            ["cluster_id", "hardware", "cluster_x", "cluster_y", "cluster_z"]
        ] = cell_split_df[0].str.split(",", expand=True)
        cell_df = cell_df.astype(
            {
                "cluster_id": "int32",
                "cluster_x": "float32",
                "cluster_y": "float32",
                "cluster_z": "float32",
            }
        )

        # Split the detector geometry information
        cell_df[
            ["barrel_endcap", "layer_disk", "eta_module", "phi_module", "side"]
        ] = cell_split_df[1].str.split(",", expand=True)[[0, 1, 2, 3, 4]]

        # true particle info
        cell_df["particle_id"] = cell_split_df[2]

        cell_shape_info = cell_split_df[4].str.split(",", expand=True).astype("float32")
        if cell_shape_info.shape[1] == 2:
            cell_df[["eta_angle", "phi_angle"]] = cell_shape_info[[0, 1]]
        elif cell_shape_info.shape[1] == 11:
            shape_feature_names = [
                "cell_count",
                "cell_val",
                "leta",
                "lphi",
                "lx",
                "ly",
                "lz",
                "geta",
                "gphi",
                "eta_angle",
                "phi_angle",
            ]
            cell_df[shape_feature_names] = cell_shape_info[
                list(range(len(shape_feature_names)))
            ]
        else:
            raise RuntimeError("Unknown cluster features", cell_shape_info.shape)

        cell_df[["norm_x", "norm_y", "norm_z"]] = cell_split_df[5].str.split(
            ",", expand=True
        )[[0, 1, 2]]

        if not hasattr(self, "particles") or self.particles is None:
            self.read_particles(evtid, run_number, event_number)

        # Do some fiddling to split the cluster entry for truth particle,
        # which could have 0 true particles, 1 true particle, or many true particles
        cleaned_cell_pids = cell_df[["cluster_id", "particle_id"]].astype(
            {"particle_id": str}
        )

        split_pids = pd.DataFrame(
            [
                [c, p]
                for c, P in cleaned_cell_pids.itertuples(index=False)
                for p in P.split("),(")
            ],
            columns=cleaned_cell_pids.columns,
        )
        split_pids = split_pids.join(
            split_pids.particle_id.str.strip("()")
            .str.split(",", expand=True)
            .rename({0: "subevent", 1: "barcode"}, axis=1)
        ).drop(columns=["particle_id", 2])

        split_pids["particle_id"] = (
            split_pids.merge(
                self.particles[["subevent", "barcode", "particle_id"]].astype(
                    {"subevent": str, "barcode": str}
                ),
                how="left",
                on=["subevent", "barcode"],
            )["particle_id"]
            .fillna(0)
            .astype(int)
        )

        split_pids = split_pids[["cluster_id", "particle_id"]]

        # Fix some types
        cell_df = cell_df.drop(columns=["particle_id"])
        cell_df = cell_df.merge(split_pids, on="cluster_id").astype({"cluster_id": int})

        # Fix indexing mismatch in DumpObjects
        self.clusters = cell_df
        return cell_df

    def read_particles(
        self, evtid=None, run_number=None, event_number=None
    ) -> pd.DataFrame:
        """Read particles from the input directory.

        Return
        ------
            pd.DataFrame: particles
        """
        self.particles = self.read_wrap(
            reader_utils.read_particles, "particles", evtid, run_number, event_number
        )
        return self.particles

    def read_true_track(self, evtid=None, run_number=None, event_number=None):
        """Read true track from the input directory"""
        self.true_tracks = self.read_wrap(
            reader_utils.read_true_track, "tracks", evtid, run_number, event_number
        )
        return self.true_tracks

    def read_detailed_matching(
        self, evtid=None, run_number=None, event_number=None
    ) -> pd.DataFrame:
        """Read Detailed True Track Collection from the input directory.

        Return
        ------
            pd.DataFrame: detailed matching
        """
        self.detailed_matching = self.read_wrap(
            reader_utils.read_detailed_matching,
            "detailedtracktruth",
            evtid,
            run_number,
            event_number,
        )
        return self.detailed_matching

    def _read(self, evtid=None, run_number=None, event_number=None) -> None:
        """Read all the data from the input directory."""
        info = (evtid, run_number, event_number)
        self.read_spacepoints(*info)
        self.read_particles(*info)
        self.read_clusters(*info)
        self.read_track_candidates(*info)
        self.read_track_candidates_clusters(*info)
        self.read_true_track(*info)
        self.read_detailed_matching(*info)

    def read(self, evtid: int = 0) -> bool:
        if evtid < 0 or evtid >= self.nevts:
            return False
        self._read(*(self.all_evtids[evtid]))
        return True

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

    def __str__(self) -> str:
        return f"{self.name} reads from {self.inputdir}."


if __name__ == "__main__":
    basedir = "/media/DataOcean/projects/tracking/integrateToAthena/run_21.9.26/RunOneEventForDebuging/GNN_noRemoval"
    reader = AthenaRawDataReader(basedir)
    print(reader)
    print(reader.all_evtids)
    print(reader.read())
