from heptracktool.io.base import BaseTrackDataReader
import uproot
import pandas as pd

import re

from heptracktool.io.utils_mcollider_data import (
    hit_branch_names,
    hit_col_names,
    particle_branch_names,
    particle_col_names,
    translator,
)


class MuonColliderTrackDataReader(BaseTrackDataReader):
    """Reader for Muon Collider track data."""

    def __init__(self, input_dir, output_dir, overwrite):
        super().__init__(input_dir, output_dir, overwrite, name="MuonColliderTrackDataReader")

        all_evts = list(self.inputdir.glob("*.root"))
        self.nevts = len(all_evts)

        file_name_pattern = "Hits_TTree_([0-9]+)_([0-9]+)-([0-9]+).root"
        self.all_evtids = [
            int(re.search(file_name_pattern, x.name).group(1).strip())
            for x in all_evts  # type: ignore
        ]

        # sort the event ids and file names
        arg_sorted = sorted(range(len(self.all_evtids)), key=lambda k: self.all_evtids[k])
        self.all_evtids = [self.all_evtids[i] for i in arg_sorted]
        self.all_files = [all_evts[i] for i in arg_sorted]

        print(f"Total {self.nevts} events in directory: {self.inputdir}")

    def read(self, evtid: int = 0):
        if (evtid is None or evtid < 1) and self.nevts > 0:
            evtid = self.all_evtids[0]
            filename = self.all_files[0]
            print(f"read event {evtid}")
        elif evtid not in self.all_evtids:
            raise ValueError(f"Event id {evtid} not found in the input directory.")
        else:
            filename = self.all_files[evtid]

        with uproot.open(filename) as file_handle:  # type: ignore
            tree = file_handle["HitTree"]
            event_info = tree.arrays(list(translator.keys()), library="np")  # type: ignore

            hit_arrays = [event_info[x] for x in hit_branch_names]
            hits = pd.DataFrame(dict(zip(hit_col_names, hit_arrays)))

            particle_arrays = [event_info[x] for x in particle_branch_names]
            particles = pd.DataFrame(dict(zip(particle_col_names, particle_arrays)))

            self.spacepoints = hits
            self.particles = particles

        return True

    def save_one_event(self, evt_id: int):
        """Save one event to the output directory as a pyG file."""
        if self.spacepoints is None or self.particles is None:
            raise ValueError("No data to save. Please read the data first.")

    def read_and_save_one_evt(self, evt_id):
        self.read(evt_id)
        self.save_one_event(evt_id)
