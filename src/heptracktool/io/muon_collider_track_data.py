import uproot.exceptions
from heptracktool.io.base import BaseTrackDataReader
import uproot
import pandas as pd
from loguru import logger

import torch

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

        logger.info(f"Total {self.nevts} events in directory: {self.inputdir}")

    def read(self, evtid: int = 0):
        if (evtid is None or evtid < 1) and self.nevts > 0:
            evtid = self.all_evtids[0]
            filename = self.all_files[0]
            print(f"read event {evtid}, using file {filename}")
        elif evtid not in self.all_evtids:
            raise ValueError(f"Event id {evtid} not found in the input directory.")
        else:
            filename = self.all_files[evtid]

        with uproot.open(filename) as file_handle:  # type: ignore
            try:
                tree = file_handle["HitTree"]
            except uproot.exceptions.KeyInFileError:
                logger.error(f"HitTree not found in {evtid} file: {filename}")
                return (None, None)

            event_info = tree.arrays(list(translator.keys()), library="np")  # type: ignore

            hit_arrays = [event_info[x][0] for x in hit_branch_names]
            hits = pd.DataFrame(dict(zip(hit_col_names, hit_arrays)))

            particle_arrays = [event_info[x][0] for x in particle_branch_names]
            particles = pd.DataFrame(dict(zip(particle_col_names, particle_arrays)))

            # ! the default particle ID is 0, which is not a good practice.
            # ! as zeros are often reserved for noise hits. Let's use 123.
            # TODO: use the particle ID from ACTS.
            muon_particle_id = 123
            particles["part_id"] = muon_particle_id
            hits["particle_id"] = 0
            hits.loc[hits["is_from_secondary"] == 0.0, "particle_id"] = muon_particle_id

        return hits, particles

    def save_one_event(self, evt_id: int, spacepoints: pd.DataFrame, particles: pd.DataFrame):
        """Save one event to the output directory as a pyG file."""
        if spacepoints is None or particles is None:
            logger.error(f"No data to save for {evt_id}.")

        data_dict = [(name, spacepoints[name].to_numpy()) for name in hit_col_names + ["particle_id"]]
        data_dict += [(name, particles[name].to_numpy()) for name in particle_col_names]
        data_dict += [("event_id", evt_id)]
        output_file = self.outdir / f"event_{evt_id:09d}.pt"
        torch.save(dict(data_dict), output_file)

    def read_and_save_one_evt(self, evt_id):
        hits, particles = self.read(evt_id)
        if hits is not None and particles is not None:
            self.save_one_event(evt_id, hits, particles)

    def save_all_events(self, num_workers: int = 1):
        """Save all events to the output directory."""
        logger.info(f"Saving all {len(self.all_evtids)} events to {self.outdir} with {num_workers} workers.")
        from tqdm import tqdm
        description = "Saving events"
        if num_workers < 2:
            for evt_id in tqdm(self.all_evtids, desc=description):
                self.read_and_save_one_evt(evt_id)
        else:
            from tqdm.contrib.concurrent import process_map
            process_map(self.read_and_save_one_evt, self.all_evtids, desc=description, max_workers=num_workers)

        logger.info("Done saving all events.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Muon Collider Track Data Reader")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Input directory containing the data files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory to save the processed data.")
    parser.add_argument("-r", "--overwrite", action="store_true", help="Overwrite existing files in the output directory.")
    parser.add_argument("-w", "--num_workers", type=int, default=1, help="Number of workers for parallel processing.")
    args = parser.parse_args()

    reader = MuonColliderTrackDataReader(args.input_dir, args.output_dir, args.overwrite)
    reader.save_all_events(num_workers=args.num_workers)


if __name__ == "__main__":
    main()
