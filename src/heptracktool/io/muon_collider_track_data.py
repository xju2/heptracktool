import re

import numpy as np
import pandas as pd
import torch
import uproot
import uproot.exceptions
from loguru import logger
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from heptracktool.io.base import BaseTrackDataReader
from heptracktool.io.utils_mcollider_data import (
    cell_branch_names,
    cell_col_names,
    extract_cell_features,
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

        file_name_pattern = "Hits_TTree_([0-9]+)_([0-9]+)-([0-9]+).root"
        self.all_evtids = [
            int(re.search(file_name_pattern, x.name).group(1).strip())  # type: ignore
            for x in all_evts  # type: ignore
        ]

        # sort the event ids and file names
        arg_sorted = sorted(range(len(self.all_evtids)), key=lambda k: self.all_evtids[k])
        self.all_evtids = [self.all_evtids[i] for i in arg_sorted]
        self.all_files = [all_evts[i] for i in arg_sorted]
        self.nevts = len(self.all_evtids)

        self.hit_pid_name = "hit_particle_id"

        logger.info(f"Total {self.nevts} events in directory: {self.inputdir}")

    def read(self, evtid: int = 0) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        assert isinstance(evtid, int), f"Event ID must be an integer. {type(evtid)} given."
        assert 0 <= evtid < self.nevts, f"Event ID {evtid} out of range [0, {self.nevts})"
        filename = self.all_files[evtid]
        logger.debug("Reading event index {} from file {}", evtid, filename)

        with uproot.open(filename) as file_handle:  # type: ignore
            try:
                tree = file_handle["HitTree"]
            except uproot.exceptions.KeyInFileError:
                logger.error(f"HitTree not found in {evtid} file: {filename}")
                return (None, None)

            event_info = tree.arrays(list(translator.keys()), library="np")  # type: ignore

            # check if the event size is zero.
            hit_x = hit_branch_names[0]
            if len(event_info[hit_x]) < 1 or event_info[hit_x][0].size < 1:
                logger.warning(f"Event {evtid} has no hits in file: {filename}")
                return (None, None)

            hit_arrays = [event_info[x][0] for x in hit_branch_names]
            hits = pd.DataFrame(dict(zip(hit_col_names, hit_arrays)))

            particle_arrays = [event_info[x][0] for x in particle_branch_names]
            particles = pd.DataFrame(dict(zip(particle_col_names, particle_arrays)))

            # ! the default particle ID is 0, which is not a good practice.
            # ! as zeros are often reserved for noise hits. Let's use 123.
            # TODO: use the particle ID from ACTS.
            muon_particle_id = 123
            particles["part_id"] = muon_particle_id
            hits[self.hit_pid_name] = 0
            hits.loc[hits["hit_is_from_secondary"] == 0.0, self.hit_pid_name] = muon_particle_id

            # process cell information. Not yet ready...
            # these will be additional features associated with each space point (hit)
            cell_arrays = [event_info[x][0] for x in cell_branch_names]
            cells = pd.DataFrame(dict(zip(cell_col_names, cell_arrays)))
            num_cells = hits["hit_charge_count"].to_numpy()
            cell_hit_id = np.repeat(np.arange(len(num_cells)), num_cells)
            cells["hit_id"] = cell_hit_id
            cell_features = extract_cell_features(cells)
            hits = pd.concat([hits, cell_features], axis=1)

        return hits, particles

    def event_filename(self, evt_id: int):
        """Generate the output filename for a given event ID."""
        return self.outdir / f"event_{evt_id:09d}.pt"

    def save_one_event(self, evt_id: int, spacepoints: pd.DataFrame, particles: pd.DataFrame):
        """Save one event to the output directory as a pyG file."""
        if spacepoints is None or particles is None:
            logger.error(f"No data to save for {evt_id}.")

        data_dict = [
            (name, spacepoints[name].to_numpy()) for name in hit_col_names + [self.hit_pid_name]
        ]
        data_dict += [(name, particles[name].to_numpy()) for name in particle_col_names]
        evt_num = self.all_evtids[evt_id]
        data_dict += [("event_id", evt_num)]
        torch.save(dict(data_dict), self.event_filename(evt_num))

    def read_and_save_one_evt(self, evt_id):
        if self.event_filename(evt_id).exists() and not self.overwrite:
            return

        hits, particles = self.read(evt_id)
        if hits is not None and particles is not None:
            self.save_one_event(evt_id, hits, particles)

    def save_all_events(self, num_workers: int = 1, max_evts: int = -1):
        """Save all events to the output directory."""
        max_evts = self.nevts if max_evts < 0 else min(max_evts, self.nevts)
        all_evtids = list(range(max_evts))
        logger.info(f"Saving all {max_evts} events to {self.outdir} with {num_workers} workers.")
        description = "Saving events"
        if num_workers < 2:
            for evt_id in tqdm(all_evtids, desc=description):
                self.read_and_save_one_evt(evt_id)
        else:
            process_map(
                self.read_and_save_one_evt,
                all_evtids,
                desc=description,
                max_workers=num_workers,
                chunksize=1,
            )

        logger.info("Done saving all events.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Muon Collider Track Data Reader")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing the data files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the processed data.",
    )
    parser.add_argument(
        "-r",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for parallel processing.",
    )
    args = parser.parse_args()

    reader = MuonColliderTrackDataReader(args.input_dir, args.output_dir, args.overwrite)
    reader.save_all_events(num_workers=args.num_workers)


if __name__ == "__main__":
    main()
