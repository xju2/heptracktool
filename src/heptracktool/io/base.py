from __future__ import annotations
from typing import Union
from pathlib import Path
import pandas as pd
from loguru import logger


class BaseTrackDataReader:
    """Base class for reading Tracking data.

    It reads and processes the data. The processed data is stored in the optional directory."""

    def __init__(
        self,
        inputdir: Union[str, Path],
        output_dir: str | None = None,
        overwrite: bool = True,
        name="BaseTrackDataReader",
    ):
        self.inputdir = Path(inputdir) if isinstance(inputdir, str) else inputdir
        if not self.inputdir.exists() or not self.inputdir.is_dir():
            raise FileNotFoundError(
                f"Input directory {self.inputdir} does not exist or is not a directory."
            )

        self.outdir = Path(output_dir) if output_dir else self.inputdir / "processed_data"
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.name = name
        self.overwrite = overwrite

        # file systems and basic event information
        self.all_evtids: list[int] = []
        self.nevts = 0

        # following are essential dataframe
        self.particles: pd.DataFrame = None
        self.clusters: pd.DataFrame = None
        self.spacepoints: pd.DataFrame = None
        # truth is the same as spacepoints, but contains truth information
        self.truth: pd.DataFrame = None
        self.true_edges: pd.DataFrame = None

        # following are optional dataframe
        # they are created from the dumping object from Athena
        # needed for truth studies
        self.tracks_clusters: pd.DataFrame | None = None
        self.tracks: pd.DataFrame | None = None
        self.true_tracks: pd.DataFrame | None = None
        self.detailed_matching: pd.DataFrame | None = None
        self.tracks_matched_to_truth: pd.DataFrame | None = None

    def read(self, evtid: int = 0) -> bool:
        """Read one event from the input directory.

        Args:
            evtid: event id to read. Default is 0.

        Returns:
            bool: True if read successfully, False otherwise.
        """
        raise NotImplementedError

    def read_by_event_number(self, evt_number: int) -> bool:
        """Read one event from the input directory by event number.

        Args:
            evt_number: event number to read.

        Returns:
            bool: True if read successfully, False otherwise.
        """
        evtid = -1
        if self.all_evtids:
            try:
                evtid = self.all_evtids.index(evt_number)
            except ValueError:
                evtid = -1
        if evtid < 0:
            logger.error(f"Event number {evt_number} not found in the event list.")
            return False
        return self.read(evtid)

    def __str__(self):
        return f"{self.name} reads from {self.inputdir}."
