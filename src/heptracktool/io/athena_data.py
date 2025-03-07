"""Read dataframes obtained by preprocessing text files dumped from Athena"""

import os
import re
from typing import Any
import numpy as np
import pandas as pd
import itertools

from heptracktool.io.base import BaseTrackDataReader


def true_edges(hits):
    hit_list = (
        hits.groupby(
            [
                "particle_id",
                "hardware",
                "barrel_endcap",
                "layer_disk",
                "eta_module",
                "phi_module",
            ],
            sort=False,
        )["index"]
        .agg(lambda x: list(x))
        .groupby(level=0)
        .agg(lambda x: list(x))
    )

    e = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            e.extend(list(itertools.product(i, j)))

    layerless_true_edges = np.array(e).T
    return layerless_true_edges


class AthenaDFReader(BaseTrackDataReader):
    """Athena dataframe reader"""

    def __init__(
        self,
        inputdir: str,
        output_dir: str | None = None,
        overwrite: bool = False,
        name: str = "AthenaDFReader",
        postfix: str = "csv",
        selections: bool = False,
        pt_cut: float = 0.3,
    ):
        super().__init__(inputdir, output_dir, overwrite, name)

        self.postfix = postfix

        all_evts = self.inputdir.glob(f"event*-truth.{postfix}")
        self.nevts = len(all_evts)
        pattern = f"event([0-9]*)-truth.{postfix}"
        self.all_evtids = sorted(
            [
                int(re.search(pattern, os.path.basename(x)).group(1).strip())
                for x in all_evts
            ]
        )
        print(f"total {self.nevts} events in directory: {self.csv_dir}")

        self.selections = selections
        if self.selections:
            print("Only select PIXEL Barrel.")
        self.ptcut = max(pt_cut, 0)
        print(f"True particles with pT > {self.ptcut} GeV")

    def read_file(self, prefix: str) -> pd.DataFrame:
        filename = f"{prefix}.{self.postfix}"

        if self.postfix == "csv":
            return pd.read_csv(filename)
        elif self.postfix == "pkl":
            return pd.read_pickle(filename)
        else:
            raise ValueError(f"Unknown postfix: {self.postfix}")

    def read(self, evtid: int) -> bool:
        """Read one event from the input directory

        Args:
            evtid (int, optional): Event ID to read. Defaults to None.

        Returns:
            bool: True if reading is successful, False otherwise.
        """
        if (evtid is None or evtid < 1) and self.nevts > 0:
            evtid = self.all_evtids[0]

        prefix = os.path.join(self.csv_dir, f"event{evtid:09d}")
        truth_fname = f"{prefix}-truth"
        truth = self.read_file(truth_fname)

        r = np.sqrt(truth.x**2 + truth.y**2)
        phi = np.arctan2(truth.y, truth.x)
        truth = truth.assign(r=r, phi=phi)

        # place selections if you want...
        if self.selections:
            pixel_only = truth.hardware == "PIXEL"
            barrel = truth.barrel_endcap == 0
            truth = truth[barrel & pixel_only]

        truth.drop_duplicates(
            subset=["hit_id", "x", "y", "z"], inplace=True, keep="first"
        )
        truth = truth.reset_index(drop=True).reset_index(drop=False)

        particle_fname = f"{prefix}-particles"
        particles = self.read_file(particle_fname)
        particles["pt"] = particles.pt.values / 1000.0  # to be GeV

        truth = truth.merge(particles, on="particle_id", how="left")

        # true edges for particles with pT > 0.3 GeV
        good_hits = truth[(truth.particle_id != 0) & (truth.pt > self.ptcut)]
        good_hits = good_hits.assign(
            R=np.sqrt(
                (good_hits.x - good_hits.vx) ** 2
                + (good_hits.y - good_hits.vy) ** 2
                + (good_hits.z - good_hits.vz) ** 2
            )
        )
        good_hits = good_hits.sort_values("R")
        edges = true_edges(good_hits)

        self.truth = truth
        self.particles = particles
        self.true_edges = edges
        return True

    def __call__(self, evtid: int, *args: Any, **kwds: Any) -> Any:
        return self.read(evtid)
