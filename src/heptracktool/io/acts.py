"""Read csv files obtained from ACTS and return MeasurementData

Assume the files are like:
acts/event000001000-hits.csv
acts/event000001000-measurements.csv
acts/event000001000-meas2hits.csv
acts/event000001000-spacepoints.csv
acts/event000001000-particles_final.csv
acts/event000001000-cells.csv
"""
from __future__ import annotations
import os
import re
import glob

from heptracktool.io.base import BaseTrackDataReader
from heptracktool.io.utils_feature_store import make_true_edges

import numpy as np
import pandas as pd


class ActsReader(BaseTrackDataReader):
    def __init__(
        self,
        inputdir: str,
        output_dir: str | None = None,
        overwrite: bool = True,
        name: str = "ActsReader",
        spname: str = "spacepoint",
    ):
        super().__init__(inputdir, output_dir, overwrite, name)
        self.spname = spname

        # count how many events in the directory
        all_evts = glob.glob(os.path.join(self.basedir, f"event*-{spname}.csv"))

        pattern = f"event([0-9]*)-{spname}.csv"
        self.all_evtids = sorted(
            [
                int(re.search(pattern, os.path.basename(x)).group(1).strip())
                for x in all_evts
            ]
        )
        self.nevts = len(self.all_evtids)
        print(f"total {self.nevts} events in directory: {self.basedir}")

    def read(self, evtid: int) -> bool:
        """Read one event from the input directory.

        Return:
            MeasurementData
        """
        if (evtid is None or evtid < 1) and self.nevts > 0:
            evtid = self.all_evtids[0]

        prefix = os.path.join(self.basedir, f"event{evtid:09d}")
        hit_fname = f"{prefix}-hits.csv"
        measurements_fname = f"{prefix}-measurements.csv"
        measurements2hits_fname = f"{prefix}-measurement-simhit-map.csv"
        sp_fname = f"{prefix}-{self.spname}.csv"
        p_name = f"{prefix}-particles_final.csv"

        # read hit files
        hits = pd.read_csv(hit_fname)
        hits = hits[hits.columns[:-1]]
        hits = hits.reset_index().rename(columns={"index": "hit_id"})

        # read measurements, maps to hits, and spacepoints
        measurements = pd.read_csv(measurements_fname)
        meas2hits = pd.read_csv(measurements2hits_fname)
        sp = pd.read_csv(sp_fname)

        # read particles and add more variables for performance evaluation
        particles = pd.read_csv(p_name)
        pt = np.sqrt(particles.px**2 + particles.py**2)
        momentum = np.sqrt(pt**2 + particles.pz**2)
        theta = np.arccos(particles.pz / momentum)
        eta = -np.log(np.tan(0.5 * theta))
        radius = np.sqrt(particles.vx**2 + particles.vy**2)
        particles = particles.assign(pt=pt, radius=radius, eta=eta)

        # read cluster information
        cell_fname = f"{prefix}-cells.csv"
        cells = pd.read_csv(cell_fname)

        # calculate cluster shape information
        direction_count_u = cells.groupby(["hit_id"]).channel0.agg(["min", "max"])
        direction_count_v = cells.groupby(["hit_id"]).channel1.agg(["min", "max"])
        nb_u = direction_count_u["max"] - direction_count_u["min"] + 1
        nb_v = direction_count_v["max"] - direction_count_v["min"] + 1
        hit_cells = cells.groupby(["hit_id"]).value.count().values
        hit_value = cells.groupby(["hit_id"]).value.sum().values
        # as I don't access to the rotation matrix and the pixel pitches,
        # I can't calculate cluster's local/global position
        sp = sp.assign(len_u=nb_u, len_v=nb_v, cell_count=hit_cells, cell_val=hit_value)

        sp_hits = sp.merge(meas2hits, on="measurement_id", how="left").merge(
            hits, on="hit_id", how="left"
        )
        sp_hits = sp_hits.merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="left"
        )

        r = np.sqrt(sp_hits.x**2 + sp_hits.y**2)
        phi = np.arctan2(sp_hits.y, sp_hits.x)
        sp_hits = sp_hits.assign(r=r, phi=phi)

        sp_hits = sp_hits.assign(
            R=np.sqrt(
                (sp_hits.x - sp_hits.vx) ** 2
                + (sp_hits.y - sp_hits.vy) ** 2
                + (sp_hits.z - sp_hits.vz) ** 2
            )
        )
        sp_hits = (
            sp_hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)
        )

        edges = make_true_edges(sp_hits)
        self.particles = particles
        self.clusters = measurements
        self.spacepoints = sp_hits
        self.true_edges = edges

        return True
