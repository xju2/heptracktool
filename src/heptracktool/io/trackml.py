"""Read TrackML data.
https://www.kaggle.com/c/trackml-particle-identification/overview

Assume the files are like:
detectors.csv
trackml/event000001000-hits.csv
trackml/event000001000-cells.csv
trackml/event000001000-particles.csv
trackml/event000001000-truth.csv
"""

import os
import glob
import re

import numpy as np
import pandas as pd

from heptracktool.io.base import BaseTrackDataReader
from heptracktool.io.utils_feature_store import make_true_edges
from heptracktool.io.trackml_cell_info import add_cluster_shape
from heptracktool.io.trackml_detector import load_detector

__all__ = ["TrackMLReader", "select_barrel_hits", "remove_noise_hits"]


# predefined layer info
# in Tracking ML, layer is defined by (volumn id and layer id)
# now I just use a unique layer id
vlids = [(7, 2), (7, 4), (7, 6), (7, 8), (7, 10), (7, 12), (7, 14),
         (8, 2), (8, 4), (8, 6), (8, 8),
         (9, 2), (9, 4), (9, 6), (9, 8), (9, 10), (9, 12), (9, 14),
         (12, 2), (12, 4), (12, 6), (12, 8), (12, 10), (12, 12),
         (13, 2), (13, 4), (13, 6), (13, 8),
         (14, 2), (14, 4), (14, 6), (14, 8), (14, 10), (14, 12),
         (16, 2), (16, 4), (16, 6), (16, 8), (16, 10), (16, 12),
         (17, 2), (17, 4),
         (18, 2), (18, 4), (18, 6), (18, 8), (18, 10), (18, 12)]
n_det_layers = len(vlids)


def select_barrel_hits(hits):
    """Select barrel hits."""
    vlids = [
        (8, 2),
        (8, 4),
        (8, 6),
        (8, 8),
        (13, 2),
        (13, 4),
        (13, 6),
        (13, 8),
        (17, 2),
        (17, 4),
    ]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(["volume_id", "layer_id"])
    hits = pd.concat([
        vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)
    ])
    return hits


def remove_noise_hits(hits):
    """Remove noise hits."""
    # Remove noise hits
    hits = hits[hits.hit_type != "noise"]
    return hits


class TrackMLReader(BaseTrackDataReader):
    def __init__(self, basedir, name="TrackMLReader", is_codalab_data: bool = True) -> None:
        super().__init__(basedir, name)
        self.suffix = ".csv.gz" if is_codalab_data else "csv"
        # count how many events in the directory
        all_evts = glob.glob(os.path.join(self.basedir, "event*-hits.csv"))

        self.nevts = len(all_evts)
        pattern = "event([0-9]*)-hits.csv"
        if is_codalab_data:
            pattern = "event([0-9]*)-hits" + self.suffix

        self.all_evtids = sorted([
            int(re.search(pattern, os.path.basename(x)).group(1).strip()) for x in all_evts
        ])

        print(f"total {self.nevts} events in directory: {self.inputdir}")

        detector_path = os.path.join(self.inputdir, "../detector.csv")
        origin_detector_info, self.detector = load_detector(detector_path)
        self.build_detector_vocabulary(origin_detector_info)

    def build_detector_vocabulary(self, detector):
        # number of modules in the detector
        # we reserve the first 6 indices for special tokens
        # Reserve the following special tokens.
        # * 1: start of the track (perigee point)
        # * 2: a hole in the track
        # * 3: the end of a track
        # * 4: unknown measurement
        # * 5: masked measurement
        # * 6: padding

        detector_umid = np.stack(
            [detector.volume_id, detector.layer_id, detector.module_id], axis=1
        )
        umid_dict = {}
        index = 1
        for i in detector_umid:
            umid_dict[tuple(i)] = index
            index += 1
        self.umid_dict = umid_dict
        self.num_modules = len(detector_umid)
        pixel_moudels = [k for k in umid_dict.keys() if k[0] in [7, 8, 9]]
        self.num_pixel_modules = len(pixel_moudels)
        # Inverting the umid_dict
        self.umid_dict_inv = {v: k for k, v in umid_dict.items()}

    def read(self, evtid: int) -> bool:
        """Read one event from the input directory"""

        if (evtid is None or evtid < 1) and self.nevts > 0:
            evtid = self.all_evtids[0]
            print(f"read event {evtid}")

        prefix = os.path.join(self.inputdir, f"event{evtid:09d}")
        hit_fname = f"{prefix}-hits{self.suffix}"
        cell_fname = f"{prefix}-cells{self.suffix}"
        particle_fname = f"{prefix}-particles{self.suffix}"
        truth_fname = f"{prefix}-truth{self.suffix}"

        # read all files
        hits = pd.read_csv(hit_fname)
        r = np.sqrt(hits.x**2 + hits.y**2)
        phi = np.arctan2(hits.y, hits.x)
        hits = hits.assign(r=r, phi=phi)

        # read truth info about hits and particles
        truth = pd.read_csv(truth_fname)
        particles = pd.read_csv(particle_fname)
        # add dummy particle information for noise hits
        # whose particle ID is zero.
        # particle_id,vx,vy,vz,px,py,pz,q,nhits
        # for codalab dataset: (there is a particle type column)
        # particle_id,particle_type,vx,vy,vz,px,py,pz,q,nhits
        particles.loc[len(particles.index)] = 0
        self.particles = particles

        truth = truth.merge(particles, on="particle_id", how="left")
        truth = truth.assign(pt=np.sqrt(truth.px**2 + truth.py**2))
        self.truth = truth

        hits = hits.merge(
            truth[["hit_id", "particle_id", "vx", "vy", "vz", "pt", "weight", "nhits"]],
            on="hit_id",
        )
        is_pixel = (hits.volume_id == 7) | (hits.volume_id == 8) | (hits.volume_id == 9)
        hits = hits.assign(is_pixel=is_pixel)

        # add detector unique module ID
        vlid_groups = hits.groupby(["volume_id", "layer_id", "module_id"])
        hits = pd.concat([
            vlid_groups.get_group(vlid).assign(umid=self.umid_dict[vlid])
            for vlid in vlid_groups.groups.keys()
        ])
        # add detector unique layer ID
        vlid_groups = hits.groupby(["volume_id", "layer_id"])
        hits = pd.concat([
            vlid_groups.get_group(vlids[i]).assign(geometry_id=i) for i in range(n_det_layers)
        ])

        true_edges = make_true_edges(hits)
        cells = pd.read_csv(cell_fname)
        hits = add_cluster_shape(hits, cells, self.detector)

        hits = hits.assign(
            R=np.sqrt((hits.x - hits.vx) ** 2 + (hits.y - hits.vy) ** 2 + (hits.z - hits.vz) ** 2)
        )
        hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)

        self.spacepoints = hits
        self.true_edges = true_edges
        return True
