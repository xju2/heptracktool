import re
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd


# <TODO: add a reconstructed tracks writer for ML-based pipeline>
class RecoTrkxReader:
    """Read reconstructed tracks from ML-based method or CKF method in athena"""

    def __init__(self, reco_track_path: str):
        self.path = reco_track_path

        all_files = Path(reco_track_path).glob("*.npz")
        self.nevts = len(all_files)
        self.all_evtids = [int(Path(x).name[:-4]) for x in all_files]

    def __call__(self, evtid: int, *args: Any, **kwds: Any) -> Any:
        tracks_fname = Path(self.path) / f"{evtid}.npz"
        arrays = np.load(tracks_fname)
        predicts = arrays["predicts"]
        if predicts.shape[1] == 2:
            columns = ["track_id", "hit_id"]
        elif predicts.shape[1] == 3:
            columns = ["track_id", "cluster_id", "hit_id"]
        else:
            return None

        submission = pd.DataFrame(predicts, columns=columns)
        submission.drop_duplicates(subset=["hit_id"], inplace=True)
        return submission


class ACTSTrkxReader:
    """Read reconstructed tracks dumped from the CKF method in ACTS"""

    def __init__(self, reco_track_path: str):
        self.path = Path(reco_track_path)
        self.filename = "event{:09}-CKFtracks.csv"

        all_files = Path(self.path).glob("event*-CKFtracks.csv")
        self.nevts = len(all_files)

        pattern = "event([0-9]*)-CKFtracks.csv"
        self.all_evtids = [
            int(re.search(pattern, Path(x).name).group(1).strip()) for x in all_files
        ]
        print(f"Total {self.nevts} events in directory: {self.path!s}")

    def __call__(self, evtid: int, *args: Any, **kwds: Any) -> Any:
        """Return reconstructed track candidates"""
        tracks_fname = self.path / self.filename.format(evtid)
        submission = pd.read_csv(tracks_fname)
        submission.rename(columns={"measurement_id": "hit_id"}, inplace=True)
        return submission
