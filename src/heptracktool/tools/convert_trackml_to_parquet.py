import contextlib
from functools import partial
from pathlib import Path

import joblib
import pandas as pd
from fastparquet import write
from joblib import Parallel, delayed
from heptracktool.io.trackml import TrackMLReader
from heptracktool.io.base import BaseTrackDataReader
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def convert_to_parquet(idx: int, reader: BaseTrackDataReader, outdir: str):
    outname = Path(f"{outdir}/spacepoints/{idx:09d}.parquet")
    if outname.exists():
        return

    tracker_data = reader.read(idx)
    particles = tracker_data.particles
    spacepoints = tracker_data.hits
    true_edges = tracker_data.true_edges

    parq_kwargs = {"compression": "GZIP", "file_scheme": "simple"}
    write(f"{outdir}/particles/particles_{idx:05}.parquet", particles, **parq_kwargs)  # type: ignore
    write(f"{outdir}/spacepoints/spacepoints_{idx:05}.parquet", spacepoints, **parq_kwargs)  # type: ignore
    df_true_edges = pd.DataFrame(true_edges.T, columns=["src", "dst"])
    write(f"{outdir}/true_edges/true_edges_{idx:05}.parquet", df_true_edges, **parq_kwargs)  # type: ignore


def convert(inputdir: str, outputdir: str, num_workers: int = 1):
    print(f"inputdir: {inputdir}")
    print(f"outputdir: {outputdir}")
    reader = TrackMLReader(inputdir)
    Path(outputdir).mkdir(parents=True, exist_ok=True)

    tot_evts = reader.nevts
    Path(f"{outputdir}/particles").mkdir(parents=True, exist_ok=True)
    Path(f"{outputdir}/spacepoints").mkdir(parents=True, exist_ok=True)
    Path(f"{outputdir}/true_edges").mkdir(parents=True, exist_ok=True)

    process_fn = partial(convert_to_parquet, reader=reader, outdir=outputdir)

    with tqdm_joblib(tqdm(desc="Converting events", total=tot_evts)) as _:
        Parallel(n_jobs=num_workers)(delayed(process_fn)(evt_idx) for evt_idx in range(tot_evts))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Reader")
    add_arg = parser.add_argument
    add_arg("inputdir", help="Input directory")
    add_arg("outputdir", help="Output directory")
    add_arg("-w", "--num_workers", type=int, default=1)

    convert(**vars(parser.parse_args()))
