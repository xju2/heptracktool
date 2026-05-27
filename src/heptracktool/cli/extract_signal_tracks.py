import click
import pandas as pd
from loguru import logger
from multiprocessing import Pool
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

_global_reader = None


def _init_worker(input_dir):
    global _global_reader
    from heptracktool.io.muon_collider_track_data import MuonColliderTrackDataReader

    _global_reader = MuonColliderTrackDataReader(
        input_dir=input_dir, output_dir=None, overwrite=False
    )


def _extract_signal_hits(args):
    global _global_reader
    evt_idx, evt_id = args
    hits, _ = _global_reader.read(evt_idx)
    if hits is None:
        return None
    signal_hits = hits[hits["hit_is_from_secondary"] == 0.0].copy()
    signal_hits.insert(0, "hit_id", range(len(signal_hits)))
    signal_hits.insert(0, "event_id", evt_id)
    return signal_hits


@click.command("extract-signal-tracks")
@click.option(
    "-i", "--input", "input_dir", type=str, required=True,
    help="Input directory containing raw data files",
)
@click.option(
    "-o", "--output", "output_path", type=str, required=True,
    help="Output parquet file path",
)
@click.option(
    "-m", "--max-evts", type=int, default=-1,
    help="Maximum number of events to process (-1 for all)",
)
@click.option(
    "-w", "--num-workers", type=int, default=1,
    help="Number of worker processes",
)
def extract_signal_tracks(input_dir, output_path, max_evts, num_workers):
    """Extract signal hits from all events and save to a single parquet file."""
    from heptracktool.io.muon_collider_track_data import MuonColliderTrackDataReader

    reader = MuonColliderTrackDataReader(input_dir=input_dir, output_dir=None, overwrite=False)
    max_evts = reader.nevts if max_evts < 0 else min(max_evts, reader.nevts)
    tasks = [(idx, reader.all_evtids[idx]) for idx in range(max_evts)]

    logger.info(f"Extracting signal hits from {max_evts} events with {num_workers} workers.")

    all_hits = []
    if num_workers < 2:
        for evt_idx, evt_id in tqdm(tasks, desc="Extracting signal hits"):
            hits, _ = reader.read(evt_idx)
            if hits is None:
                continue
            signal_hits = hits[hits["hit_is_from_secondary"] == 0.0].copy()
            signal_hits.insert(0, "hit_id", range(len(signal_hits)))
            signal_hits.insert(0, "event_id", evt_id)
            all_hits.append(signal_hits)
    else:
        with Pool(num_workers, initializer=_init_worker, initargs=(input_dir,)) as pool:
            results = list(
                tqdm(
                    pool.imap(_extract_signal_hits, tasks),
                    total=len(tasks),
                    desc="Extracting signal hits",
                )
            )
        all_hits = [r for r in results if r is not None]

    if not all_hits:
        logger.warning("No signal hits found.")
        return

    df = pd.concat(all_hits, ignore_index=True)
    logger.info(f"Total signal hits: {len(df)}. Saving to {output_path}")
    df.to_parquet(output_path, index=False)
    logger.info("Done.")
