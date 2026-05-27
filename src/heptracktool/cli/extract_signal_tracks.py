import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

_TASK_TIMEOUT = 120


def _extract_signal_hits(reader, evt_idx, evt_id):
    hits, _ = reader.read(evt_idx)
    if hits is None:
        return None
    mask = hits["hit_is_from_secondary"] == 0.0
    signal_hits = hits.loc[mask].copy()
    del hits, mask
    signal_hits.insert(0, "hit_id", range(len(signal_hits)))
    signal_hits.insert(0, "event_id", evt_id)
    return signal_hits


@click.command("extract-signal-tracks")
@click.option("-i", "--input", "input_dir", type=str, required=True,
              help="Input directory containing raw data files")
@click.option("-o", "--output", "output_path", type=str, required=True,
              help="Output parquet file path")
@click.option("-m", "--max-evts", type=int, default=-1,
              help="Maximum number of events to process (-1 for all)")
@click.option("-w", "--num-workers", type=int, default=1,
              help="Number of worker threads")
@click.option("-t", "--timeout", "task_timeout", type=int, default=_TASK_TIMEOUT,
              help="Per-event timeout in seconds (default: 120)")
def extract_signal_tracks(input_dir, output_path, max_evts, num_workers, task_timeout):
    """Extract signal hits from all events and save to a single parquet file."""
    from heptracktool.io.muon_collider_track_data import MuonColliderTrackDataReader

    reader = MuonColliderTrackDataReader(input_dir=input_dir, output_dir=None, overwrite=False)
    max_evts = reader.nevts if max_evts < 0 else min(max_evts, reader.nevts)
    tasks = [(idx, reader.all_evtids[idx]) for idx in range(max_evts)]

    logger.info(f"Extracting signal hits from {max_evts} events with {num_workers} threads.")

    skipped = []
    all_hits = []

    if num_workers < 2:
        for evt_idx, evt_id in tqdm(tasks, desc="Extracting signal hits"):
            result = _extract_signal_hits(reader, evt_idx, evt_id)
            if result is not None:
                all_hits.append(result)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_extract_signal_hits, reader, idx, eid): (idx, eid)
                for idx, eid in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting signal hits"):
                evt_idx, evt_id = futures[future]
                try:
                    result = future.result(timeout=task_timeout)
                    if result is not None:
                        all_hits.append(result)
                except Exception as exc:
                    logger.warning(f"Event idx={evt_idx} id={evt_id} skipped: {exc}")
                    skipped.append(evt_idx)

    if not all_hits:
        logger.warning("No signal hits found.")
        return

    if skipped:
        logger.warning(f"Skipped {len(skipped)} events: {skipped}")

    logger.info("Concatenating and writing to parquet...")
    table = pa.Table.from_pandas(pd.concat(all_hits, ignore_index=True))
    del all_hits
    pq.write_table(table, output_path)

    logger.info(f"Total signal hits: {len(table)}. Saved to {output_path}")
    logger.info("Done.")
