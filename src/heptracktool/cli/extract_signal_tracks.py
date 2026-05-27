import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from threading import Lock
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

_TASK_TIMEOUT = 120
_BATCH_SIZE = 500


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
@click.option("-b", "--batch-size", "batch_size", type=int, default=_BATCH_SIZE,
              help="Events submitted per batch; controls peak memory (default: 500)")
def extract_signal_tracks(input_dir, output_path, max_evts, num_workers, task_timeout, batch_size):
    """Extract signal hits from all events and save to a single parquet file."""
    from heptracktool.io.muon_collider_track_data import MuonColliderTrackDataReader

    reader = MuonColliderTrackDataReader(input_dir=input_dir, output_dir=None, overwrite=False)
    max_evts = reader.nevts if max_evts < 0 else min(max_evts, reader.nevts)
    tasks = [(idx, reader.all_evtids[idx]) for idx in range(max_evts)]

    logger.info(f"Extracting signal hits from {max_evts} events with {num_workers} threads.")

    skipped = []
    total_hits = 0
    writer = None
    write_lock = Lock()

    def flush(batch):
        nonlocal total_hits, writer
        if not batch:
            return
        n = sum(len(df) for df in batch)
        table = pa.Table.from_pandas(pd.concat(batch, ignore_index=True))
        batch.clear()
        with write_lock:
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
        total_hits += n

    if num_workers < 2:
        batch = []
        for evt_idx, evt_id in tqdm(tasks, desc="Extracting signal hits"):
            result = _extract_signal_hits(reader, evt_idx, evt_id)
            if result is not None:
                batch.append(result)
            if len(batch) >= batch_size:
                flush(batch)
        flush(batch)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=max_evts, desc="Extracting signal hits") as pbar:
                for start in range(0, len(tasks), batch_size):
                    chunk = tasks[start: start + batch_size]
                    futures = {
                        executor.submit(_extract_signal_hits, reader, idx, eid): (idx, eid)
                        for idx, eid in chunk
                    }
                    batch = []
                    for future in as_completed(futures):
                        evt_idx, evt_id = futures[future]
                        try:
                            result = future.result(timeout=task_timeout)
                            if result is not None:
                                batch.append(result)
                        except Exception as exc:
                            logger.warning(f"Event idx={evt_idx} id={evt_id} skipped: {exc}")
                            skipped.append(evt_idx)
                        pbar.update(1)
                    flush(batch)

    if writer is not None:
        writer.close()
    else:
        logger.warning("No signal hits found.")
        return

    if skipped:
        logger.warning(f"Skipped {len(skipped)} events: {skipped}")

    logger.info(f"Total signal hits: {total_hits}. Saved to {output_path}")
    logger.info("Done.")
