import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from multiprocessing import Pool
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

_global_reader = None

# Per-event read timeout (seconds).
_TASK_TIMEOUT = 120
# Number of tasks submitted to the pool at a time; bounds in-flight memory.
_BATCH_SIZE = 500


def _init_worker(input_dir):
    global _global_reader
    from heptracktool.io.muon_collider_track_data import MuonColliderTrackDataReader

    logger.disable("heptracktool")
    _global_reader = MuonColliderTrackDataReader(
        input_dir=input_dir, output_dir=None, overwrite=False
    )
    logger.enable("heptracktool")


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


def _flush(writer_box, batch, output_path):
    """Concat batch, write to parquet, return updated writer. Clears batch in-place."""
    if not batch:
        return writer_box
    table = pa.Table.from_pandas(pd.concat(batch, ignore_index=True))
    batch.clear()
    if writer_box[0] is None:
        writer_box[0] = pq.ParquetWriter(output_path, table.schema)
    writer_box[0].write_table(table)
    return writer_box


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
@click.option(
    "-t", "--timeout", "task_timeout", type=int, default=_TASK_TIMEOUT,
    help="Per-event timeout in seconds (default: 120)",
)
@click.option(
    "-b", "--batch-size", "batch_size", type=int, default=_BATCH_SIZE,
    help="Tasks submitted to the pool at once; controls peak memory (default: 500)",
)
def extract_signal_tracks(input_dir, output_path, max_evts, num_workers, task_timeout, batch_size):
    """Extract signal hits from all events and save to a single parquet file."""
    from heptracktool.io.muon_collider_track_data import MuonColliderTrackDataReader

    reader = MuonColliderTrackDataReader(input_dir=input_dir, output_dir=None, overwrite=False)
    max_evts = reader.nevts if max_evts < 0 else min(max_evts, reader.nevts)
    tasks = [(idx, reader.all_evtids[idx]) for idx in range(max_evts)]

    logger.info(f"Extracting signal hits from {max_evts} events with {num_workers} workers.")

    skipped = []
    total_hits = 0
    # writer_box is a mutable container so _flush can assign the writer lazily.
    writer_box = [None]
    batch = []

    def flush():
        nonlocal total_hits
        if batch:
            before = sum(len(df) for df in batch)
            _flush(writer_box, batch, output_path)
            total_hits += before

    if num_workers < 2:
        for evt_idx, evt_id in tqdm(tasks, desc="Extracting signal hits"):
            hits, _ = reader.read(evt_idx)
            if hits is None:
                continue
            signal_hits = hits[hits["hit_is_from_secondary"] == 0.0].copy()
            signal_hits.insert(0, "hit_id", range(len(signal_hits)))
            signal_hits.insert(0, "event_id", evt_id)
            batch.append(signal_hits)
            if len(batch) >= batch_size:
                flush()
    else:
        with Pool(num_workers, initializer=_init_worker, initargs=(input_dir,),
                  maxtasksperchild=200) as pool:
            with tqdm(total=max_evts, desc="Extracting signal hits") as pbar:
                # Submit tasks in batches to bound the number of in-flight results.
                for start in range(0, len(tasks), batch_size):
                    chunk = tasks[start: start + batch_size]
                    async_results = [(args, pool.apply_async(_extract_signal_hits, (args,)))
                                     for args in chunk]
                    for args, ar in async_results:
                        evt_idx, evt_id = args
                        try:
                            result = ar.get(timeout=task_timeout)
                            if result is not None:
                                batch.append(result)
                        except Exception as exc:
                            logger.warning(f"Event idx={evt_idx} id={evt_id} skipped: {exc}")
                            skipped.append(evt_idx)
                        pbar.update(1)
                    flush()

    flush()  # write any remainder (single-worker path)

    if writer_box[0] is not None:
        writer_box[0].close()
    else:
        logger.warning("No signal hits found.")
        return

    if skipped:
        logger.warning(f"Skipped {len(skipped)} events: {skipped}")

    logger.info(f"Total signal hits: {total_hits}. Saved to {output_path}")
    logger.info("Done.")
