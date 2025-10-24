import click
import loguru as logger

logger.logger.remove()
logger.logger.add(lambda msg: print(msg, end=""), level="INFO")


@click.command("preprocess")
@click.option(
    "-t",
    "--type",
    "data_type",
    type=str,
    required=True,
    help="Type of the input data (e.g. 'TrackML', 'MuonCollider')",
)
@click.option(
    "-i", "--inputdir", type=str, required=True, help="Input directory containing TrackML data"
)
@click.option(
    "-o", "--outputdir", type=str, required=True, help="Output directory for Parquet data"
)
@click.option(
    "-m",
    "--max-evts",
    type=int,
    default=-1,
    help="Maximum number of events to process (-1 for all events)",
)
@click.option("-w", "--num-workers", type=int, default=1, help="Number of worker processes")
def preprocess_data(data_type, inputdir, outputdir, max_evts, num_workers):
    """Convert Existing open data to Parquet format."""
    if data_type == "TrackML":
        from heptracktool.tools.convert_trackml_to_parquet import convert

        convert(
            inputdir=inputdir,
            outputdir=outputdir,
            num_workers=num_workers,
        )
    elif data_type == "MuonCollider":
        from heptracktool.io.muon_collider_track_data import MuonColliderTrackDataReader

        reader = MuonColliderTrackDataReader(inputdir, outputdir, overwrite=True)
        reader.save_all_events(num_workers=num_workers, max_evts=max_evts)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
