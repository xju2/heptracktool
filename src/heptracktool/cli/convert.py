import click


@click.command("preprocess")
@click.option(
    "-t",
    "--type",
    type=str,
    required=True,
    help="Type of the input data (e.g. 'TrackML', 'MuonCollider')",
    choices=["TrackML", "MuonCollider"],
)
@click.option(
    "-i", "--inputdir", type=str, required=True, help="Input directory containing TrackML data"
)
@click.option(
    "-o", "--outputdir", type=str, required=True, help="Output directory for Parquet data"
)
@click.option("-w", "--num_workers", type=int, default=1, help="Number of worker processes")
def preprocess_data(data_type, inputdir, outputdir, num_workers):
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
        reader.save_all_events(num_workers=num_workers)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
