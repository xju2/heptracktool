import click
from heptracktool.tools.convert_trackml_to_parquet import convert


@click.command("convert")
@click.option(
    "-i", "--inputdir", type=str, required=True, help="Input directory containing TrackML data"
)
@click.option(
    "-o", "--outputdir", type=str, required=True, help="Output directory for Parquet data"
)
@click.option("-w", "--num_workers", type=int, default=1, help="Number of worker processes")
def convert_trackml(inputdir, outputdir, num_workers):
    """Convert TrackML data to Parquet format."""
    convert(inputdir=inputdir, outputdir=outputdir, num_workers=num_workers)
