from pathlib import Path
from typing import Union
import pickle

import pandas as pd


def save_data(df, filename: Union[str, Path]):
    """Save the dataframe to a file"""
    if isinstance(df, pd.DataFrame):
        df.to_parquet(filename, compression="gzip")
    else:
        with open(filename, "wb") as f:
            pickle.dump(df, f)


def load_data(filename: Union[str, Path]):
    """Load the dataframe from a file"""
    if isinstance(filename, Path):
        filename = str(filename)

    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return pd.read_parquet(filename)


def read_or_create(outname: Union[str, Path] = None, overwrite: bool = False):
    """Decorator for reading or creating data"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if Path(outname).exists() and not overwrite:
                data = load_data(outname)
            else:
                data = func(*args, **kwargs)
                save_data(data, outname)
            return data

        return wrapper

    return decorator
