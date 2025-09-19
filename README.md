# HEP Tracking Tools


## Introduction
This repository goes with the [ACORN](https://gitlab.cern.ch/gnn4itkteam/acorn/-/tree/dev/acorn?ref_type=heads) framework, which is a framework for training ML models for HEP tracking.


## Installation
First install the `poetry` tool by following the <script src="https://gist.github.com/xju2/b4fd9d7db62a7a5345b190aae75d231e.js"></script> instructions.

Then you can install the `heptracktool` package by
```bash
poetry install --with dev,docs
```
Then you need to install the `FRNN` and `torch-cluster` packages.
```bash
pip install --no-cache-dir --force-reinstall torch_cluster  -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```


## Preprocessing TrackML data

```bash
heptracktool convert -i "/global/cfs/cdirs/m3443/data/trackml-codalab/train_100" -o "/global/cfs/cdirs/m3443/usr/xju/data/trackml/train_100_parquet" -w 32


heptracktool convert -i "/global/cfs/cdirs/m3443/data/trackml-codalab/train_all" -o "/global/cfs/cdirs/m3443/usr/xju/data/trackml/train_all_parquet" -w 32
```
