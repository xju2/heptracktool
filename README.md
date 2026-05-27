# HEP Tracking Tools


## Introduction
This repository goes with the [ACORN](https://gitlab.cern.ch/gnn4itkteam/acorn/-/tree/dev/acorn?ref_type=heads) framework, which is a framework for training ML models for HEP tracking.


## Installation
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the `heptracktool` package:
```bash
uv sync --group dev --group docs
```

To install additional PyG packages such as `torch-cluster`:
```bash
uv run pip install --no-cache-dir --force-reinstall torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```


## Preprocessing TrackML data

```bash
heptracktool preprocess -t TrackML -i "/global/cfs/cdirs/m3443/data/trackml-codalab/train_100" -o "/global/cfs/cdirs/m3443/usr/xju/data/trackml/train_100_parquet" -w 32
heptracktool preprocess -t TrackML -i "/global/cfs/cdirs/m3443/data/trackml-codalab/train_all" -o "/global/cfs/cdirs/m3443/usr/xju/data/trackml/train_all_parquet" -w 32

heptracktool preprocess -t MuonCollider -i /global/cfs/cdirs/m3443/data/TrackingInMuonCollider/singleMuonV2/New -o /global/cfs/cdirs/m3443/data/TrackingInMuonCollider/singleMuonV2_feature_store -m -1 -w 32
```
