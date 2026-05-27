# Extract signal tracks
Write a script that creates a new parquet file that contains all signal hits from different events so that I can perform various statistical analysis on the signal hits, for example, comparing signal hit features with all hits.

An example of how to extract signal hits from one event:
```python
from heptracktool.io.muon_collider_track_data import MuonColliderTrackDataReader

import loguru as logger
# log info level
logger.logger.remove()
logger.logger.add(lambda msg: print(msg, end=''), level="INFO")

input_dir = "/global/cfs/cdirs/m3443/data/TrackingInMuonCollider/singleMuonV2"
output_dir = "/global/cfs/cdirs/m3443/data/TrackingInMuonCollider/singleMuonV2_feature_store"
reader = MuonColliderTrackDataReader(
    input_dir=input_dir,
    output_dir=output_dir,
    overwrite=False)

all_signal_hits = []
for idx in range(reader.nevts)[:5000]:
    spacepoints, _ = reader.read(idx)
    signal_hits = spacepoints[spacepoints["hit_is_from_secondary"] == 0.0]
    all_signal_hits.append(signal_hits)

# save all_signal_hits to a parquet file.
```

The parquet file should have columns like `event_id`, `hit_id`, `hit_toa`, `hit_energy`.

Prefer to create a CLI command for this, for example:
```bash
heptracktool extract-signal-tracks -i /global/cfs/cdirs/m3443/data/TrackingInMuonCollider/singleMuonV2 -o /global/cfs/cdirs/m3443/data/TrackingInMuonCollider/singleMuonV2_signal_hits_5000evts.parquet -m 5000 -w 32
```