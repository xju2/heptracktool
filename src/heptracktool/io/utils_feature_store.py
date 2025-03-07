"""Utilities for feature stores
* read feature files that used for embedding.
* save measurement data for training embedding.
"""
import numpy as np
import itertools

scales = np.array([1000, np.pi, 1000], dtype=np.float32)

particle_features = ["particle_id", "pt", "eta", "vx", "vy", "vz", "radius"]
cluster_shape_features = [
    "len_u",
    "len_v",
    "cell_count",
    "cell_val",
]


def save_to_np(outname, data):
    hits = data.spacepoints
    edges = data.true_edges
    if "len_u" in hits.columns:
        cells = hits[cluster_shape_features].values
    else:
        cells = None

    np.savez_compressed(
        outname,
        x=hits[["r", "phi", "z"]].values / scales,
        pid=hits["particle_id"].values,
        hid=hits["hit_id"].values,
        true_edges=edges,
        particles=data.particles[particle_features].values,
        cells=cells,
    )


def load_from_np(fname, edge_name="true_edges"):
    arrays = np.load(fname)
    hits, pids, true_edges, cells, particles = (
        arrays["x"],
        arrays["pid"],
        arrays[edge_name],
        arrays["cells"],
        arrays["particles"],
    )
    return (hits, pids, true_edges, cells, particles)


def dump_data(data):
    # data are those load from numpy array,
    # i.e. hits, pids, true_edges, cells, particles
    hits, pids, true_edges, cells, particles = data
    print("hits:", hits.shape)
    print("pids:", pids.shape)
    print("true_edges:", true_edges.shape)
    print("cells:", cells.shape)
    print("particles:", particles.shape)
    # <TODO> plot the edges


def make_true_edges(hits):
    hits = hits.assign(R=np.sqrt((hits.x - hits.vx)**2 + (hits.y - hits.vy)**2 + (hits.z - hits.vz)**2))
    hits = hits.sort_values('R').reset_index(drop=True).reset_index(drop=False)
    hit_list = hits.groupby(['particle_id', 'geometry_id'], sort=False)['index'].agg(lambda x: list(x)).groupby(level=0).agg(lambda x: list(x))

    e = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            e.extend(list(itertools.product(i, j)))

    layerless_true_edges = np.array(e).T
    return layerless_true_edges
