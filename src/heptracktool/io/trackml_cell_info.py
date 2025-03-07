import numpy as np
import pandas as pd


def cartesion_to_spherical(x, y, z):
    r3 = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r3)
    return r3, theta, phi


def theta_to_eta(theta):
    return -np.log(np.tan(0.5 * theta))


def get_all_local_angles(hits, cells, detector):
    direction_count_u = cells.groupby(["hit_id"]).ch0.agg(["min", "max"])
    direction_count_v = cells.groupby(["hit_id"]).ch1.agg(["min", "max"])
    nb_u = direction_count_u["max"] - direction_count_u["min"] + 1
    nb_v = direction_count_v["max"] - direction_count_v["min"] + 1

    vols = hits["volume_id"].values
    layers = hits["layer_id"].values
    modules = hits["module_id"].values

    pitch = detector["pixel_size"]
    thickness = detector["thicknesses"]

    pitch_cells = pitch[vols, layers, modules]
    thickness_cells = thickness[vols, layers, modules]

    l_u = nb_u * pitch_cells[:, 0]
    l_v = nb_v * pitch_cells[:, 1]
    l_w = 2 * thickness_cells
    return l_u, l_v, l_w


def get_all_rotated(hits, detector, l_u, l_v, l_w):
    vols = hits["volume_id"].values
    layers = hits["layer_id"].values
    modules = hits["module_id"].values
    rotations = detector["rotations"]
    rotations_hits = rotations[vols, layers, modules]
    u = l_u.values.reshape(-1, 1)
    v = l_v.values.reshape(-1, 1)
    w = l_w.reshape(-1, 1)
    dirs = np.concatenate((u, v, w), axis=1)

    dirs = np.expand_dims(dirs, axis=2)
    vecRot = np.matmul(rotations_hits, dirs).squeeze(2)
    return vecRot


def extract_dir_new(hits, cells, detector):
    l_u, l_v, l_w = get_all_local_angles(hits, cells, detector)
    g_matrix_all = get_all_rotated(hits, detector, l_u, l_v, l_w)
    hit_ids, cell_counts, cell_vals = (
        hits["hit_id"].to_numpy(),
        hits["cell_count"].to_numpy(),
        hits["cell_val"].to_numpy(),
    )

    l_u, l_v = l_u.to_numpy(), l_v.to_numpy()

    _, g_theta, g_phi = np.vstack(cartesion_to_spherical(*list(g_matrix_all.T)))
    _, l_theta, l_phi = cartesion_to_spherical(l_u, l_v, l_w)

    l_eta = theta_to_eta(l_theta)
    g_eta = theta_to_eta(g_theta)

    angles = np.vstack(
        [hit_ids, cell_counts, cell_vals, l_eta, l_phi, l_u, l_v, l_w, g_eta, g_phi]
    ).T
    df_angles = pd.DataFrame(
        angles,
        columns=[
            "hit_id",
            "cell_count",
            "cell_val",
            "leta",
            "lphi",
            "lx",
            "ly",
            "lz",
            "geta",
            "gphi",
        ],
    )

    return df_angles


#############################################
#           FEATURE_AUGMENTATION            #
#############################################


def add_cluster_shape(hits, cells, detector):
    hit_cells = cells.groupby(["hit_id"]).value.count().values
    hit_value = cells.groupby(["hit_id"]).value.sum().values
    hits["cell_count"] = hit_cells
    hits["cell_val"] = hit_value

    l_u, l_v, l_w = get_all_local_angles(hits, cells, detector)
    g_matrix_all = get_all_rotated(hits, detector, l_u, l_v, l_w)
    hit_ids, cell_counts, cell_vals = (
        hits["hit_id"].to_numpy(),
        hits["cell_count"].to_numpy(),
        hits["cell_val"].to_numpy(),
    )

    l_u, l_v = l_u.to_numpy(), l_v.to_numpy()

    _, g_theta, g_phi = np.vstack(cartesion_to_spherical(*list(g_matrix_all.T)))
    _, l_theta, l_phi = cartesion_to_spherical(l_u, l_v, l_w)

    l_eta = theta_to_eta(l_theta)
    g_eta = theta_to_eta(g_theta)

    hits = hits.assign(
        leta=l_eta, lphi=l_phi, lx=l_u, ly=l_v, lz=l_w, geta=g_eta, gphi=g_phi
    )

    return hits
