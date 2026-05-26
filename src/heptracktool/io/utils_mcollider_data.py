import pandas as pd
import numpy as np


translator = {
    "Cluster_x": "hit_x",
    "Cluster_y": "hit_y",
    "Cluster_z": "hit_z",
    "Cluster_ArrivalTime": "hit_toa",
    "Cluster_EnergyDeposited": "hit_charge",
    "Cluster_isFromSecondary": "hit_is_from_secondary",
    "Cluster_Npixels": "hit_charge_count",
    "PixelHits_x": "cell_x",
    "PixelHits_y": "cell_y",
    "PixelHits_ArrivalTime": "cell_toa",
    "PixelHits_EnergyDeposited": "cell_charge",
    "MCP_Vx": "par_vx",
    "MCP_Vy": "par_vy",
    "MCP_Vz": "par_vz",
    "MCP_Px": "par_px",
    "MCP_Py": "par_py",
    "MCP_Pz": "par_pz",
    "MCP_ID": "par_id",  # the same as particle_id
    "MCP_PDGID": "par_pdg_id",
    "MCP_Charge": "par_charge",
}
hit_info = [(b, c) for b, c in translator.items() if b.startswith("Cluster_")]
hit_branch_names = [b for b, _ in hit_info]
hit_col_names = [c for _, c in hit_info]

cell_info = [(b, c) for b, c in translator.items() if b.startswith("PixelHits_")]
cell_branch_names = [b for b, _ in cell_info]
cell_col_names = [c for _, c in cell_info]

particle_info = [(b, c) for b, c in translator.items() if b.startswith("MCP_")]
particle_branch_names = [b for b, _ in particle_info]
particle_col_names = [c for _, c in particle_info]


def extract_cell_features(cells):
    lx_minmax = cells.groupby("hit_id").cell_x.agg(["min", "max"])
    ly_minmax = cells.groupby("hit_id").cell_y.agg(["min", "max"])
    l_x = lx_minmax["max"] - lx_minmax["min"]
    l_y = ly_minmax["max"] - ly_minmax["min"]
    l_phi = np.arctan2(l_y, l_x)
    l_r2 = l_x**2 + l_y**2
    cell_features = pd.DataFrame({
        "hit_lx": l_x,
        "hit_ly": l_y,
        "hit_lphi": l_phi,
        "hit_lr2": l_r2,
    })
    return cell_features
