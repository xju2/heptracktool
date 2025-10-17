translator = {
    "Hit_x": "x",
    "Hit_y": "y",
    "Hit_z": "z",
    "Hit_ArrivalTime": "toa",
    "Hit_EnergyDeposited": "energy",
    "Hit_isFromSecondary": "is_from_secondary",
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
hit_info = [(b, c) for b, c in translator.items() if b.startswith("Hit_")]
hit_branch_names = [b for b, _ in hit_info]
hit_col_names = [c for _, c in hit_info]

particle_info = [(b, c) for b, c in translator.items() if b.startswith("MCP_")]
particle_branch_names = [b for b, _ in particle_info]
particle_col_names = [c for _, c in particle_info]
