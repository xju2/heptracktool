translator = {
    "Hit_x": "x",
    "Hit_y": "y",
    "Hit_z": "z",
    "Hit_ArrivalTime": "toa",
    "Hit_EnergyDeposited": "energy",
    "Hit_isFromSecondary": "is_from_secondary",
    "MCP_Vx": "vx",
    "MCP_Vy": "vy",
    "MCP_Vz": "vz",
    "MCP_Px": "px",
    "MCP_Py": "py",
    "MCP_Pz": "pz",
    "MCP_ID": "particle_id",
    "MCP_PDGID": "pdg_id",
    "MCP_Charge": "charge",
}
hit_info = [(b, c) for b, c in translator.items() if b.startswith("Hit_")]
hit_branch_names = [b for b, _ in hit_info]
hit_col_names = [c for _, c in hit_info]

particle_info = [(b, c) for b, c in translator.items() if b.startswith("MCP_")]
particle_branch_names = [b for b, _ in particle_info]
particle_col_names = [c for _, c in particle_info]
