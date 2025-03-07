import numpy as np

translator = {
    "Part_event_number": "subevent",
    "Part_barcode": "barcode",
    "Part_px": "px",
    "Part_py": "py",
    "Part_pz": "pz",
    "Part_pt": "pt",
    "Part_eta": "eta",
    "Part_vx": "vx",
    "Part_vy": "vy",
    "Part_vz": "vz",
    "Part_radius": "radius",
    "Part_status": "status",
    "Part_charge": "charge",
    "Part_pdg_id": "pdgId",
    "Part_passed": "pass",
    "Part_vProdNin": "vProdNIn",
    "Part_vProdNout": "vProdNOut",
    "Part_vProdStatus": "vProdStatus",
    "Part_vProdBarcode": "vProdBarcode",
    "SPindex": "hit_id",
    "SPx": "x",
    "SPy": "y",
    "SPz": "z",
    "SPCL1_index": "cluster_index_1",
    "SPCL2_index": "cluster_index_2",
    "SPisOverlap": "SPisOverlap",
    "CLindex": "cluster_id",
    "CLhardware": "hardware",
    "CLx": "cluster_x",
    "CLy": "cluster_y",
    "CLz": "cluster_z",
    "CLbarrel_endcap": "barrel_endcap",
    "CLlayer_disk": "layer_disk",
    "CLeta_module": "eta_module",
    "CLphi_module": "phi_module",
    "CLside": "side",
    "CLmoduleID": "module_id",
    "CLpixel_count": "count",
    "CLcharge_count": "charge_count",
    "CLloc_eta": "loc_eta",
    "CLloc_phi": "loc_phi",
    "CLloc_direction1": "localDir0",
    "CLloc_direction2": "localDir1",
    "CLloc_direction3": "localDir2",
    "CLJan_loc_direction1": "lengthDir0",
    "CLJan_loc_direction2": "lengthDir1",
    "CLJan_loc_direction3": "lengthDir2",
    "CLglob_eta": "glob_eta",
    "CLglob_phi": "glob_phi",
    "CLeta_angle": "eta_angle",
    "CLphi_angle": "phi_angle",
    "CLnorm_x": "norm_x",
    "CLnorm_y": "norm_y",
    "CLnorm_z": "norm_z",
    "CLparticleLink_eventIndex": "subevent",
    "CLparticleLink_barcode": "barcode",
    "DTTindex": "dtt_trkid",
    "DTTsize": "dtt_num_matched",
    "DTTtrajectory_eventindex": "dtt_eventindex",
    "DTTtrajectory_barcode": "dtt_barcode",
    "DTTstTruth_subDetType": "dtt_true_hits",
    "DTTstTrack_subDetType": "dtt_track_hits",
    "DTTstCommon_subDetType": "dtt_common_hits",
    "TTCevent_index": ("subevent", int),
    "TTCparticle_link": ("barcode", np.int64),
    "TTCprobability": ("probability", float),
    "TRKindex": ("trkid", np.int64),
    "TRKtrack_fitter": ("track_fitter", int),
    "TRKparticle_hypothesis": ("particle_hypothesis", int),
    "TRKndof": ("nDoF", int),
    "TRKchiSq": ("chi2", float),
    "TRKmot": ("mot", int),
    "TRKoot": ("oot", int),
    "TRKcharge": ("charge", int),
    "TRKspacepointsIdxOnTrack": ("spIdxOnTrack", np.int64),
    "TRKspacepointsIdxOnTrack_trkIndex": ("trkid", np.int64),
    "TRKspacepointsIsPixel": ("isPixel", int),
}

event_branch_names = ["run_number", "event_number"]
particle_info = [(b, c) for b, c in translator.items() if b.startswith("Part_")]
particle_branch_names = [b for b, _ in particle_info]
particle_col_names = [c for _, c in particle_info]

spacepoint_info = [(b, c) for b, c in translator.items() if b.startswith("SP")]
spacepoint_branch_names = [b for b, _ in spacepoint_info]
spacepoint_col_names = [c for _, c in spacepoint_info]

cluster_info = [(b, c) for b, c in translator.items() if b.startswith("CL")]
cluster_branch_names = [b for b, _ in cluster_info]
cluster_col_names = [c for _, c in cluster_info]

detailed_truth_info = [(b, c) for b, c in translator.items() if b.startswith("DTT")]
detailed_truth_branch_names = [b for b, _ in detailed_truth_info]
detailed_truth_col_names = [c for _, c in detailed_truth_info]

reco_track_info = [
    (b, c)
    for b, c in translator.items()
    if ((b.startswith(("TT", "TRK"))) and not b.startswith("TRKspacepoints"))
]
reco_track_branch_names = [b for b, _ in reco_track_info]
reco_track_col_names = [c[0] for _, c in reco_track_info]
reco_track_col_types = {c[0]: c[1] for _, c in reco_track_info}

reco_track_sp_info = [
    (b, c) for b, c in translator.items() if b.startswith("TRKspacepoints")
]
reco_track_sp_branch_names = [b for b, _ in reco_track_sp_info]
reco_track_sp_col_names = [c[0] for _, c in reco_track_sp_info]
reco_track_sp_col_types = {c[0]: c[1] for _, c in reco_track_sp_info}

all_branches = (
    event_branch_names
    + particle_branch_names
    + spacepoint_branch_names
    + cluster_branch_names
    + detailed_truth_branch_names
    + reco_track_branch_names
    + reco_track_sp_branch_names
    + ["TRKmeasurementsOnTrack_pixcl_sctcl_index"]
)

cluster_link_branch_names = ["CLparticleLink_eventIndex", "CLparticleLink_barcode"]
# To get the same column order as with txt reading
truth_col_order = [
    "hit_id",
    "x",
    "y",
    "z",
    "cluster_index_1",
    "cluster_index_2",
    "hardware",
    "cluster_x_1",
    "cluster_y_1",
    "cluster_z_1",
    "barrel_endcap",
    "layer_disk",
    "eta_module",
    "phi_module",
    "side_1",
    "norm_x_1",
    "norm_y_1",
    "norm_z_1",
    "count_1",
    "charge_count_1",
    "loc_eta_1",
    "loc_phi_1",
    "localDir0_1",
    "localDir1_1",
    "localDir2_1",
    "lengthDir0_1",
    "lengthDir1_1",
    "lengthDir2_1",
    "glob_eta_1",
    "glob_phi_1",
    "eta_angle_1",
    "phi_angle_1",
    "particle_id_1",
    "cluster_x_2",
    "cluster_y_2",
    "cluster_z_2",
    "side_2",
    "norm_x_2",
    "norm_y_2",
    "norm_z_2",
    "count_2",
    "charge_count_2",
    "loc_eta_2",
    "loc_phi_2",
    "localDir0_2",
    "localDir1_2",
    "localDir2_2",
    "lengthDir0_2",
    "lengthDir1_2",
    "lengthDir2_2",
    "glob_eta_2",
    "glob_phi_2",
    "eta_angle_2",
    "phi_angle_2",
    "particle_id_2",
    "particle_id",
    "region",
    "module_id",
    "SPisOverlap",
]

particles_col_order = [
    "particle_id",
    "subevent",
    "barcode",
    "px",
    "py",
    "pz",
    "pt",
    "eta",
    "vx",
    "vy",
    "vz",
    "radius",
    "status",
    "charge",
    "pdgId",
    "pass",
    "vProdNIn",
    "vProdNOut",
    "vProdStatus",
    "vProdBarcode",
    "num_clusters",
]
