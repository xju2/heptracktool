import numpy as np


def get_matching_df(reconstruction_df, particles_df, min_track_length=1, min_particle_length=1):
    # Get track lengths
    candidate_lengths = (
        reconstruction_df.track_id.value_counts(sort=False)
        .reset_index()
        .rename(columns={"index": "track_id", "track_id": "n_reco_hits"})
    )

    # Get true track lengths
    particle_lengths = (
        reconstruction_df.drop_duplicates(subset=["hit_id"])
        .particle_id.value_counts(sort=False)
        .reset_index()
        .rename(columns={"index": "particle_id", "particle_id": "n_true_hits"})
    )

    spacepoint_matching = (
        reconstruction_df.groupby(["track_id", "particle_id"])
        .size()
        .reset_index()
        .rename(columns={0: "n_shared"})
    )

    spacepoint_matching = spacepoint_matching.merge(candidate_lengths, on=["track_id"], how="left")
    spacepoint_matching = spacepoint_matching.merge(
        particle_lengths, on=["particle_id"], how="left"
    )
    spacepoint_matching = spacepoint_matching.merge(particles_df, on=["particle_id"], how="left")

    # Filter out tracks with too few shared spacepoints
    spacepoint_matching["is_matchable"] = spacepoint_matching.n_reco_hits >= min_track_length
    spacepoint_matching["is_reconstructable"] = (
        spacepoint_matching.n_true_hits >= min_particle_length
    )

    return spacepoint_matching


def calculate_matching_fraction(spacepoint_matching_df):
    spacepoint_matching_df = spacepoint_matching_df.assign(
        purity_reco=np.true_divide(
            spacepoint_matching_df.n_shared, spacepoint_matching_df.n_reco_hits
        )
    )
    spacepoint_matching_df = spacepoint_matching_df.assign(
        eff_true=np.true_divide(
            spacepoint_matching_df.n_shared, spacepoint_matching_df.n_true_hits
        )
    )

    return spacepoint_matching_df
