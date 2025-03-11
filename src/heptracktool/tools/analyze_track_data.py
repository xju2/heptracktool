"""Analyze the track data handled by a data reader"""

import matplotlib.pyplot as plt
import pandas as pd

from heptracktool.utils.utils_plot import add_mean_std, create_figure
from heptracktool.io.base import BaseTrackDataReader


class TrackAnalyzer:
    def __init__(self, reader: BaseTrackDataReader):
        self.reader = reader

    def study_cluster_features(self, evtid: int = 0):
        """Study the cluster features"""
        self.reader.read(evtid)
        if self.reader.clusters is None:
            print(f"No cluster information found in {self.reader.name} for event {evtid}.")

        clusters = self.reader.clusters
        print("Number of clusters: ", clusters.shape[0])
        pixel_clusters = clusters[clusters["hardware"] == "PIXEL"]
        strip_clusters = clusters[clusters["hardware"] == "STRIP"]
        print("Number of pixel clusters: ", pixel_clusters.shape[0])
        print("Number of strip clusters: ", strip_clusters.shape[0])

        # number of pixels per cluster
        _, ax = create_figure()
        config = dict(bins=31, range=(-0.5, 30.5), histtype="step", lw=2, alpha=0.8)
        ax.hist(pixel_clusters["pixel_count"], label="Pixel clusters", **config)
        ax.hist(strip_clusters["pixel_count"], label="Strip clusters", **config)
        ax.set_xlabel("# of Pixels per cluster")
        plt.legend()
        plt.show()

        # number of charges per cluster
        print("Charge information is not available for strip clusters")
        _, ax = create_figure()
        config = dict(bins=51, range=(-0.5, 50.5), histtype="step", lw=2)
        ax.hist(pixel_clusters["charge_count"], label="Pixel clusters", **config)
        ax.set_xlabel("# of Charges per cluster")
        add_mean_std(pixel_clusters["charge_count"], 15, 15000, ax, dy=2500)
        plt.legend()
        plt.show()

        # cluster position
        _, ax = create_figure()
        config = dict(bins=600, range=(-3000, 3000), histtype="step", lw=2, alpha=0.8)
        ax.hist(pixel_clusters["cluster_x"], label="pixel x", **config)
        ax.hist(pixel_clusters["cluster_y"], label="pixel y", **config)
        ax.hist(pixel_clusters["cluster_z"], label="pixel z", **config)
        ax.hist(strip_clusters["cluster_x"], label="strip x", **config, linestyle="--")
        ax.hist(strip_clusters["cluster_y"], label="strip y", **config, linestyle="--")
        ax.hist(strip_clusters["cluster_z"], label="strip z", **config, linestyle="--")
        ax.set_xlabel("Cluster position [mm]")
        plt.legend()
        plt.show()

        # # local directions of clusters
        # _, ax = create_figure()
        # config = dict(bins=50, range=(0, 30.0), histtype="step", lw=2, alpha=0.8)
        # ax.hist(pixel_clusters["localDir0"], label="pixel x", **config)
        # ax.hist(pixel_clusters["localDir1"], label="pixel y", **config)
        # ax.hist(pixel_clusters["localDir2"], label="pixel z", **config)
        # ax.hist(strip_clusters["localDir0"], label="strip x", **config, linestyle="--")
        # ax.hist(strip_clusters["localDir1"], label="strip y", **config, linestyle="--")
        # ax.hist(strip_clusters["localDir2"], label="strip z", **config, linestyle="--")
        # ax.set_xlabel("Local directions")
        # plt.legend()
        # plt.show()

        # eta/phi from local/global directions of clusters
        _, ax = create_figure("Pixel clusters")
        config = dict(bins=50, range=(-3.15, 3.15), histtype="step", lw=2, alpha=0.8)
        ax.hist(pixel_clusters["leta"], label=r"local $\eta$", **config)
        ax.hist(pixel_clusters["lphi"], label=r"local $\phi$", **config)
        ax.hist(pixel_clusters["glob_eta"], label=r"global $\eta$", **config, linestyle="--")
        ax.hist(pixel_clusters["glob_phi"], label=r"global $\phi$", **config, linestyle="--")
        ax.set_xlabel("Cluster Shapes")
        plt.legend()
        plt.show()

        # eta/phi from local/global directions of clusters
        _, ax = create_figure("Strip clusters")
        config = dict(bins=50, range=(-3.15, 3.15), histtype="step", lw=2, alpha=0.8)
        ax.hist(strip_clusters["leta"], label=r"local $\eta$", **config)
        ax.hist(strip_clusters["lphi"], label=r"local $\phi$", **config)
        ax.hist(strip_clusters["glob_eta"], label=r"global $\eta$", **config, linestyle="--")
        ax.hist(strip_clusters["glob_phi"], label=r"global $\phi$", **config, linestyle="--")
        ax.set_xlabel("Cluster Shapes")
        plt.legend()
        plt.show()

        # phi_angle and eta_angle
        _, ax = create_figure("Strip clusters")
        config = dict(bins=50, range=(0, 3.15), histtype="step", lw=2, alpha=0.8)
        ax.hist(pixel_clusters["phi_angle"], label=r"Pixel $\phi$", **config)
        ax.hist(pixel_clusters["eta_angle"], label=r"Pixel $\eta$", **config)
        ax.hist(strip_clusters["phi_angle"], label=r"Strip $\phi$", **config)
        ax.hist(strip_clusters["eta_angle"], label=r"Strip $\eta$", **config)
        ax.set_xlabel("Cluster Angles")
        plt.legend()
        plt.show()

    @classmethod
    def apply_eta_dep_cuts(
        cls,
        track: pd.DataFrame,
        eta_bins: tuple[float] = (-0.01, 2.0, 2.6, 4.0),
        min_hits: tuple[int] = (9, 8, 7),
        min_pT: tuple[float] = (900, 400, 400),  # MeV
        max_oot: tuple[float] = (10, 10, 10),
        chi2_ndof: tuple[float] = (7, 7, 7),
    ) -> pd.Series:
        """Apply eta dependent cuts.
        The default cuts are taken the ATLAS ITk performance paper.

        eta: eta of the track
        pT: pT of the track
        mot: measurements on track
        oot: outliers on track
        chi2_ndof: chi2/ndof of the track

        Return:
            pass_allcuts: boolean array indicating if the track passes all cuts
        """
        required_features = ["eta", "pt", "mot", "oot", "chi2_ndof"]
        assert all(
            f in track.columns for f in required_features
        ), f"Track data does not contain all required features: {required_features}"

        # apply eta dependent cuts on number of hits
        track = track.assign(abseta=track["eta"].abs())

        def apply_cuts(value: str, cut_list: list[float], cmp_opt: str = ">"):
            query = "|".join([
                f"((abseta > {eta_bins[idx]}) & (abseta <= {eta_bins[idx + 1]}) & ({value} {cmp_opt} {cut}))"
                for idx, cut in enumerate(cut_list)
            ])
            # print(query)
            return track.eval(query)

        num_hits_cuts = apply_cuts("mot", min_hits, "<")
        pt_cuts = apply_cuts("pt", min_pT, "<=")
        outlier_cuts = apply_cuts("oot", max_oot, ">")
        chi2_cuts = apply_cuts("chi2ndo", chi2_ndof, ">")

        num_failed_hits = track[num_hits_cuts].shape[0]
        num_failed_pt = track[pt_cuts].shape[0]
        num_failed_outliers = track[outlier_cuts].shape[0]
        num_failed_chi2 = track[chi2_cuts].shape[0]
        num_failed_all = track[num_hits_cuts | pt_cuts | outlier_cuts | chi2_cuts].shape[0]
        print("Total number of tracks: ", track.shape[0])
        print(
            "Number of tracks failed number of hits cut: ",
            num_failed_hits,
            f"({num_failed_hits / track.shape[0]:.2%})",
        )
        print(
            "Number of tracks failed pT cut: ",
            num_failed_pt,
            f"({num_failed_pt / track.shape[0]:.2%})",
        )
        print(
            "Number of tracks failed outlier cut: ",
            num_failed_outliers,
            f"({num_failed_outliers / track.shape[0]:.2%})",
        )
        print(
            "Number of tracks failed chi2 cut: ",
            num_failed_chi2,
            f"({num_failed_chi2 / track.shape[0]:.2%})",
        )
        print(
            "Number of tracks failed all cuts: ",
            num_failed_all,
            f"({num_failed_all / track.shape[0]:.2%})",
        )

        # number of hits vs |eta|
        _, ax = create_figure()
        ax.scatter(track.abseta, track.mot, alpha=0.5, s=10)

        def add_cut_lines(ax, cut_list: list[float], color="red"):
            if len(eta_bins) > 1:
                for idx, cut in enumerate(cut_list):
                    ax.plot([eta_bins[idx], eta_bins[idx + 1]], [cut, cut], color=color)
                for idx in range(1, len(eta_bins) - 1):
                    ax.plot(
                        [eta_bins[idx], eta_bins[idx]],
                        [cut_list[idx - 1], cut_list[idx]],
                        color=color,
                    )
            else:
                plt.axhline(cut_list[0], color=color)

        add_cut_lines(ax, min_hits)
        ax.set_xlabel(r"$|\eta|$")
        ax.set_ylabel("# of clusters")
        plt.show()

        # pT vs |eta|
        _, ax = create_figure()
        ax.scatter(track.abseta, track.pt, alpha=0.5, s=10)
        add_cut_lines(ax, min_pT)
        ax.set_xlabel(r"$|\eta|$")
        ax.set_ylabel("pT [MeV]")
        ax.set_ylim(0, 5000)
        plt.show()

        pass_allcuts = not (num_hits_cuts | pt_cuts | outlier_cuts | chi2_cuts)
        return pass_allcuts
