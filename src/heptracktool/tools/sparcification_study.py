from heptracktool.models.metric_learning import MetricLearning
from heptracktool.tools.utils_graph import build_edges
from heptracktool.io.pyg_data_reader import TrackGraphDataReader
from heptracktool.tools.edge_perf import EdgePerformance
import torch
from loguru import logger
from typing import Any
import numpy as np

import scipy.sparse as sp
from scipy.io import mmwrite, mmread


class SparcifyStudy:
    def __init__(
        self,
        metric_learning_ckpt_filename,
        pyg_reader: TrackGraphDataReader,
        kval: int | None = None,
        rval: float | None = None,
        device: str | None = None,
    ):
        self.model = MetricLearning.load_from_checkpoint(metric_learning_ckpt_filename)
        self.reader = pyg_reader
        self.edge_perf = EdgePerformance(self.reader)

        self.node_features = self.model.hparams["node_features"]
        self.node_scales = self.model.hparams["node_scales"]

        self.kval = self.model.hparams["knn_val"] if kval is None else kval
        self.rval = self.model.hparams["r_train"] if rval is None else rval
        self.device = (
            device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"# of node features: {len(self.node_features)}")
        logger.info(f"Node features: {self.node_features}")
        logger.info(f"Node scales: {self.node_scales}")
        logger.info(f"KNN neighbours: {self.kval}")
        logger.info(f"KNN radius: {self.rval}")

    def build_knn_graph(
        self, input_pyg_filename: str, kval: int | None = None, rval: float | None = None
    ) -> sp.coo_matrix:
        self.reader.read_by_filename(input_pyg_filename)

        kval = kval if kval is not None else self.kval
        rval = rval if rval is not None else self.rval

        in_features = self.reader.get_node_features(self.node_features, self.node_scales).to(
            self.device
        )

        num_nodes = in_features.size(0)
        with torch.no_grad():
            embedding = self.model(in_features)
        knn_backend = "FRNN" if self.device == "cuda" else "torch"
        edge_index = build_edges(embedding, None, rval, kval, backend=knn_backend)

        # sort the edges and remove duplicated.
        edge_index[:, edge_index[0] > edge_index[1]] = edge_index[
            :, edge_index[0] > edge_index[1]
        ].flip(0)
        edge_index = torch.unique(edge_index, dim=-1)

        # Treat the embedding distances between the edges as weights
        weights = torch.norm(embedding[edge_index[0]] - embedding[edge_index[1]], dim=1)
        sparse_matrix = sp.coo_matrix(
            (weights.cpu().numpy(), edge_index.cpu().numpy()), shape=(num_nodes, num_nodes)
        )
        return sparse_matrix

    def eval_sparcification(
        self, sp_coo_matrix: sp.coo_matrix, input_pyg_filename: str | None = None
    ) -> Any:
        if input_pyg_filename is not None:
            self.reader.read_by_filename(input_pyg_filename)
        elif self.reader.data is None:
            raise ValueError(
                "No data loaded. Please run self.reader.read_by_filename(filename) first."
            )

        array = np.array([sp_coo_matrix.row, sp_coo_matrix.col], dtype=int)
        edge_index = torch.from_numpy(array)
        # make the graph undirected.
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        print(f"Number of edges: {edge_index.size(1):,}")
        return self.edge_perf.eval(edge_index)

    def read_sparse_matrix(self, input_filename: str) -> sp.coo_matrix:
        return mmread(input_filename)

    def save_sparse_matrix(self, sparse_matrix: sp.coo_matrix, output_filename: str):
        mmwrite(output_filename, sparse_matrix)
