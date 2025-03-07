"""This module provides a class that takes following parameters as inputs
* model_path: path to the trained model
* data_path: path to the data to be used for inference
* output_path: path to the output file
* kwargs: other parameters that are needed for inference
"""

from typing import Optional
from pathlib import Path
import numpy as np
import psutil
import datetime

from heptracktool.io.pyg_data_reader import TrackGraphDataReader
from heptracktool.tools.utils_graph import build_edges
from heptracktool.tools.edge_perf import EdgePerformance

import yaml
import torch
from torch import Tensor


def batched_inference(model, senders, receivers, batch_size: int = 1024, debug: bool = False):
    results = []
    n_batches = int(np.ceil(senders.shape[0] / batch_size))
    if debug:
        print(f"processing {n_batches:,} batches")
    for i in range(n_batches):
        if debug and i % 1000 == 0:
            print(f"processing batch {i:,}")
        if i == n_batches - 1:
            batch_senders = senders[i * batch_size :]
            batch_receivers = receivers[i * batch_size :]
        else:
            batch_senders = senders[i * batch_size : (i + 1) * batch_size]
            batch_receivers = receivers[i * batch_size : (i + 1) * batch_size]

        with torch.no_grad():
            batch_edge_scores = model.forward(batch_senders, batch_receivers).detach()
            results.append(batch_edge_scores)

    results = torch.cat(results, dim=0)
    return results


class ModelLoader:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        print("Loading model from", self.model_path)
        self.model = torch.jit.load(self.model_path)
        self.model.eval()

    def predict(self, *arg) -> Tensor:
        return self.model.forward(*arg)


class TorchModelInference:
    def __init__(
        self,
        config_fname: str,
        data_type: str,
        model_path: str,
        output_path: Optional[str] = None,
        name="TorchModelInference",
    ) -> None:
        self.name = name
        # data type can be trainset, valset, or testset
        assert data_type in [
            "trainset",
            "valset",
            "testset",
        ], "data_type must be trainset, valset, or testset"

        with open(config_fname) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = config

        self.model_reader = ModelLoader(model_path)
        self.model_reader.load()

        data_path = Path(config["input_dir"])
        self.data_reader = TrackGraphDataReader(data_path / data_type, name=data_type)

        self.output_path = output_path
        self.stage_name = config["stage"]

        self.edge_perf = EdgePerformance(self.data_reader)

    def inference(
        self,
        evtid: int,
        radius: float = 0.1,
        knn: int = 1000,
        knn_backend: Optional[str] = None,
    ) -> Tensor:
        self.data_reader.read(evtid)  # tell the reader to read the event
        if self.stage_name == "graph_construction":
            node_features = self.config["node_features"]
            node_scales = torch.Tensor(self.config["node_scales"])

            features = self.data_reader.get_node_features(node_features, node_scales)
            # print("model is at:", self.model_reader.model.device, "and input data is at:", features.device)
            embedding = self.model_reader.predict(features)

            edge_index = build_edges(embedding, r_max=radius, k_max=knn, backend=knn_backend)
            (
                truth_labels,
                true_edges,
                _,
                _,
            ) = self.edge_perf.eval(edge_index)

            return (
                edge_index,
                truth_labels,
                true_edges,
            )
        else:
            pass


class ExaTrkxInference:
    def __init__(
        self,
        config_fname: str,
        data_path: str,
        device: str = "cpu",
        name="ExaTrkxInference",
    ) -> None:
        self.name = name
        self.device = device

        with open(config_fname) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = config
        self.process = psutil.Process()
        self.system_data = []

        self.data_reader = TrackGraphDataReader(data_path, name=self.name + "DataReader")
        self.edge_perf = EdgePerformance(self.data_reader)

        e_config = config["embedding"]
        self.embedding_model = torch.jit.load(e_config["model_path"], map_location=self.device)
        self.embedding_model.eval()
        self.embedding_model.hparams = e_config

        f_config = config["filtering"]
        self.filtering_model = torch.jit.load(f_config["model_path"], map_location=self.device)
        self.filtering_model.eval()
        self.filtering_model.hparams = f_config

        g_config = config.get("gnn", None)
        self.gnn_model = None
        if g_config is not None:
            self.gnn_model = torch.jit.load(g_config["model_path"], map_location=self.device)
            self.gnn_model.eval()
            self.gnn_model.hparams = g_config

    def _get_system_info(self):
        # return memory usage in MB
        # return cpu time in seconds
        results = {}
        process = self.process
        with process.oneshot():
            results["memory"] = int(process.memory_full_info().uss / 1024**2)
            results["cpu_time"] = int(process.cpu_times().system)
            results["datetime"] = datetime.datetime.now()
        return results

    def _add_system_data(self, tag_name: str = "start"):
        results = self._get_system_info()
        results["tag"] = tag_name
        self.system_data.append(results)

    def _print_memory_usage(self, tag_name: str):
        print(
            "{} {}, CPU memory usage: {:,} MB".format(
                datetime.datetime.now().strftime("%H:%M:%S"),
                tag_name,
                self._get_system_info()["memory"],
            )
        )

    def __call__(
        self, evtid, filter_cut: float = 0.57, save_output: bool = False, *args, **kwargs
    ):
        self._print_memory_usage("Start")

        _ = self.data_reader.read(evtid)
        self._print_memory_usage("After reading data")
        # embedding
        node_features = self.embedding_model.hparams["node_features"]
        node_scales = self.embedding_model.hparams["node_scales"]
        features = self.data_reader.get_node_features(node_features, node_scales)
        with torch.no_grad():
            embedding = self.embedding_model.forward(features).detach()
        self._print_memory_usage("After embedding")

        # FRNN
        r_max = self.embedding_model.hparams["r_infer"]
        k_max = self.embedding_model.hparams["knn_infer"]
        knn_backend = self.embedding_model.hparams["knn_backend"]
        edge_index = build_edges(embedding, r_max=r_max, k_max=k_max, backend=knn_backend)
        self._print_memory_usage("After FRNN")
        # edge-level performance
        (
            truth_labels,
            true_edges,
            per_edge_efficiency,
            per_edge_purity,
        ) = self.edge_perf.eval(edge_index)
        self._print_memory_usage("After Edge Evaluation")

        # fetching filtering node features
        node_features = self.filtering_model.hparams["node_features"]
        node_scales = self.filtering_model.hparams["node_scales"]
        features = self.data_reader.get_node_features(node_features, node_scales)
        self._print_memory_usage("After Retrieving Filtering Node features")

        # get the senders and recievers
        batch_size = self.filtering_model.hparams["batch_size"]
        senders, receivers = features[edge_index[0]], features[edge_index[1]]
        self._print_memory_usage("After Splitting Node features")

        # filtering inference
        filter_edge_scores = batched_inference(
            self.filtering_model, senders, receivers, batch_size=batch_size
        )
        self._print_memory_usage("After Filtering")
        # apply sigmoid
        filter_edge_scores = torch.sigmoid(filter_edge_scores)

        # weights = self.data_reader.data.weights if "weights" in self.data_reader.data else None
        # evaluate the edge scores
        filter_perf = self.edge_perf.eval_edge_scores(
            filter_edge_scores,
            truth_labels,
            outname=f"perf_filtering_evt{evtid}",
        )

        # obtain a threshold that gives 99% efficiency
        # threshold = filter_perf[-1]
        threshold = filter_cut
        filtering_selections = filter_edge_scores > threshold
        edge_index = edge_index[:, filtering_selections]
        # update the truth labels
        truth_labels = truth_labels[filtering_selections]
        print(f"After filtering, {edge_index.shape[1]:,} edges remain")

        # apply the GNN to the filtered edges
        if self.gnn_model is not None:
            node_features = self.gnn_model.hparams["node_features"]
            node_scales = self.gnn_model.hparams["node_scales"]
            features = self.data_reader.get_node_features(node_features, node_scales)
            self._print_memory_usage("After Retrieving GNN Node features")

            # get the senders and recievers
            senders, receivers = features[edge_index[0]], features[edge_index[1]]
            self._print_memory_usage("After Splitting Node features")

            # gnn inference
            gnn_edge_scores = self.gnn_model.forward(senders, receivers)
            self._print_memory_usage("After GNN")
            # apply sigmoid
            gnn_edge_scores = torch.sigmoid(gnn_edge_scores)

            # evaluate the edge scores
            gnn_perf = self.edge_perf.eval_edge_scores(
                gnn_edge_scores, truth_labels, outname=f"perf_gnn_evt{evtid}"
            )

        return dict(
            edge_index=edge_index,
            truth_labels=truth_labels,
            true_edges=true_edges,
            filter_edge_scores=filter_edge_scores,
            per_edge_efficiency_embedding=per_edge_efficiency,
            per_edge_purity_embedding=per_edge_purity,
            filter_perf=filter_perf,
            gnn_perf=gnn_perf,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ExaTrkX Inference")
    add_arg = parser.add_argument
    add_arg("config", help="configuration file")
    add_arg("data", help="data path")

    args = parser.parse_args()
    inf = ExaTrkxInference(args.config, args.data)
    results = inf(0)
