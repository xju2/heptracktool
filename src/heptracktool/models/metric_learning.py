# 3rd party imports
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch_geometric.data import Data

# Local imports
from heptracktool.utils.utils_graph import build_edges
from heptracktool.utils.utils_graph import graph_intersection


class MetricLearning(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        r_max: float = 0.1,
        k_max: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model"],
        )
        self.network = model
        self.delta_r = 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.network(x)
        return F.normalize(x_out)

    def configure_optimizers(self):
        opt = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            sch = self.hparams.scheduler(optimizer=opt)
            return (
                {
                    "optimizer": opt,
                    "lr_scheduler": {
                        "scheduler": sch,
                        "monitor": "val_loss",
                        "interval": "step",
                        "frequency": self.trainer.val_check_interval,
                        "strict": True,
                        "name": "LRScheduler",
                    },
                },
            )
        return {"optimizer": opt, "frequency": 1}

    def on_train_batch_start(self, batch: Data, batch_idx: int) -> None:
        # prepare the edge list for the current batch.
        # and their labels.

        assert hasattr(batch, "track_edges")
        assert hasattr(batch, "x")

        knn_edges = self._get_knn_edges(batch)
        edge_list = torch.cat([batch.track_edges, knn_edges], dim=1)
        self._order_edge_direction(batch.R, edge_list)

        # remove duplicate edges.
        edge_list = torch.unique(edge_list, dim=1)

        # randomly shuffle the edges
        edge_list = edge_list[:, torch.randperm(edge_list.shape[1])]

        # create the labels for the edges
        # and truth to predict labels.
        edge_y, truth_map = graph_intersection(
            edge_list,
            batch.track_edges,
            return_truth_to_pred=True,
        )

        batch.edge_index = edge_list
        batch.y = edge_y
        batch.truth_map = truth_map
        batch.edge_weights = self._get_edge_weights(batch)

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        embedding = self(batch.x)
        loss = self._compute_loss(batch, embedding).float()
        self.log("train_loss", loss, batch_size=1, prog_bar=True)
        return loss

    def _compute_loss(self, batch, embedding: torch.Tensor) -> torch.Tensor:
        # compute the distances between the nodes in the edge list
        pred_edges = batch.edge_index
        d = self._get_distances(embedding, pred_edges)

        # compute the loss
        y = batch.y
        pos_mask = y == 1
        neg_mask = y == 0

        pos_loss = torch.mean(d[pos_mask])
        neg_loss = torch.mean(F.relu(self.hparams.r_max - d[neg_mask]))

        loss = pos_loss + neg_loss
        return loss

    def _get_distances(self, embedding, pred_edges):
        reference = embedding[pred_edges[1]]
        neighbors = embedding[pred_edges[0]]
        d = torch.sum((reference - neighbors) ** 2, dim=-1)
        return d

    def _order_edge_direction(self, R: torch.Tensor, edge_list: torch.Tensor):
        # make sure edge (a, b) where a is always closer to beam spot than b

        mask = (R[edge_list[0]] > R[edge_list[1]]) | (
            R[edge_list[0]] == R[edge_list[1]] & (edge_list[0] > edge_list[1])
        )
        edge_list[:, mask] = edge_list[:, mask].flip(0)

    def _get_knn_edges(self, batch: Data) -> torch.Tensor:
        """Get knn edges for the current batch and return them with their labels."""

        self.eval()
        # create hard negative edges using FRNN.
        with torch.no_grad():
            embedding = self(batch.x)

        knn_edges = build_edges(
            embedding,
            embedding,
            r_max=self.hparams.r_max + self.delta_r,
            k_max=self.hparams.k_max,
            backend="FRNN",
        )
        self.train()
        return knn_edges

    def _get_edge_weights(self, batch: Data) -> torch.Tensor:
        """Get the weights for the edges in the batch."""
        assert hasattr(batch, "edge_index")
        assert hasattr(batch, "y")

        # get the weights for the edges
        edge_weights = torch.ones(batch.edge_index.shape[1])
        edge_weights[batch.y == 0] = self.hparams.r_max

        return edge_weights
