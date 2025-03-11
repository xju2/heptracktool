# 3rd party imports
from pytorch_lightning import LightningModule
import torch.nn.functional as F

# Local imports
from heptracktool.models.model_utils import make_mlp


class MetricLearning(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        in_channels = len(hparams["node_features"])

        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters(hparams)

    def forward(self, x):
        x_out = self.network(x)
        return F.normalize(x_out)
