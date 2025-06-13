from __future__ import annotations
import re
import torch
from torch_geometric.data import Data
from pathlib import Path


def extract_variables(expr: str) -> set[str]:
    """
    Extract variables from the configuration dictionary.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        set: A set of variable names.
    """
    # Remove abs markers like |x| → just x
    expr = re.sub(r"\|(\w+)\|", r"\1", expr)

    # Find all variable-like words that are not numbers
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_-]*\b", expr)

    # Remove common keywords or known operators if needed
    return sorted(set(tokens))


class EdgeWeighting:
    """
    Assign weights to edges based on a YAML configuration or a dictionary.


    """

    def __init__(self, config_file: dict | str):
        super().__init__()

        self.edge_weight_config: dict[str, float] = {}

        if isinstance(config_file, str):
            self.edge_weight_config = self._load_config(config_file)
        elif isinstance(config_file, dict):
            self.edge_weight_config = config_file
        else:
            raise ValueError("config_file must be a dictionary or a string.")

        # get the variables used in the config
        self.variables = set()
        for conditions in self.edge_weight_config.keys():
            # Extract the variables from the condition
            self.variables.update(extract_variables(conditions))

    def _load_config(self, config_file: str) -> dict:
        # Load the configuration from a YAML file
        import yaml

        with Path(config_file).open() as file:
            config = yaml.safe_load(file)
        return config

    def __call__(self, batch: Data, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Assign weights to edges based on the configuration.

        Args:
            batch (Data): The input batch of data.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            torch.Tensor: The edge weights.
        """
        # verify if all variables are present in the batch.
        for var in self.variables:
            if not hasattr(batch, var):
                raise ValueError(f"Variable '{var}' not found in the batch.")

        edge_weights = torch.ones(edge_index.shape[1], dtype=torch.float32)
        for conditions, weight in self.edge_weight_config.items():
            # Evaluate the condition
            condition = eval(conditions, {}, {var: getattr(batch, var) for var in self.variables})
            # Assign the weight to the edges that satisfy the condition
            edge_weights[condition] = weight
