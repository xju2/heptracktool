from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = False,
        is_causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model (embedding dimension) must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.dropout = dropout

        self.c_attn = nn.Linear(d_model, d_model * 3, bias=bias)

        # output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, dim=-1)

        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal
        )

        y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        y = self.c_proj(y)

        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, dropout_attn=0.0, **kwargs):
        super().__init__()

        self.attn = CausalSelfAttention(d_model, num_heads, dropout=dropout_attn, **kwargs)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        attn_output = self.attn(x, mask=mask)
        x = x + self.dropout(attn_output)
        x = self.ln1(x)

        ff_output = self.mlp(x)
        x = x + self.dropout(ff_output)
        x = self.ln2(x)
        return x


@torch.jit.script
def sliding_window_torch_multi(
    arr: torch.Tensor, L: int, S: int, dim: int = 0, pad_num: float = 0.0
) -> torch.Tensor:
    """
    Splits a multi-dimensional PyTorch tensor into overlapping chunks along a given dimension,
    padding with zeros if necessary.

    Args:
        arr (torch.Tensor): The input tensor (can be multi-dimensional).
        L (int): Length of each chunk.
        S (int): Step size (stride).
        dim (int): Dimension along which to apply the sliding window.

    Returns:
        torch.Tensor: Overlapping chunks of shape (..., num_chunks, L, ...).
    """
    shape = list(arr.shape)
    original_length = shape[dim]

    if L > original_length:  # Handle the case where the window is larger than the dimension
        return arr

    # Compute padding size
    remainder = (original_length - S) % L
    if remainder != 0:
        pad_size = L - remainder
        pad_shape = shape.copy()
        pad_shape[dim] = pad_size
        pad_tensor = torch.full(pad_shape, pad_num, dtype=arr.dtype, device=arr.device)
        arr = torch.cat([arr, pad_tensor], dim=dim)

    # Apply sliding window using unfold
    return arr.unfold(dim, L, S)


class MetricLearningTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        num_transformer_layers: int,
        max_sequence_len: int,
        slice_length: int,
        reduce: str = "mean",
        bias: bool = False,
        is_causal: bool = False,
        dropout: float = 0.1,
        dropout_attn: float = 0.0,
    ):
        super().__init__()

        # hyper parameters
        self.out_dim = out_dim
        self.max_sequence_len = max_sequence_len
        self.slice_length = slice_length
        self.reduce = reduce

        # model layers
        self.in_proj = nn.Linear(in_dim, d_model)

        self.num_transformer_layers = num_transformer_layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, ff_dim, dropout, dropout_attn, bias=bias, is_causal=is_causal
            )
            for _ in range(num_transformer_layers)
        ])
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor):
        num_sp = x.size(0)  # use shape[0] would fail in TorchScript.

        # turn a sequence into a batch of overlapping sequences.
        batched = sliding_window_torch_multi(
            x, self.max_sequence_len, self.slice_length
        ).transpose(1, 2)

        batched = self.in_proj(batched)
        for layer in self.transformer_layers:
            batched = layer(batched)
        y = self.out_proj(batched)

        # reduce the overlapping sequences back to a single sequence.
        device = x.device
        dtype = x.dtype
        x_indices = torch.arange(0, num_sp, device=x.device)
        batched_indices = sliding_window_torch_multi(
            x_indices, self.max_sequence_len, self.slice_length, pad_num=-1.0
        )
        batched_indices = batched_indices.reshape(-1)

        # remove the padding
        non_pad_mask = batched_indices != -1
        batched_indices = batched_indices[non_pad_mask]
        y = y.reshape(-1, self.out_dim)[non_pad_mask]

        y = torch.scatter_reduce(
            torch.zeros(num_sp, self.out_dim, device=device, dtype=dtype),
            0,
            batched_indices.unsqueeze(1).expand(-1, self.out_dim).to(device),
            y,
            reduce=self.reduce,
            include_self=False,
        )
        return y
