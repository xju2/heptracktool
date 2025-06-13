from typing import Union
from loguru import logger


import torch
from torch_geometric.nn import radius

try:
    import frnn

    FRNN_AVAILABLE = True
    logger.warning("FRNN is available")
except ImportError:
    FRNN_AVAILABLE = False
    logger.warning(
        "FRNN is not available, install it at https://github.com/murnanedaniel/FRNN. Using PyG radius instead."
    )

faiss_avail = False
try:
    import faiss

    faiss_avail = True
    logger.warning("FAISS is available")
except ImportError:
    faiss_avail = False
    logger.warning(
        'FAISS is not available, install it at "conda install faiss-gpu -c pytorch" or \
                    "pip install faiss-gpu". Using PyG radius instead.'
    )

if not torch.cuda.is_available():
    FRNN_AVAILABLE = False
    logger.warning("FRNN is not available, as no GPU is available")


# ---------------------------- Dataset Processing -------------------------

# ---------------------------- Edge Building ------------------------------


def build_edges(
    embedding: torch.Tensor,
    indices: torch.Tensor | None = None,
    r_max: float = 1.0,
    k_max: int = 10,
    return_indices: bool = False,
    backend: str = "FRNN",
    nlist: int = 100,
    nprobe: int = 5,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    backend = backend.lower() if backend is not None else backend
    # Type hint
    if backend == "frnn" and FRNN_AVAILABLE:
        embedding = embedding.cuda()
        # Compute edges
        dists, idxs, _, _ = frnn.frnn_grid_points(
            points1=embedding.unsqueeze(0),
            points2=embedding.unsqueeze(0),
            lengths1=None,
            lengths2=None,
            K=k_max,
            r=r_max,
            grid=None,
            return_nn=False,
            return_sorted=True,
        )

        idxs: torch.Tensor = idxs.squeeze().int()
        ind = torch.arange(idxs.shape[0], device=embedding.device).repeat(idxs.shape[1], 1).T.int()
        positive_idxs = idxs >= 0
        edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
    elif faiss_avail and "faiss" in backend:
        emebdding_array = embedding.detach().cpu().numpy()
        if backend == "faiss-cpu-ivf":
            quantizer = faiss.IndexFlatL2(embedding.shape[1])
            index = faiss.IndexIVFFlat(quantizer, embedding.shape[1], nlist, faiss.METRIC_L2)
            index.train(emebdding_array)
            # default # of probes is 1
            index.nprobe = nprobe
            index.add(emebdding_array)
            dists, idxs = index.search(emebdding_array, k_max)
        elif backend == "faiss-cpu-flatl2":
            index_flat = faiss.IndexFlatL2(embedding.shape[1])
            index_flat.add(emebdding_array)
            dists, idxs = index_flat.search(emebdding_array, k_max)
        elif backend == "faiss-cpu-scalarquantizer":
            index = faiss.IndexScalarQuantizer(embedding.shape[1], faiss.METRIC_L2)
            index.train(emebdding_array)
            index.add(emebdding_array)
            dists, idxs = index.search(emebdding_array, k_max)
        elif backend == "faiss-gpu" and torch.cuda.is_available():  # GPU version
            res = faiss.StandardGpuResources()
            index_flat = faiss.GpuIndexFlatL2(res, embedding.shape[1])
            index_flat.add(emebdding_array)
            dists, idxs = index_flat.search(emebdding_array, k_max)
        elif (
            backend == "faiss-gpu-quantized" and torch.cuda.is_available()
        ):  # GPU version of quantized, not work
            raise NotImplementedError("GPU version of quantized is not implemented yet")
            # res = faiss.StandardGpuResources()
            # index = faiss.GpuIndexIVFFlat(
            #     res, embedding.shape[1], nlist, faiss.METRIC_L2
            # )
            # print("after gpu index")
            # index.train(emebdding_array)
            # print("after train")
            # # default # of probes is 1
            # # index.nprobe = nprobe

            # index.add(emebdding_array)
            # print("after add")
            # dists, idxs = index.search(emebdding_array, k_max)
            # print("after search")
        else:
            raise ValueError(
                f"faiss is available, but the mode {backend} is not supported,"
                "please chose faiss-cpu, faiss-cpu-scalarquantizer, or faiss-gpu"
            )

        dists, idxs = torch.from_numpy(dists), torch.from_numpy(idxs)
        ind = torch.arange(idxs.shape[0], device=embedding.device).repeat(idxs.shape[1], 1).T.int()
        positive_idxs = dists <= r_max**2
        edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
    else:
        edge_list = radius(embedding, embedding, r=r_max, max_num_neighbors=k_max)

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return (edge_list, dists, idxs, ind) if (return_indices and backend == "FRNN") else edge_list


def graph_intersection(
    edge_index: torch.Tensor,
    true_edges: torch.Tensor,
    return_truth_to_pred: bool = False,
) -> torch.Tensor:
    """
    Use sparse representation to compare the predicted graph
    and the truth graph so as to label the edges in the predicted graph
    to be 1 as true and 0 as false.
    """
    num_reco_edges = edge_index.shape[1]
    # check if the two tensors in the same device. If not move the truth edges to the same device
    device = edge_index.device
    if device != true_edges.device:
        true_edges = true_edges.to(device)

    # unique edges in the union of the predicted and truth edges
    # inverse is the index of the unique edges in the union
    unique_edges, inverse = torch.unique(
        torch.cat([edge_index, true_edges], dim=1),
        dim=1,
        return_inverse=True,
    )

    # among the unique edges, which exit in the predicted graph
    inverse_pred_map = torch.ones_like(unique_edges[1]).to(device) * -1
    inverse_pred_map[inverse[:num_reco_edges]] = torch.arange(num_reco_edges, device=device)

    # among the unique edges, which exit in the truth graph
    inverse_truth_map = torch.ones_like(unique_edges[1]).to(device) * -1
    inverse_truth_map[inverse[num_reco_edges:]] = torch.arange(true_edges.shape[1], device=device)

    # which edges in the predicted graph are also in the truth graph
    edge_index_y = inverse_truth_map[inverse[:num_reco_edges]] >= 0

    return_tensors = [edge_index_y]
    if return_truth_to_pred:
        truth_to_pred = inverse_pred_map[inverse[num_reco_edges:]]
        return_tensors.append(truth_to_pred)

    return return_tensors if len(return_tensors) > 1 else return_tensors[0]
