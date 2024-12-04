import paddle
"""Utility functions for graphs."""
from typing import List, Optional
import numpy as np


def get_edges_index(combined_mask: paddle.Tensor, pos: Optional[paddle.
    Tensor]=None, edge_cutoff: Optional[float]=None, remove_self_edge: bool
    =False) ->paddle.Tensor:
    """

    Args:
        combined_mask (Tensor): Combined mask for all fragments.
            Edges are built for nodes with the same indexes in the mask.
        pos (Optional[Tensor]): 3D coordinations of nodes. Defaults to None.
        edge_cutoff (Optional[float]): cutoff for building edges within a fragment.
            Defaults to None.
        remove_self_edge (bool): whether to remove self-connecting edge (i.e., ii).
            Defaults to False.

    Returns:
        Tensor: [2, n_edges], i for node index.
    """
    adj = combined_mask[:, None] == combined_mask[None, :]
    if edge_cutoff is not None:
        adj = adj & (paddle.cdist(x=pos, y=pos) <= edge_cutoff)
    if remove_self_edge:
        adj = adj.fill_diagonal_(value=False)
    edges_i, edges_j = paddle.where(adj)
    edges_i = edges_i.squeeze(axis=1)
    edges_j = edges_j.squeeze(axis=1)
    edges = paddle.stack(x=(edges_i, edges_j), axis=0)
    return edges


def get_subgraph_mask(edge_index: paddle.Tensor, n_frag_switch: paddle.Tensor
    ) ->paddle.Tensor:
    """Filter out edges that have inter-fragment connections.
    Example:
    edge_index: [
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1],
        ]
    n_frag_switch: [0, 0, 1]
    -> [1, 0, 1, 0, 0, 0]

    Args:
        edge_index (Tensor): e_ij
        n_frag_switch (Tensor): fragment that a node belongs to

    Returns:
        Tensor: [n_edge], 1 for inner- and 0 for inter-fragment edge
    """
    subgraph_mask = paddle.zeros(shape=edge_index.shape[1]).astype(dtype=
        'int64')
    in_same_frag = n_frag_switch[edge_index[0]] == n_frag_switch[edge_index[1]]
    subgraph_mask[paddle.where(in_same_frag)] = 1
    return subgraph_mask #.to(edge_index.place)


def get_n_frag_switch(natm_list: List[paddle.Tensor]) ->paddle.Tensor:
    """Get the type of fragments to which each node belongs
    Example: [Tensor(1, 1), Tensor(2, 1)] -> [0, 0, 1, 1 ,1]

    Args:
        natm_list (List[Tensor]): [Tensor([number of atoms per small fragment])]

    Returns:
        Tensor: [n_nodes], type of fragment each node belongs to
    """
    shapes = [tuple(natm.shape)[0] for natm in natm_list]
    assert np.std(shapes
        ) == 0, 'Tensor must be the same length for <natom_list>'
    n_frag_switch = paddle.repeat_interleave(x=paddle.arange(end=len(
        natm_list)), repeats=paddle.to_tensor(data=[paddle.sum(x=natm).item
        () for natm in natm_list], place=natm_list[0].place))
    return n_frag_switch #.to(natm_list[0].place)


def get_mask_for_frag(natm: paddle.Tensor) ->paddle.Tensor:
    """Get fragment index for each node
    Example: Tensor([2, 0, 3]) -> [0, 0, 2, 2, 2]

    Args:
        natm (Tensor): number of nodes per small fragment

    Returns:
        Tensor: [n_node], the natural index of fragment a node belongs to
    """
    return paddle.repeat_interleave(x=paddle.arange(end=natm.shape[0]),
        repeats=natm) #.to(natm.place)


def get_inner_edge_index(subgraph_mask: paddle.Tensor):
    return paddle.stack(x=paddle.where(subgraph_mask), axis=0)
