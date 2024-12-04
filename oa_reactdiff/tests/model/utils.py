import sys
sys.path.append('/root/ssd3/zhangzhimin04/workspaces_11.6/OAReactDiff_Paddle/utils'
    )
import paddle_aux
import paddle
"""Ultility functions used in test cases."""
egnn_config = dict(in_node_nf=8, in_edge_nf=5, hidden_nf=64, edge_hidden_nf
    =64, act_fn='swish', n_layers=6, attention=True, out_node_nf=None, tanh
    =True, coords_range=15.0, norm_constant=1.0, inv_sublayers=2,
    sin_embedding=False, normalization_factor=1.0, aggregation_method='mean')
left_config = dict(pos_require_grad=False, cutoff=20.0, num_layers=6,
    hidden_channels=32, num_radial=32, in_node_nf=8, reflect_equiv=True)


def tensor_relative_diff(x1, x2):
    return paddle.max(x=paddle.abs(x=x1 - x2) / (x1 + x2 + 1e-06) * 2)


def init_weights(m):
    """Weight initialization for all MLP.

    Args:
        m: a nn.Module
    """
    if isinstance(m, paddle.nn.Linear):
        gain = 1.0
        init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=gain)
        init_XavierUniform(m.weight)
        if m.bias is not None:
            init_Uniform = paddle.nn.initializer.Uniform(low=-gain, high=gain)
            init_Uniform(m.bias)


def generate_full_eij(n_atom: int):
    """Get fully-connected graphs for n_atoms."""
    edge_index = []
    for ii in range(n_atom):
        for jj in range(n_atom):
            if ii != jj:
                edge_index.append([ii, jj])
    return paddle.transpose(x=paddle.to_tensor(data=edge_index), perm=
        paddle_aux.transpose_aux_func(paddle.to_tensor(data=edge_index).
        ndim, 1, 0)).astype(dtype='int64')


def get_cut_graph_mask(edge_index, n_cut):
    """Get mask for a graph cut at n_cut, with ij representing cross-subgraph edgs being 0."""
    ind_sum = paddle.where(condition=edge_index < n_cut, x=1, y=0).sum(axis=0)
    subgraph_mask = paddle.zeros(shape=edge_index.shape[1]).astype(dtype=
        'int64')
    subgraph_mask[ind_sum == 2] = 1
    subgraph_mask[ind_sum == 0] = 1
    subgraph_mask = subgraph_mask[:, None]
    return subgraph_mask
