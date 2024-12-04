import sys
sys.path.append('/root/ssd3/zhangzhimin04/workspaces_11.6/OAReactDiff_Paddle/utils'
    )
import paddle_aux
import paddle
"""Utility functions for model"""


def move_by_com(pos):
    return pos - paddle.mean(x=pos, axis=0)


def coord2cross(x, edge_index, norm_constant=1):
    row, col = edge_index
    cross = paddle.cross(x=x[row], y=x[col], axis=1)
    norm = paddle.linalg.norm(x=cross, axis=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = paddle.sum(x=coord_diff ** 2, axis=1).unsqueeze(axis=1)
    norm = paddle.sqrt(x=radial + 1e-08)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments,
    normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = num_segments, data.shape[1]
    result = paddle.full(shape=result_shape, fill_value=0, dtype=data.dtype)
    segment_ids = segment_ids.unsqueeze(axis=-1).expand(shape=[-1, data.
        shape[1]])
    result.put_along_axis_(axis=0, indices=segment_ids, values=data, reduce
        ='add')
    if aggregation_method == 'sum':
        result = result / normalization_factor
    if aggregation_method == 'mean':
        norm = paddle.zeros(shape=tuple(result.shape), dtype=data.dtype)
        norm.put_along_axis_(axis=0, indices=segment_ids, values=paddle.
            ones(shape=tuple(data.shape), dtype=data.dtype), reduce='add')
        norm[norm == 0] = 1
        result = result / norm
    return result


def get_ji_bond_index(bond_atom_indices: paddle.Tensor) ->paddle.Tensor:
    """Get the index for e_ji
    for example, bond_atom_indices = [[0, 1], [1, 0]], returns [1, 0]

    Args:
        bond_atom_indices (Tensor): (2, n_bonds) for ij

    Returns:
        Tensor: index for ji
    """
    bond_atom_indices = paddle.transpose(x=bond_atom_indices, perm=
        paddle_aux.transpose_aux_func(bond_atom_indices.ndim, 0, 1))
    _index = paddle.to_tensor(data=[1, 0], dtype='int64')
    reverse_bond_atom_indices = bond_atom_indices[:, _index]
    bond_ji_index = []
    for ij in range(tuple(bond_atom_indices.shape)[0]):
        bond_ji_index.append(paddle.where((bond_atom_indices ==
            reverse_bond_atom_indices[ij]).astype('bool').all(axis=1))[0])
    return paddle.concat(x=bond_ji_index).astype(dtype='int64')


def symmetrize_edge(edge_attr: paddle.Tensor, edge_ji_indices: paddle.Tensor
    ) ->paddle.Tensor:
    return (edge_attr + edge_attr[edge_ji_indices]) / 2
