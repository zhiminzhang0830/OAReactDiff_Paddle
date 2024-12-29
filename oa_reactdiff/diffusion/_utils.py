import sys
sys.path.append('/root/ssd3/zhangzhimin04/workspaces_11.6/OAReactDiff_Paddle/utils'
    )
import paddle_aux
import paddle
from typing import List
import math
import numpy as np
from oa_reactdiff.model.scatter.scatter import scatter

def remove_mean_batch(x, indices):
    mean = scatter(x, indices, dim=0, reduce='mean')
    x = x - mean[indices]
    return x


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    largest_value = x.abs().max().item()
    error = scatter(x, node_mask, dim=0, reduce='sum').abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 0.01, f'Mean is not zero, relative_error {rel_error}'


def sample_center_gravity_zero_gaussian_batch(size: List[int], indices:
    List[paddle.Tensor]) ->paddle.Tensor:
    assert len(size) == 2
    x = paddle.randn(shape=size)

    # The following code is for precision alignment with torch code
    # x = np.random.randn(*size)
    # x = x.astype('float32')
    # x = paddle.to_tensor(x)

    x_projected = remove_mean_batch(x, paddle.concat(x=indices))
    return x_projected


def sum_except_batch(x, indices, dim_size):
    return scatter(x.sum(axis=-1), indices, dim=0, dim_size=dim_size, reduce='sum')


def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + paddle.erf(x=x / math.sqrt(2)))

def sample_gaussian(size, device):
    x = paddle.randn(shape=size)

    # The following code is for precision alignment with torch code
    # x = np.random.randn(*size)
    # x = x.astype('float32')
    # x = paddle.to_tensor(x)
    return x


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples
    if isinstance(num_nodes, paddle.Tensor):
        num_nodes = num_nodes.to(device)
    sample_inds = paddle.arange(end=n_samples)
    return paddle.repeat_interleave(x=sample_inds, repeats=num_nodes)
