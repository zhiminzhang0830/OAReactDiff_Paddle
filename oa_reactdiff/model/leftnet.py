import sys
from utils import paddle_aux
import paddle
import math
from math import pi
from typing import Optional, Tuple, Callable
import numpy as np
# from torch_geometric.nn.conv import MessagePassing
from .message_passing.message_passing import MessagePassing
# from torch_scatter import scatter, scatter_mean
from .scatter.scatter import scatter
from oa_reactdiff.model.util_funcs import unsorted_segment_sum
from oa_reactdiff.model.core import MLP
EPS = 1e-06


def swish(x):
    return x * paddle.nn.functional.sigmoid(x=x)


def com(x):
    return x - paddle.mean(x=x, axis=0)


def remove_mean_batch(x, indices):
    mean = scatter(x, indices, dim=0, reduce='mean')
    x = x - mean[indices]
    return x


class RBFEmb(paddle.nn.Layer):
    """
    radial basis function to embed distances
    modified: delete cutoff with r
    """

    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()
        self.register_buffer(name='means', tensor=means)
        self.register_buffer(name='betas', tensor=betas)

    def _initial_params(self):
        start_value = paddle.exp(x=paddle.to_tensor(data=-self.rbound_upper
            ).astype(paddle.float32))
        end_value = paddle.exp(x=paddle.to_tensor(data=-self.rbound_lower).
            astype(paddle.float32))
        means = paddle.linspace(start=start_value, stop=end_value, num=self
            .num_rbf)
        betas = paddle.to_tensor(data=[(2 / self.num_rbf * (end_value -
            start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        paddle.assign(means, output=self.means.data)
        paddle.assign(betas, output=self.betas.data)

    def forward(self, dist):
        dist = dist.unsqueeze(axis=-1)
        rbounds = 0.5 * (paddle.cos(x=dist * pi / self.rbound_upper) + 1.0)
        rbounds = rbounds * (dist < self.rbound_upper).astype(dtype='float32')
        return rbounds * paddle.exp(x=-self.betas * paddle.square(x=paddle.
            exp(x=-dist) - self.means))


class NeighborEmb(MessagePassing):
    """Initialize node features based on neighboring nodes."""

    def __init__(self, hid_dim, in_hidden_channels=5):
        super(NeighborEmb, self).__init__(aggr='add')
        self.embedding = paddle.nn.Linear(in_features=in_hidden_channels,
            out_features=hid_dim)
        self.hid_dim = hid_dim
        self.ln_emb = paddle.nn.LayerNorm(normalized_shape=hid_dim,
            weight_attr=False, bias_attr=False)

    def forward(self, z, s, edge_index, embs):
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)
        s = s + s_neighbors
        return s

    def message(self, x_j, norm):
        return norm.reshape([-1, self.hid_dim]) * x_j


class CFConvS2V(MessagePassing):
    """Scalar to vector."""

    def __init__(self, hid_dim: int):
        super(CFConvS2V, self).__init__(aggr='add')
        self.hid_dim = hid_dim
        self.lin1 = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            hid_dim, out_features=hid_dim), paddle.nn.LayerNorm(
            normalized_shape=hid_dim, weight_attr=False, bias_attr=False),
            paddle.nn.Silu())

    def forward(self, s, v, edge_index, emb):
        """_summary_

        Args:
            s (_type_): _description_, [n_atom, n_z, n_embed]
            v (_type_): _description_, [n_edge, n_pos, n_embed]
            edge_index (_type_): _description_, [2, n_edge]
            emb (_type_): _description_, [n_edge, n_embed]

        Returns:
            _type_: _description_
        """
        s = self.lin1(s)
        emb = emb.unsqueeze(axis=1) * v
        v = self.propagate(edge_index, x=s, norm=emb)
        return v.reshape([-1, 3, self.hid_dim])

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(axis=1)
        a = norm.reshape([-1, 3, self.hid_dim]) * x_j
        return a.reshape([-1, 3 * self.hid_dim])


class GCLMessage(paddle.nn.Layer):

    def __init__(self, hidden_channels, num_radial, act_fn: str='swish',
        legacy: bool=False):
        super().__init__()
        self.edge_mlp = MLP(in_dim=hidden_channels * 2 + 3 *
            hidden_channels + num_radial, out_dims=[hidden_channels,
            hidden_channels], activation=act_fn)
        self.node_mlp = MLP(in_dim=hidden_channels + hidden_channels,
            out_dims=[hidden_channels, hidden_channels], activation=act_fn,
            last_layer_no_activation=True if legacy else False)
        self.edge_out_trans = MLP(in_dim=hidden_channels, out_dims=[3 *
            hidden_channels + num_radial], activation=act_fn)
        self.att_mlp = MLP(hidden_channels, [1], activation=act_fn)
        self.x_layernorm = paddle.nn.LayerNorm(normalized_shape=hidden_channels
            )
        # self.x_layernorm.reset_parameters()

    def forward(self, x, edge_index, weight):
        xh = self.x_layernorm(x)
        edgeh = weight
        ii, jj = edge_index
        m_ij = self.edge_message(xh[ii], xh[jj], edgeh)
        xh = self.node_message(xh, edge_index, m_ij)
        edgeh = edgeh + self.edge_out_trans(m_ij)
        return xh, edgeh

    def edge_message(self, xh_i, xh_j, edgeh):
        m_ij = self.edge_mlp(paddle.concat(x=[xh_i, xh_j, edgeh], axis=1))
        m_ij = m_ij * self.att_mlp(m_ij)
        return m_ij

    def node_message(self, xh, edge_index, m_ij):
        ii, jj = edge_index
        agg = unsorted_segment_sum(m_ij, ii, num_segments=xh.shape[0],
            normalization_factor=1, aggregation_method='mean')
        agg = paddle.concat(x=[xh, agg], axis=1)
        xh = xh + self.node_mlp(agg)
        return xh


class EquiMessage(MessagePassing):

    def __init__(self, hidden_channels, num_radial, reflect_equiv):
        super(EquiMessage, self).__init__(aggr='add', node_dim=0)
        self.hidden_channels = hidden_channels
        self.num_radial = num_radial
        self.reflect_equiv = reflect_equiv
        self.dir_proj = paddle.nn.Sequential(paddle.nn.Linear(in_features=3 *
            self.hidden_channels + self.num_radial, out_features=self.
            hidden_channels * 3), paddle.nn.Silu(), paddle.nn.Linear(
            in_features=self.hidden_channels * 3, out_features=self.
            hidden_channels * 3))
        self.x_proj = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            hidden_channels, out_features=hidden_channels, bias_attr=False),
            paddle.nn.Silu(), paddle.nn.Linear(in_features=hidden_channels,
            out_features=hidden_channels * 3, bias_attr=False))
        self.rbf_proj = paddle.nn.Linear(in_features=num_radial,
            out_features=hidden_channels * 3, bias_attr=False)
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = paddle.nn.LayerNorm(normalized_shape=hidden_channels
            )
        self.reset_parameters()

    def reset_parameters(self):
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.x_proj[0].weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.x_proj[2].weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.rbf_proj.weight)
        # self.x_layernorm.reset_parameters()

    def forward(self, x, vec, edge_index, edge_rbf, weight, edge_vector,
        edge_cross):
        xh = self.x_proj(self.x_layernorm(x))
        rbfh = self.rbf_proj(edge_rbf)
        weight = self.dir_proj(weight)
        rbfh = rbfh * weight
        dx, dvec = self.propagate(edge_index, xh=xh, vec=vec, rbfh_ij=rbfh,
            r_ij=edge_vector, edge_cross=edge_cross, size=None)
        return dx, dvec

    def message(self, xh_j, xh_i, vec_j, rbfh_ij, r_ij, edge_cross):
        x, xh2, xh3 = paddle_aux.split(x=(xh_j + xh_i) * rbfh_ij,
            num_or_sections=self.hidden_channels, axis=-1)
        xh2 = xh2 * self.inv_sqrt_3
        vec = vec_j * xh2.unsqueeze(axis=1) + xh3.unsqueeze(axis=1
            ) * r_ij.unsqueeze(axis=2)
        if not self.reflect_equiv:
            vec = vec + x.unsqueeze(axis=1) * edge_cross.unsqueeze(axis=2)
        vec = vec * self.inv_sqrt_h
        return x, vec

    def aggregate(self, features: Tuple[paddle.Tensor, paddle.Tensor],
        index: paddle.Tensor, ptr: Optional[paddle.Tensor], dim_size:
        Optional[int]) ->Tuple[paddle.Tensor, paddle.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=
            'sum')
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size,
            reduce='sum')
        return x, vec

    def update(self, inputs: Tuple[paddle.Tensor, paddle.Tensor]) ->Tuple[
        paddle.Tensor, paddle.Tensor]:
        return inputs


class EquiUpdate(paddle.nn.Layer):

    def __init__(self, hidden_channels, reflect_equiv: bool=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.vec_proj = paddle.nn.Linear(in_features=hidden_channels,
            out_features=hidden_channels * 2, bias_attr=False)
        self.xvec_proj = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            hidden_channels * 2, out_features=hidden_channels, bias_attr=
            False), paddle.nn.Silu(), paddle.nn.Linear(in_features=
            hidden_channels, out_features=hidden_channels * 3, bias_attr=False)
            )
        self.lin3 = paddle.nn.Sequential(paddle.nn.Linear(in_features=3,
            out_features=48), paddle.nn.Silu(), paddle.nn.Linear(
            in_features=48, out_features=8), paddle.nn.Silu(), paddle.nn.
            Linear(in_features=8, out_features=1))
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.reflect_equiv = reflect_equiv
        self.reset_parameters()

    def reset_parameters(self):
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.vec_proj.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.xvec_proj[0].weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.xvec_proj[2].weight)

    def forward(self, x, vec, nodeframe):
        vec = self.vec_proj(vec)
        vec1, vec2 = paddle_aux.split(x=vec, num_or_sections=self.
            hidden_channels, axis=-1)
        scalrization = paddle.sum(x=vec1.unsqueeze(axis=2) * nodeframe.
            unsqueeze(axis=-1), axis=1)
        if self.reflect_equiv:
            scalrization[:, 1, :] = paddle.abs(x=scalrization[:, 1, :].clone())
        scalar = self.lin3(paddle.transpose(x=scalrization, perm=(0, 2, 1))
            ).squeeze(axis=-1)
        vec_dot = (vec1 * vec2).sum(axis=1)
        vec_dot = vec_dot * self.inv_sqrt_h
        x_vec_h = self.xvec_proj(paddle.concat(x=[x, scalar], axis=-1))
        xvec1, xvec2, xvec3 = paddle_aux.split(x=x_vec_h, num_or_sections=
            self.hidden_channels, axis=-1)
        dx = xvec1 + xvec2 + vec_dot
        dx = dx * self.inv_sqrt_2
        dvec = xvec3.unsqueeze(axis=1) * vec2
        return dx, dvec


class _EquiUpdate(paddle.nn.Layer):

    def __init__(self, hidden_channels, reflect_equiv: bool=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.vec_proj = paddle.nn.Linear(in_features=hidden_channels,
            out_features=hidden_channels * 2, bias_attr=False)
        self.vec_proj2 = paddle.nn.Linear(in_features=hidden_channels,
            out_features=hidden_channels, bias_attr=False)
        self.xvec_proj = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            hidden_channels * 2, out_features=hidden_channels), paddle.nn.
            Silu(), paddle.nn.Linear(in_features=hidden_channels,
            out_features=hidden_channels * 3))
        self.lin3 = paddle.nn.Sequential(paddle.nn.Linear(in_features=3,
            out_features=64), paddle.nn.Silu(), paddle.nn.Linear(
            in_features=64, out_features=8), paddle.nn.Silu(), paddle.nn.
            Linear(in_features=8, out_features=1))
        self.lin4 = paddle.nn.Sequential(paddle.nn.Linear(in_features=6,
            out_features=64), paddle.nn.Silu(), paddle.nn.Linear(
            in_features=64, out_features=8), paddle.nn.Silu(), paddle.nn.
            Linear(in_features=8, out_features=1))
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.vec_proj.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(value=0)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(value=0)

    def forward(self, x, vec, nodeframe):
        vec = self.vec_proj(vec)
        vec1, vec2 = paddle_aux.split(x=vec, num_or_sections=self.
            hidden_channels, axis=-1)
        scalrization = paddle.sum(x=vec1.unsqueeze(axis=2) * nodeframe.
            unsqueeze(axis=-1), axis=1)
        scalrization[:, 1, :] = paddle.abs(x=scalrization[:, 1, :].clone())
        scalar = paddle.sqrt(x=paddle.sum(x=vec1 ** 2, axis=-2))
        scalrization1 = paddle.sum(x=vec2.unsqueeze(axis=2) * nodeframe.
            unsqueeze(axis=-1), axis=1)
        scalrization1[:, 1, :] = paddle.abs(x=scalrization1[:, 1, :].clone())
        vec_dot = self.lin4(paddle.transpose(x=paddle.concat(x=[
            scalrization, scalrization1], axis=-2), perm=(0, 2, 1))).squeeze(
            axis=-1)
        vec_dot = vec_dot * self.inv_sqrt_h
        x_vec_h = self.xvec_proj(paddle.concat(x=[x, scalar], axis=-1))
        xvec1, xvec2, xvec3 = paddle_aux.split(x=x_vec_h, num_or_sections=
            self.hidden_channels, axis=-1)
        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2
        dvec = xvec3.unsqueeze(axis=1) * vec2
        return dx, dvec


class vector(MessagePassing):

    def __init__(self):
        super(vector, self).__init__(aggr='mean')

    def forward(self, vector, edge_index):
        v = self.propagate(edge_index, x=vector)
        return v


def nn_vector(dist: paddle.Tensor, edge_index: paddle.Tensor, pos: paddle.
    Tensor):
    """Added by Chenru: Getting the nearest neighbor position to construct nodeframe.

    Args:
        dist (Tensor): (n_edge)
        edge_index (Tensor): (2, n_edge)
        pos (Tensor): (n_atom, 3)
    Returns:
        Tensor: (n_atom, 3)
    """
    ii, jj = edge_index
    vec = []
    pairs = {}
    for n in range(pos.shape[0]):
        if n not in pairs:
            inds = paddle.where(ii == n)[0].squeeze() # >>>>>>
            if not len(inds):
                nn_j = n
            else:
                min_ind = paddle.argmin(x=dist[inds])
                nn_j = jj[inds][min_ind].item()
            pairs.update({nn_j: n})
        else:
            nn_j = pairs[n]
        vec.append(pos[nn_j])
    vec = paddle.stack(x=vec)
    return vec


def assert_rot_equiv(func: Callable, dist: paddle.Tensor, edge_index:
    paddle.Tensor, pos: paddle.Tensor):
    """Added by Chenru: test a func for constructing y1 is equivariant.

    Args:
        func (Callable): _description_
        dist (Tensor): _description_
        edge_index (Tensor): _description_
        pos (Tensor): _description_
    """
    theta = 0.4
    alpha = 0.9
    rot_x = paddle.to_tensor(data=[[1, 0, 0], [0, np.cos(theta), -np.sin(
        theta)], [0, np.sin(theta), np.cos(theta)]], dtype='float64')
    rot_y = paddle.to_tensor(data=[[np.cos(alpha), 0, np.sin(alpha)], [0, 1,
        0], [-np.sin(alpha), 0, np.cos(alpha)]], dtype='float64')
    rot = paddle.matmul(x=rot_y, y=rot_x).astype(dtype='float64')
    y1 = func(dist, edge_index, pos)
    pos_new = paddle.matmul(x=pos, y=rot).astype(dtype='float64')
    y1_new = func(dist, edge_index, pos_new)
    assert paddle.allclose(x=paddle.matmul(x=y1, y=rot).astype(dtype=
        'float64'), y=y1_new).item()


class EquiOutput(paddle.nn.Layer):

    def __init__(self, hidden_channels, out_channels=1, single_layer_output
        =True):
        super().__init__()
        self.hidden_channels = hidden_channels
        if single_layer_output:
            self.output_network = paddle.nn.LayerList(sublayers=[
                GatedEquivariantBlock(hidden_channels, out_channels)])
        else:
            self.output_network = paddle.nn.LayerList(sublayers=[
                GatedEquivariantBlock(hidden_channels, hidden_channels // 2
                ), GatedEquivariantBlock(hidden_channels // 2, out_channels)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return x, vec.squeeze()


class GatedEquivariantBlock(paddle.nn.Layer):
    """
    Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra.

    Borrowed from TorchMD-Net
    """

    def __init__(self, hidden_channels, out_channels):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels
        self.vec1_proj = paddle.nn.Linear(in_features=hidden_channels,
            out_features=hidden_channels, bias_attr=False)
        self.vec2_proj = paddle.nn.Linear(in_features=hidden_channels,
            out_features=out_channels, bias_attr=False)
        self.update_net = paddle.nn.Sequential(paddle.nn.Linear(in_features
            =hidden_channels * 2, out_features=hidden_channels), paddle.nn.
            Silu(), paddle.nn.Linear(in_features=hidden_channels,
            out_features=out_channels * 2))
        self.act = paddle.nn.Identity()

    def reset_parameters(self):
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.vec1_proj.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.vec2_proj.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.update_net[0].weight)

        init_Constant = paddle.nn.initializer.Constant(value=0.)
        init_Constant(self.update_net[0].bias)
        # self.update_net[0].bias.data.fill_(value=0)

        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.update_net[2].weight)

        init_Constant = paddle.nn.initializer.Constant(value=0.)
        init_Constant(self.update_net[2].bias)
        # self.update_net[2].bias.data.fill_(value=0)

    def forward(self, x, v):
        vec1 = paddle.linalg.norm(x=self.vec1_proj(v), axis=-2)
        vec2 = self.vec2_proj(v)
        x = paddle.concat(x=[x, vec1], axis=-1)
        x, v = paddle_aux.split(x=self.update_net(x), num_or_sections=self.
            out_channels, axis=-1)
        v = v.unsqueeze(axis=1) * vec2
        x = self.act(x)
        return x, v


class LEFTNet(paddle.nn.Layer):
    """
    LEFTNet

    Args:
        pos_require_grad (bool, optional): If set to :obj:`True`, will require to take derivative of model output with respect to the atomic positions. (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
        num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
        hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
        num_radial (int, optional): Number of radial basis functions. (default: :obj:`96`)
        y_mean (float, optional): Mean value of the labels of training data. (default: :obj:`0`)
        y_std (float, optional): Standard deviation of the labels of training data. (default: :obj:`1`)

    """

    def __init__(self, pos_require_grad=False, cutoff=10.0, num_layers=4,
        hidden_channels=128, num_radial=96, in_hidden_channels: int=8,
        reflect_equiv: bool=True, legacy: bool=True, update: bool=True,
        pos_grad: bool=False, single_layer_output: bool=True, for_conf:
        bool=False, ff: bool=False, object_aware: bool=True, **kwargs):
        super(LEFTNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.pos_require_grad = pos_require_grad
        self.reflect_equiv = reflect_equiv
        self.legacy = legacy
        self.update = update
        self.pos_grad = pos_grad
        self.for_conf = for_conf
        self.ff = ff
        self.object_aware = object_aware
        self.embedding = paddle.nn.Linear(in_features=in_hidden_channels,
            out_features=hidden_channels)
        self.embedding_out = paddle.nn.Linear(in_features=hidden_channels,
            out_features=in_hidden_channels)
        self.radial_emb = RBFEmb(num_radial, self.cutoff)
        self.neighbor_emb = NeighborEmb(hidden_channels, in_hidden_channels)
        self.s2v = CFConvS2V(hidden_channels)
        self.radial_lin = paddle.nn.Sequential(paddle.nn.Linear(in_features
            =num_radial, out_features=hidden_channels), paddle.nn.Silu(),
            paddle.nn.Linear(in_features=hidden_channels, out_features=
            hidden_channels))
        self.lin3 = paddle.nn.Sequential(paddle.nn.Linear(in_features=3,
            out_features=hidden_channels // 4), paddle.nn.Silu(), paddle.nn
            .Linear(in_features=hidden_channels // 4, out_features=1))
        self.pos_expansion = MLP(in_dim=3, out_dims=[hidden_channels // 2,
            hidden_channels], activation='swish', last_layer_no_activation=
            True, bias=False)
        if self.legacy:
            self.distance_embedding = MLP(in_dim=num_radial, out_dims=[
                hidden_channels // 2, hidden_channels], activation='swish',
                bias=False)
        if self.pos_grad:
            self.dynamic_mlp_modules = paddle.nn.Sequential(paddle.nn.
                Linear(in_features=hidden_channels, out_features=
                hidden_channels // 2), paddle.nn.Silu(), paddle.nn.Linear(
                in_features=hidden_channels // 2, out_features=3))
        self.gcl_layers = paddle.nn.LayerList()
        self.message_layers = paddle.nn.LayerList()
        self.update_layers = paddle.nn.LayerList()
        for _ in range(num_layers):
            self.gcl_layers.append(GCLMessage(hidden_channels, num_radial,
                legacy=legacy))
            # self.message_layers.append(EquiMessage(hidden_channels,
            #     num_radial, reflect_equiv).jittable())
            self.message_layers.append(EquiMessage(hidden_channels,
                num_radial, reflect_equiv))
            self.update_layers.append(EquiUpdate(hidden_channels,
                reflect_equiv))
        self.last_layer = paddle.nn.Linear(in_features=hidden_channels,
            out_features=1)
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.out_pos = EquiOutput(hidden_channels, out_channels=1,
            single_layer_output=single_layer_output)
        self.vec = vector()
        self.reset_parameters()

    def reset_parameters(self):
        self.radial_emb.reset_parameters()

    def scalarization(self, pos, edge_index):
        i, j = edge_index
        dist = (pos[i] - pos[j]).pow(y=2).sum(axis=-1).sqrt()
        coord_diff = pos[i] - pos[j]
        radial = paddle.sum(x=coord_diff ** 2, axis=1).unsqueeze(axis=1)
        coord_cross = paddle.cross(x=pos[i], y=pos[j])
        norm = paddle.sqrt(x=radial) + EPS
        coord_diff = coord_diff / norm
        cross_norm = paddle.sqrt(x=paddle.sum(x=coord_cross ** 2, axis=1).
            unsqueeze(axis=1)) + EPS
        coord_cross = coord_cross / cross_norm
        coord_vertical = paddle.cross(x=coord_diff, y=coord_cross)
        return dist, coord_diff, coord_cross, coord_vertical

    @staticmethod
    def assemble_nodemask(edge_index: paddle.Tensor, pos: paddle.Tensor):
        node_mask = paddle.zeros(shape=pos.shape[0])
        node_mask[:] = -1
        _i, _j = edge_index
        _ind = 0
        for center in range(pos.shape[0]):
            if node_mask[center] > -1:
                continue
            _connected = _j[paddle.where(_i == center)[0].squeeze(1)] # >>>>>>
            _connected = paddle.concat(x=[_connected, paddle.to_tensor(data
                =[center], place=pos.place)])
            node_mask[_connected] = _ind
            _ind += 1
        return node_mask

    def forward(
        self, 
        h: paddle.Tensor, 
        pos: paddle.Tensor, 
        edge_index: paddle.Tensor, 
        edge_attr: Optional[paddle.Tensor]=None, 
        node_mask: Optional[paddle.Tensor]=None, 
        edge_mask: Optional[paddle.Tensor]= None, 
        update_coords_mask: Optional[paddle.Tensor]=None,
        subgraph_mask: Optional[paddle.Tensor]=None
    ):
        if not self.object_aware:
            subgraph_mask = None
        i, j = edge_index
        z_emb = self.embedding(h)
        i, j = edge_index
        dist = (pos[i] - pos[j]).pow(y=2).sum(axis=-1).sqrt()
        inner_subgraph_mask = paddle.zeros(shape=[edge_index.shape[1], 1])
        inner_subgraph_mask[paddle.where(dist < self.cutoff)[0].squeeze()] = 1
        all_edge_masks = inner_subgraph_mask
        if subgraph_mask is not None:
            all_edge_masks = all_edge_masks * subgraph_mask
        edge_index_w_cutoff = edge_index.T[paddle.where(all_edge_masks > 0)[0].squeeze()
            ].T
        node_mask_w_cutoff = self.assemble_nodemask(edge_index=
            edge_index_w_cutoff, pos=pos)
        pos_frame = pos.clone()
        pos_frame = remove_mean_batch(pos_frame, node_mask_w_cutoff.astype(
            dtype='int64'))
        dist, coord_diff, coord_cross, coord_vertical = self.scalarization(
            pos_frame, edge_index)
        dist = dist * all_edge_masks.squeeze(axis=-1)
        coord_diff = coord_diff * all_edge_masks
        coord_cross = coord_cross * all_edge_masks
        coord_vertical = coord_vertical * all_edge_masks
        frame = paddle.concat(x=(coord_diff.unsqueeze(axis=-1), coord_cross
            .unsqueeze(axis=-1), coord_vertical.unsqueeze(axis=-1)), axis=-1)
        radial_emb = self.radial_emb(dist)
        radial_emb = radial_emb * all_edge_masks
        f = self.radial_lin(radial_emb)
        rbounds = 0.5 * (paddle.cos(x=dist * pi / self.cutoff) + 1.0)
        f = rbounds.unsqueeze(axis=-1) * f
        s = self.neighbor_emb(h, z_emb, edge_index, f)
        NE1 = self.s2v(s, coord_diff.unsqueeze(axis=-1), edge_index, f)
        scalrization1 = paddle.sum(x=NE1[i].unsqueeze(axis=2) * frame.
            unsqueeze(axis=-1), axis=1)
        scalrization2 = paddle.sum(x=NE1[j].unsqueeze(axis=2) * frame.
            unsqueeze(axis=-1), axis=1)
        if self.reflect_equiv:
            scalrization1[:, 1, :] = paddle.abs(x=scalrization1[:, 1, :].
                clone())
            scalrization2[:, 1, :] = paddle.abs(x=scalrization2[:, 1, :].
                clone())
        scalar3 = (self.lin3(paddle.transpose(x=scalrization1, perm=(0, 2, 
            1))) + paddle.transpose(x=scalrization1, perm=(0, 2, 1))[:, :, 
            0].unsqueeze(axis=2)).squeeze(axis=-1)
        scalar4 = (self.lin3(paddle.transpose(x=scalrization2, perm=(0, 2, 
            1))) + paddle.transpose(x=scalrization2, perm=(0, 2, 1))[:, :, 
            0].unsqueeze(axis=2)).squeeze(axis=-1)
        edgeweight = paddle.concat(x=(scalar3, scalar4), axis=-1
            ) * rbounds.unsqueeze(axis=-1)
        edgeweight = paddle.concat(x=(edgeweight, f), axis=-1)
        edgeweight = paddle.concat(x=(edgeweight, radial_emb), axis=-1)
        a = pos_frame
        if self.legacy:
            b = self.vec(pos_frame, edge_index)
        else:
            eff_edge_ij = paddle.where(all_edge_masks.squeeze(axis=-1) == 1)[0].squeeze()
            eff_edge_index = edge_index[:, eff_edge_ij]
            eff_dist = dist[eff_edge_ij]
            b = nn_vector(eff_dist, eff_edge_index, pos_frame)
        x1 = (a - b) / (paddle.sqrt(x=paddle.sum(x=(a - b) ** 2, axis=1).
            unsqueeze(axis=1)) + EPS)
        y1 = paddle.cross(x=a, y=b)
        
        # The following code is for precision alignment with torch code
        # data = np.load('y1.npy')
        # y1 = paddle.to_tensor(data)
        normy = paddle.sqrt(x=paddle.sum(x=y1 ** 2, axis=1).unsqueeze(axis=1)
            ) + EPS
        y1 = y1 / normy
        z1 = paddle.cross(x=x1, y=y1)
        nodeframe = paddle.concat(x=(x1.unsqueeze(axis=-1), y1.unsqueeze(
            axis=-1), z1.unsqueeze(axis=-1)), axis=-1)
        pos_prjt = paddle.sum(x=pos_frame.unsqueeze(axis=-1) * nodeframe,
            axis=1)
        vec = paddle.zeros(shape=[s.shape[0], 3, s.shape[1]])
        gradient = paddle.zeros(shape=[s.shape[0], 3])
        for i in range(self.num_layers):
            if self.legacy or i == 0:
                s = s + self.pos_expansion(pos_prjt)
            s, edgeweight = self.gcl_layers[i](s, edge_index, edgeweight)
            dx, dvec = self.message_layers[i](s, vec, edge_index,
                radial_emb, edgeweight, coord_diff, coord_cross)
            s = s + dx
            vec = vec + dvec
            s = s * self.inv_sqrt_2
            if self.update:
                dx, dvec = self.update_layers[i](s, vec, nodeframe)
                s = s + dx
                vec = vec + dvec
            if self.pos_grad:
                dynamic_coff = self.dynamic_mlp_modules(s)
                basis_mix = dynamic_coff[:, :1] * x1 + dynamic_coff[:, 1:2
                    ] * y1 + dynamic_coff[:, 2:3] * z1
                gradient = gradient + basis_mix / self.num_layers
        if self.for_conf:
            return s
        _, dpos = self.out_pos(s, vec)
        if update_coords_mask is not None:
            dpos = update_coords_mask * dpos
        pos = pos + dpos + gradient
        if self.ff:
            return s, dpos
        h = self.embedding_out(s)
        if node_mask is not None:
            h = h * node_mask
        edge_attr = None
        return h, pos, edge_attr
