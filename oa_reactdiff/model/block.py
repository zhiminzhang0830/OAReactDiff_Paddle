import paddle
"""Base layers for model."""
from typing import Tuple, Optional
import math
from .util_funcs import unsorted_segment_sum, coord2diff, coord2cross
from .core import MLP


class GCL(paddle.nn.Layer):

    def __init__(self, node_nf: int=8, hidden_nf: int=256,
        normalization_factor: float=1.0, aggregation_method: str='sum',
        edge_nf: int=0, nodes_attr_dim: int=0, act_fn: str='swish',
        attention: bool=False):
        """graph convolution layer.

        Args:
            node_nf (int): number of node features. Defaults to 8.
            hidden_nf (int): number of hidden units. Defaults to 256.
            normalization_factor (int): normalization factore in scattering.
                Defaults to 1.
            aggregation_method (str): aggregation options in scattering.
                Defaults to "sum".
            edge_nf (int): number of edge features. Defaults to 0.
            nodes_attr_dim (int): number of addition node attribues. Not useful
                in current diffusion process. Defaults to 0.
            act_fn (str): activation function. Defaults to "swish".
            attention (bool): whether to use self attention. Defaults to False.
        """
        super().__init__()
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.edge_mlp = MLP(in_dim=node_nf * 2 + hidden_nf, out_dims=[
            hidden_nf, hidden_nf], activation=act_fn)
        self.node_mlp = MLP(in_dim=hidden_nf + node_nf + nodes_attr_dim,
            out_dims=[hidden_nf, hidden_nf], activation=act_fn,
            last_layer_no_activation=True)
        if self.attention:
            self.att_mlp = MLP(hidden_nf, [1], activation=act_fn)

    def edge_model(self, source: paddle.Tensor, target: paddle.Tensor,
        edge_attr: paddle.Tensor, edge_mask: Optional[paddle.Tensor]=None
        ) ->Tuple[paddle.Tensor, paddle.Tensor]:
        """edge update function.

        Args:
            source (Tensor): h_i
            target (Tensor): h_j
            edge_attr (Tensor): e_ij
            edge_mask (Tensor): mask for {ij}

        Returns:
            Tuple[Tensor, Tensor]: e_ij_prime, m_ij
        """
        if edge_attr is None:
            out = paddle.concat(x=[source, target], axis=1)
        else:
            out = paddle.concat(x=[source, target, edge_attr], axis=1)
        mij = self.edge_mlp(out)
        out = mij if not self.attention else mij * self.att_mlp(mij)
        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, h: paddle.Tensor, edge_index: paddle.Tensor,
        edge_attr: paddle.Tensor, node_attr: Optional[paddle.Tensor]=None,
        node_mask: Optional[paddle.Tensor]=None) ->Tuple[paddle.Tensor,
        paddle.Tensor]:
        """node update function

        Args:
            h (Tensor): node features_
            edge_index (Tensor): {ij}
            edge_attr (Tensor): e_ij
            node_attr (Optional[Tensor]): additional node attributes. Defaults to None.
            node_mask (Optional[Tensor]): mask for {i}. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: h_i_prime, aggregationed node features
        """
        ii, jj = edge_index
        agg = unsorted_segment_sum(edge_attr, ii, num_segments=h.shape[0],
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method)
        agg = paddle.concat(x=[h, agg], axis=1)
        if node_attr is not None:
            agg = paddle.concat(x=[agg, node_attr], axis=1)
        out = h + self.node_mlp(agg)
        if node_mask is not None:
            out = out * node_mask
            agg = agg * node_mask
        return out, agg

    def forward(self, h: paddle.Tensor, edge_index: paddle.Tensor,
        edge_attr: Optional[paddle.Tensor]=None, node_attr: Optional[paddle
        .Tensor]=None, node_mask: Optional[paddle.Tensor]=None, edge_mask:
        Optional[paddle.Tensor]=None) ->Tuple[paddle.Tensor, paddle.Tensor]:
        """
        e_ij = phi_e(h_i ⊕ h_j ⊕ e_ij); phi_e can be MLP or gated MLP (i.e., with self attention.)
        h_i = h_i + phi_h(h_i ⊕ \\sum_j e_ij); phi_h is an MLP

        Args:
            h (Tensor): node features
            edge_index (Tensor): {ij}
            edge_attr (Optional[Tensor]): e_ij. Defaults to None.
            node_attr (Optional[Tensor]): additional node attributes. Defaults to None.
            node_mask (Optional[Tensor]): mask for {i}. Defaults to None.
            edge_mask (Optional[Tensor]): mask for {ij}. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: h_prime, e_ij_prime
        """
        ii, jj = edge_index
        edge_feat, mij = self.edge_model(h[ii], h[jj], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, node_mask
            )
        return h, edge_feat


class EquivariantUpdate(paddle.nn.Layer):

    def __init__(self, hidden_nf: int=256, normalization_factor: float=1.0,
        aggregation_method: str='sum', dist_dim: int=1, act_fn: str='swish',
        tanh: bool=False, coords_range: float=15.0, reflect_equiv: bool=True):
        """equivariant update layer for positions.

        Args:
            hidden_nf (int): number of hidden units. Defaults to 256.
            normalization_factor (float): distance normalizating factor. Defaults to 1.0.
            aggregation_method (str): aggregation options in scattering.
                Defaults to "sum".
            dist_dim (int): number of distance features. Defaults to 1.
            act_fn (str): activation function. Defaults to "swish".
            tanh (bool): whether to use tanh activation as additional activation.
                Defaults to False.
            coords_range (float): range factor, only used in tanh = True.
                Defaults to 15.0.
            reflect_equiv (bool): whether to ignore reflection.
                Defaults to True.
        """
        super().__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.reflect_equiv = reflect_equiv
        input_edge = hidden_nf * 2 + hidden_nf
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.distance_embedding = MLP(in_dim=dist_dim, out_dims=[16,
            hidden_nf], activation=act_fn)
        self.apply(self.init_distance_embedding)
        self.coord_mlp = MLP(in_dim=input_edge, out_dims=[hidden_nf,
            hidden_nf, 1], activation=act_fn)
        if not reflect_equiv:
            self.cross_product_mlp = MLP(in_dim=input_edge, out_dims=[
                hidden_nf, hidden_nf, 1], activation=act_fn)
            init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=0.001
                )
            init_XavierUniform(self.cross_product_mlp.mlp[-1].linear.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=0.001)
        init_XavierUniform(self.coord_mlp.mlp[-1].linear.weight)

    @staticmethod
    def init_distance_embedding(m):
        if isinstance(m, paddle.nn.Linear):
            gain = 1.0
            init_XavierUniform = paddle.nn.initializer.XavierUniform(gain=gain)
            init_XavierUniform(m.weight)
            if m.bias is not None:
                init_Uniform = paddle.nn.initializer.Uniform(low=-gain,
                    high=gain)
                init_Uniform(m.bias)

    def dist2h_model(self, h: paddle.Tensor, distances: paddle.Tensor,
        edge_index: paddle.Tensor, subgraph_mask: Optional[paddle.Tensor]=None
        ):
        ii, jj = edge_index
        if subgraph_mask is not None:
            distances = distances * subgraph_mask
        agg = unsorted_segment_sum(distances, ii, num_segments=h.shape[0],
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method)
        h = h + self.distance_embedding(agg)
        return h

    def coord_model(self, h: paddle.Tensor, pos: paddle.Tensor, edge_index:
        paddle.Tensor, coord_diff: paddle.Tensor, edge_attr: paddle.Tensor,
        coord_cross: Optional[paddle.Tensor]=None, edge_mask: Optional[
        paddle.Tensor]=None, update_coords_mask: Optional[paddle.Tensor]=
        None, subgraph_mask: Optional[paddle.Tensor]=None) ->paddle.Tensor:
        """position update function.

        Args:
            h (Tensor): node features.
            pos (Tensor): node position tensor.
            edge_index (Tensor): {ij}
            coord_diff (Tensor): position difference.
            edge_attr (Tensor): e_ij
            edge_mask (Optional[Tensor]): mask for {ij}.
                Defaults to None.
            update_coords_mask (Optional[Tensor]): mask for position updates.
                Defaults to None to update all nodes
            subgraph_mask (Optional[Tensor]): mask for positions aggregations.
                The idea is keep subgraph (i.e., fragment) level equivariance.
                Defaults to None.
            coord_cross (Optional[Tensor]): cross product of pos with com at 0.
                Defaults to None.

        Returns:
            Tensor: updated positions.
        """
        ii, jj = edge_index
        input_tensor = paddle.concat(x=[h[ii], h[jj], edge_attr], axis=1)
        if self.tanh:
            trans = coord_diff * paddle.nn.functional.tanh(x=self.coord_mlp
                (input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if not self.reflect_equiv:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = paddle.nn.functional.tanh(x=phi_cross)
            trans = trans + coord_cross * phi_cross
        if subgraph_mask is not None:
            trans = trans * subgraph_mask
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, ii, num_segments=pos.shape[0],
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method)
        if update_coords_mask is not None:
            agg = update_coords_mask * agg
        pos = pos + agg
        return pos

    def forward(self, h: paddle.Tensor, pos: paddle.Tensor, edge_index:
        paddle.Tensor, coord_diff: paddle.Tensor, distances: paddle.Tensor,
        edge_attr: paddle.Tensor, coord_cross: Optional[paddle.Tensor]=None,
        node_mask: Optional[paddle.Tensor]=None, edge_mask: Optional[paddle
        .Tensor]=None, update_coords_mask: Optional[paddle.Tensor]=None,
        subgraph_mask: Optional[paddle.Tensor]=None):
        """
        pos_i = pos_i + \\sum_j (pos_i - pos_j) * phi_pos(h_i ⊕ h_j ⊕ e_ij)
        """
        pos = self.coord_model(h, pos, edge_index, coord_diff, edge_attr,
            coord_cross, edge_mask, update_coords_mask=update_coords_mask,
            subgraph_mask=subgraph_mask)
        h = self.dist2h_model(h, distances, edge_index, subgraph_mask=
            subgraph_mask)
        if node_mask is not None:
            pos = pos * node_mask
            h = h * node_mask
        return pos, h


class EquivariantBlock(paddle.nn.Layer):

    def __init__(self, hidden_nf: int=256, edge_feat_nf: int=1, act_fn: str
        ='swish', n_layers: int=2, attention: bool=True, tanh: bool=True,
        coords_range: float=15.0, norm_constant: float=1.0, sin_embedding:
        Optional[paddle.nn.Layer]=None, normalization_factor: float=1.0,
        aggregation_method: str='sum', reflect_equiv: bool=True):
        """Bloak that combines GCL and position equivariant updates.

        Args:
            hidden_nf (int): number of hidden units. Defaults to 256.
            edge_feat_nf (int): number of edge features. Defaults to 1.
            act_fn (str): activation function. Defaults to "swish".
            n_layers (int): number of GCL layer. Defaults to 2.
            attention (bool): whether to use self attention. Defaults to True.
            tanh (bool): whether to use tanh activation as additional activation.
                Defaults to False.
            coords_range (float): range factor, only used in tanh = True.
                Defaults to 15.0.
            norm_constant (float): distance normalizating factor.. Defaults to 1.0.
            sin_embedding (Optional[nn.Module]): whether to use edge distance embedding.
                Defaults to None.
            normalization_factor (float): distance normalization used in coord2diff.
                Defaults to 1.0.
            aggregation_method (str): aggregation options in scattering.
                Defaults to "sum".
            reflect_equiv (bool): whether to ignore reflection.
                Defaults to True.
        """
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = coords_range
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflect_equiv = reflect_equiv
        self.dist_dim = 1 if sin_embedding is None else sin_embedding.dim
        for i in range(0, n_layers):
            self.add_sublayer(name='gcl_%d' % i, sublayer=GCL(node_nf=
                hidden_nf, hidden_nf=hidden_nf, edge_nf=edge_feat_nf,
                act_fn=act_fn, attention=attention, normalization_factor=
                normalization_factor, aggregation_method=aggregation_method))
        self.add_sublayer(name='gcl_equiv', sublayer=EquivariantUpdate(
            hidden_nf=hidden_nf, dist_dim=self.dist_dim, act_fn=act_fn,
            tanh=tanh, coords_range=coords_range, normalization_factor=
            normalization_factor, aggregation_method=aggregation_method,
            reflect_equiv=reflect_equiv))

    def forward(self, h: paddle.Tensor, pos: paddle.Tensor, edge_index:
        paddle.Tensor, edge_attr: paddle.Tensor, node_mask: Optional[paddle
        .Tensor]=None, edge_mask: Optional[paddle.Tensor]=None,
        update_coords_mask: Optional[paddle.Tensor]=None, subgraph_mask:
        Optional[paddle.Tensor]=None):
        """
        e_ij = ||pos_i - pos_j||^2 ⊕ e_ij
        for _ in range(n_inner_layer):
            h_i, e_ij = GCL(h_i, ij, e_ij, ...)
        pos_i = EquivUpdate(h_i, pos_i, ij, pos_i - pos_j, e_ij, ...)
        """
        dist_dim = 1
        distances, coord_diff = coord2diff(pos, edge_index, self.norm_constant)
        coord_cross = coord2cross(pos, edge_index, self.norm_constant)
        if subgraph_mask is not None:
            distances = distances * subgraph_mask
            coord_diff = coord_diff * subgraph_mask
            coord_cross = coord_cross * subgraph_mask
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
            dist_dim = self.sin_embedding.dim
        edge_attr = paddle.concat(x=[distances, edge_attr], axis=1)
        for i in range(0, self.n_layers):
            h, edge_attr = self._modules['gcl_%d' % i](h, edge_index,
                edge_attr, node_mask, edge_mask)
        pos, h = self._modules['gcl_equiv'](h, pos, edge_index, coord_diff,
            distances, edge_attr, coord_cross, node_mask, edge_mask,
            update_coords_mask=update_coords_mask, subgraph_mask=subgraph_mask)
        if node_mask is not None:
            h = h * node_mask
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask
        return h, pos, edge_attr[:, dist_dim:]


class SinusoidsEmbeddingNew(paddle.nn.Layer):

    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** paddle.arange(end=
            self.n_frequencies) / max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = paddle.sqrt(x=x + 1e-08)
        emb = x * self.frequencies[None, :].to(x.place)
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb.detach()
