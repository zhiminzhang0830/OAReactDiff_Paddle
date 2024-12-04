import paddle
from typing import Dict, List, Optional, Tuple
from torch_scatter import scatter_mean
from oa_reactdiff.model import EGNN
from oa_reactdiff.model.core import GatedMLP
from oa_reactdiff.utils import get_subgraph_mask, get_n_frag_switch, get_mask_for_frag, get_edges_index
from ._base import BaseDynamics
FEATURE_MAPPING = ['pos', 'one_hot', 'charge']


class Confidence(BaseDynamics):

    def __init__(self, model_config: Dict, fragment_names: List[str],
        node_nfs: List[int], edge_nf: int, condition_nf: int=0, pos_dim:
        int=3, edge_cutoff: Optional[float]=None, model: paddle.nn.Layer=
        EGNN, device: (paddle.CPUPlace, paddle.CUDAPlace, str)=str('cuda').
        replace('cuda', 'gpu'), enforce_same_encoding: Optional[List]=None,
        source: Optional[Dict]=None, **kwargs) ->None:
        """Confindence score for generated samples.

        Args:
            model_config (Dict): config for the equivariant model.
            fragment_names (List[str]): list of names for fragments
            node_nfs (List[int]): list of number of input node attributues.
            edge_nf (int): number of input edge attributes.
            condition_nf (int): number of attributes for conditional generation.
            Defaults to 0.
            pos_dim (int): dimension for position vector. Defaults to 3.
            update_pocket_coords (bool): whether to update positions of everything.
                Defaults to True.
            condition_time (bool): whether to condition on time. Defaults to True.
            edge_cutoff (Optional[float]): cutoff for building intra-fragment edges.
                Defaults to None.
            model (Optional[nn.Module]): Module for equivariant model. Defaults to None.
        """
        model_config.update({'for_conf': True})
        update_pocket_coords = True
        condition_time = True,
        super().__init__(model_config, fragment_names, node_nfs, edge_nf,
            condition_nf, pos_dim, update_pocket_coords, condition_time,
            edge_cutoff, model, device, enforce_same_encoding, source=source)
        hidden_channels = model_config['hidden_channels']
        self.readout = GatedMLP(in_dim=hidden_channels, out_dims=[
            hidden_channels, hidden_channels, 1], activation='swish', bias=
            True, last_layer_no_activation=True)

    def _forward(self, xh: List[paddle.Tensor], edge_index: paddle.Tensor,
        t: paddle.Tensor, conditions: paddle.Tensor, n_frag_switch: paddle.
        Tensor, combined_mask: paddle.Tensor, edge_attr: Optional[paddle.
        Tensor]=None) ->paddle.Tensor:
        """predict confidence.

        Args:
            xh (List[Tensor]): list of concatenated tensors for pos and h
            edge_index (Tensor): [n_edge, 2]
            t (Tensor): time tensor. If dim is 1, same for all samples;
                otherwise different t for different samples
            conditions (Tensor): condition tensors
            n_frag_switch (Tensor): [n_nodes], fragment index for each nodes
            combined_mask (Tensor): [n_nodes], sample index for each node
            edge_attr (Optional[Tensor]): [n_edge, dim_edge_attribute]. Defaults to None.

        Raises:
            NotImplementedError: The fragement-position-fixed mode is not implement.

        Returns:
            Tensor: binary probability of confidence fo each graph.
        """
        pos = paddle.concat(x=[_xh[:, :self.pos_dim].clone() for _xh in xh],
            axis=0)
        h = paddle.concat(x=[self.encoders[ii](xh[ii][:, self.pos_dim:].
            clone()) for ii, name in enumerate(self.fragment_names)], axis=0)
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        condition_dim = 0
        if self.condition_time:
            if len(tuple(t.shape)) == 1:
                h_time = paddle.empty_like(x=h[:, 0:1]).fill_(value=t.item())
            else:
                h_time = t[combined_mask]
            h = paddle.concat(x=[h, h_time], axis=1)
            condition_dim += 1
        if self.condition_nf > 0:
            h_condition = conditions[combined_mask]
            h = paddle.concat(x=[h, h_condition], axis=1)
            condition_dim += self.condition_nf
        subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch)
        if self.update_pocket_coords:
            update_coords_mask = None
        else:
            raise NotImplementedError
        node_features = self.model(h, pos, edge_index, edge_attr, node_mask
            =None, edge_mask=None, update_coords_mask=update_coords_mask,
            subgraph_mask=subgraph_mask[:, None])
        graph_features = scatter_mean(node_features, index=combined_mask, dim=0
            )
        conf = self.readout(graph_features)
        return conf.squeeze()

    def forward(self, representations: List[Dict], conditions: paddle.Tensor):
        masks = [repre['mask'] for repre in representations]
        combined_mask = paddle.concat(x=masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [repr['size'] for repr in representations]
        n_frag_switch = get_n_frag_switch(fragments_nodes)
        xh = [paddle.concat(x=[repre[feature_type] for feature_type in
            FEATURE_MAPPING], axis=1) for repre in representations]
        pred = self._forward(xh=xh, edge_index=edge_index, t=paddle.
            to_tensor(data=[0]), conditions=conditions, n_frag_switch=
            n_frag_switch, combined_mask=combined_mask, edge_attr=None)
        return pred
