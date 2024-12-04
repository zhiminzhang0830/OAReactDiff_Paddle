import paddle
import unittest
from oa_reactdiff.utils import get_edges_index, get_subgraph_mask, get_n_frag_switch, get_mask_for_frag


class TestBasics(unittest.TestCase):

    def test_get_mask_for_frag(self):
        natms = paddle.to_tensor(data=[2, 0, 3], dtype='float32').astype(dtype
            ='int64')
        res = get_mask_for_frag(natms)
        self.assertTrue(paddle.allclose(x=res, y=paddle.to_tensor(data=[0, 
            0, 2, 2, 2], dtype='float32').astype(dtype='int64')).item())

    def test_get_n_frag_switch(self):
        natm_list = [paddle.to_tensor(data=[2, 0]), paddle.to_tensor(data=[
            1, 3]), paddle.to_tensor(data=[3, 2])]
        res = get_n_frag_switch(natm_list)
        self.assertTrue(paddle.allclose(x=res, y=paddle.to_tensor(data=[0, 
            0, 1, 1, 1, 1, 2, 2, 2, 2, 2])).item())

    def test_get_subgraph_mask(self):
        edge_index = paddle.to_tensor(data=[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2,
            0, 1]])
        n_frag_switch = paddle.to_tensor(data=[0, 0, 1])
        res = get_subgraph_mask(edge_index, n_frag_switch)
        self.assertTrue(paddle.allclose(x=res, y=paddle.to_tensor(data=[1, 
            0, 1, 0, 0, 0])).item())

    def test_complete_generation(self):
        natm_inorg_node = paddle.to_tensor(data=[2, 0])
        natm_org_edge = paddle.to_tensor(data=[2, 3])
        natm_org_node = paddle.to_tensor(data=[1, 2])
        inorg_node_mask = get_mask_for_frag(natm_inorg_node)
        org_edge_mask = get_mask_for_frag(natm_org_edge)
        org_node_mask = get_mask_for_frag(natm_org_node)
        n_frag_switch = get_n_frag_switch([natm_inorg_node, natm_org_edge,
            natm_org_node])
        self.assertTrue(paddle.allclose(x=n_frag_switch, y=paddle.to_tensor
            (data=[0, 0, 1, 1, 1, 1, 1, 2, 2, 2])).item())
        combined_mask = paddle.concat(x=(inorg_node_mask, org_edge_mask,
            org_node_mask))
        _edge_index = get_edges_index(combined_mask)
        self.assertTrue(tuple(_edge_index.shape) == (2, 2 * 5 + 0 * 5 + 2 *
            5 + 3 * 5 + 1 * 5 + 2 * 5))
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        self.assertTrue(tuple(edge_index.shape) == (2, 2 * 5 + 0 * 5 + 2 * 
            5 + 3 * 5 + 1 * 5 + 2 * 5 - 10))
        subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch)
        self.assertTrue(paddle.sum(x=subgraph_mask) == 2 + 2 + 6 + 2)
