import paddle
"""Test model forward pass and equivariance."""
import unittest
import numpy as np
from oa_reactdiff.model import EGNN, LEFTNet
from .utils import tensor_relative_diff, egnn_config, init_weights, left_config
default_float = 'float64'
paddle.set_default_dtype(d=default_float)
EPS = 1e-08
TIGHT_EPS = 1e-08
theta = 0.4
alpha = 0.9
# >>>>>>pytorch_lightning.seed_everything(42, workers=True)


def com(x):
    return x - paddle.mean(x=x, axis=0)


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) ->None:
        cls.egnn = EGNN(**egnn_config)
        cls.leftnet = LEFTNet(**left_config)
        cls.edge_index = paddle.to_tensor(data=[[0, 1, 1, 2, 3, 0], [1, 0, 
            2, 1, 0, 3]], dtype='int64')
        cls.h = paddle.rand(shape=[4, 'egnn_config[in_node_nf]'])
        cls.pos = paddle.rand(shape=[4, 3])
        cls.edge_attr = paddle.rand(shape=[cls.edge_index.size(1),
            'egnn_config[in_edge_nf]'])
        egnn_config.update({'reflect_equiv': False})
        cls.egnn_no_reflect_equiv = EGNN(**egnn_config)
        egnn_config.update({'reflect_equiv': True})
        left_config.update({'reflect_equiv': False})
        cls.leftnet_no_reflect_equiv = LEFTNet(**left_config)
        left_config.update({'reflect_equiv': True})
        egnn_config['in_edge_nf'] = 0
        cls.egnn_no_edge_attr = EGNN(**egnn_config)
        cls.edge_attr_zeros = paddle.rand(shape=[cls.edge_index.size(1),
            'egnn_config[in_edge_nf]'])
        cls.edge_attr_null = None
        rot_x = paddle.to_tensor(data=[[1, 0, 0], [0, np.cos(theta), -np.
            sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype=
            default_float)
        rot_y = paddle.to_tensor(data=[[np.cos(alpha), 0, np.sin(alpha)], [
            0, 1, 0], [-np.sin(alpha), 0, np.cos(alpha)]], dtype=default_float)
        cls.rot = paddle.matmul(x=rot_y, y=rot_x).astype(dtype='float64')
        cls.trans = paddle.rand(shape=[3]) * 100
        cls.egnn.apply(init_weights)
        cls.egnn_no_edge_attr.apply(init_weights)
        cls.leftnet.apply(init_weights)
        cls.models = [cls.egnn, cls.leftnet]
        cls.models_no_edge_attr = [cls.egnn_no_edge_attr, cls.leftnet]
        cls.models_no_reflect_equiv = [cls.egnn_no_reflect_equiv, cls.
            leftnet_no_reflect_equiv]

    def test_rotation(self):
        for model in self.models:
            _h, _pos, _edge_attr = model.forward(self.h, self.pos, self.
                edge_index, self.edge_attr)
            _h_rot, _pos_rot, _edge_attr_rot = model.forward(self.h, paddle
                .matmul(x=self.pos, y=self.rot).astype(dtype='float64'),
                self.edge_index, self.edge_attr)
            print(tensor_relative_diff(_h, _h_rot))
            self.assertTrue(tensor_relative_diff(_h, _h_rot) < EPS)
            if _edge_attr is not None:
                self.assertTrue(tensor_relative_diff(_edge_attr,
                    _edge_attr_rot) < EPS)
            self.assertTrue(tensor_relative_diff(paddle.matmul(x=_pos, y=
                self.rot).astype(dtype='float64'), _pos_rot) < EPS)

    def test_translation(self):
        for ii, model in enumerate(self.models):
            if ii == 1:
                continue
            _h, _pos, _edge_attr = model.forward(self.h, self.pos, self.
                edge_index, self.edge_attr)
            _h_trans, _pos_trans, _edge_attr_trans = model.forward(self.h, 
                self.pos - self.trans, self.edge_index, self.edge_attr)
            print(tensor_relative_diff(_h, _h_trans))
            self.assertTrue(tensor_relative_diff(_h, _h_trans) < EPS)
            if _edge_attr is not None:
                self.assertTrue(tensor_relative_diff(_edge_attr,
                    _edge_attr_trans) < EPS)
            self.assertTrue(tensor_relative_diff(_pos - self.trans,
                _pos_trans) < EPS)

    def test_rotation_no_edge_attr(self):
        for model in self.models_no_edge_attr:
            _h, _pos, _edge_attr = model.forward(self.h, self.pos, self.
                edge_index, self.edge_attr_zeros)
            _h_rot, _pos_rot, _edge_attr_rot = model.forward(self.h, paddle
                .matmul(x=self.pos, y=self.rot), self.edge_index, self.
                edge_attr_zeros)
            if _edge_attr is not None:
                self.assertTrue(_edge_attr.shape[-1] == 1)
            print(paddle.max(x=_h - _h_rot))
            self.assertTrue(tensor_relative_diff(_h, _h_rot) < TIGHT_EPS)
            if _edge_attr is not None:
                self.assertTrue(tensor_relative_diff(_edge_attr,
                    _edge_attr_rot) < TIGHT_EPS)
            self.assertTrue(tensor_relative_diff(paddle.matmul(x=_pos, y=
                self.rot), _pos_rot) < EPS)
            _h_null, _pos_null, _edge_attr_null = model.forward(self.h,
                self.pos, self.edge_index, self.edge_attr_null)
            self.assertTrue(tensor_relative_diff(_h, _h_null) < TIGHT_EPS)
            if _edge_attr is not None:
                self.assertTrue(tensor_relative_diff(_edge_attr,
                    _edge_attr_null) < TIGHT_EPS)
            self.assertTrue(tensor_relative_diff(_pos, _pos_null) < TIGHT_EPS)

    def test_no_reflection_equiv(self):
        pos_reflect = paddle.concat(x=[self.pos[:, :2], -self.pos[:, 2:]],
            axis=1)
        for model in self.models_no_reflect_equiv:
            _h, _pos, _edge_attr = model.forward(self.h, self.pos, self.
                edge_index, self.edge_attr)
            _h_reflect, _pos_reflect, _ = model.forward(self.h, pos_reflect,
                self.edge_index, self.edge_attr)
            print(tensor_relative_diff(_h, _h_reflect))
            print(_pos)
            print(_pos_reflect)
            self.assertTrue(tensor_relative_diff(_pos, _pos_reflect) > 1e-05)


class TestDisconnected(unittest.TestCase):

    @classmethod
    def setUpClass(cls) ->None:
        cls.egnn = EGNN(**egnn_config)
        cls.leftnet = LEFTNet(**left_config)
        cls.models = [cls.egnn, cls.leftnet]
        cls.edge_index = paddle.to_tensor(data=[[0, 1, 1, 2, 0, 3, 4, 6, 4,
            5], [1, 0, 2, 1, 3, 0, 6, 4, 5, 4]], dtype='int64')
        cls.h = paddle.rand(shape=[4 + 3, 'egnn_config[in_node_nf]'])
        cls.pos = paddle.rand(shape=[4 + 3, 3])
        cls.edge_attr = paddle.rand(shape=[cls.edge_index.size(1),
            'egnn_config[in_edge_nf]'])

    def test_equal_subgraph(self):
        for model in self.models:
            _h, _pos, _edge_attr = model.forward(self.h, self.pos, self.
                edge_index, self.edge_attr)
            _h_cut, _pos_cut, _edge_attr_cut = model.forward(self.h[:4],
                self.pos[:4], self.edge_index[:, :6], self.edge_attr[:6])
            self.assertTrue(tensor_relative_diff(_h[:4], _h_cut) < TIGHT_EPS)
            if _edge_attr is not None:
                self.assertTrue(tensor_relative_diff(_edge_attr[:6],
                    _edge_attr_cut) < TIGHT_EPS)
            self.assertTrue(tensor_relative_diff(_pos[:4], _pos_cut) <
                TIGHT_EPS)
