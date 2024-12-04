import paddle
"""
Core layers provide basic operations, e.g., MLP
"""
from typing import List, Union
ACTIVATION_MAPPING = {'swish': paddle.nn.Silu(), 'silu': paddle.nn.Silu(),
    'relu': paddle.nn.ReLU(), 'sigmoid': paddle.nn.Sigmoid()}


class ZeroLayer(paddle.nn.Layer):
    """A skeleton layer that returns zeros."""

    def forward(self, inputs: List[paddle.Tensor], **kwargs) ->paddle.Tensor:
        return 0


class ConcatLayer(paddle.nn.Layer):
    """Concatnate layer."""

    def __init__(self, dim: int=-1) ->None:
        super().__init__()
        self.register_buffer(name='dim', tensor=paddle.to_tensor(data=dim))

    def forward(self, inputs: List[paddle.Tensor], **kwargs) ->paddle.Tensor:
        return paddle.concat(x=inputs, axis=self.dim)


class OneLayerActivation(paddle.nn.Layer):
    """One layer NN with activation."""

    def __init__(self, in_dim: int, out_dim: int, bias: int=True,
        activation=Union[str, None]) ->None:
        super().__init__()
        self.linear = paddle.nn.Linear(in_features=in_dim, out_features=
            out_dim, bias_attr=bias)
        self.activation = ACTIVATION_MAPPING[activation
            ] if activation is not None else paddle.nn.Identity()

    def forward(self, input: paddle.Tensor) ->paddle.Tensor:
        return self.activation(self.linear(input))


class MLP(paddle.nn.Layer):
    """Multi-layer perceptron."""

    def __init__(self, in_dim: int, out_dims: list, bias: bool=True,
        activation: Union[list[Union[str, None]], str, None]='swish',
        last_layer_no_activation: bool=False):
        super().__init__()
        input_dim = in_dim
        if isinstance(activation, str) or activation is None:
            activation = [activation] * len(out_dims)
        else:
            assert len(activation) == len(out_dims
                ), 'activation and out_dims must have the same length'
        if last_layer_no_activation:
            activation[-1] = None
        for _activation in activation:
            assert _activation is None or _activation in ACTIVATION_MAPPING, f'activation {activation} not avail.'
        module_list = []
        for ii in range(len(out_dims)):
            module_list.append(OneLayerActivation(in_dim=input_dim, out_dim
                =out_dims[ii], bias=bias, activation=activation[ii]))
            input_dim = out_dims[ii]
        self.mlp = paddle.nn.Sequential(*module_list)

    def forward(self, input: paddle.Tensor) ->paddle.Tensor:
        return self.mlp(input)


class GatedMLP(paddle.nn.Layer):
    """
    Gated MLP implementation. It implements the following
        `out = MLP(x) * MLP_\\\\sigmoid(x)`

    The current implementation is slightly different from the tf version,
    where the last activation from an MLP is forced to be sigmoid.
    """

    def __init__(self, in_dim: int, out_dims: list, bias: bool=True,
        activation: Union[list[Union[str, None]], str, None]='swish',
        gate_activation: str='sigmoid', last_layer_no_activation: bool=False):
        super().__init__()
        self.mlp = MLP(in_dim, out_dims, bias, activation,
            last_layer_no_activation=last_layer_no_activation)
        self.gmlp = MLP(in_dim, out_dims, bias, activation,
            last_layer_no_activation=last_layer_no_activation)
        self.gate_activation = ACTIVATION_MAPPING[gate_activation]

    def forward(self, input: paddle.Tensor) ->paddle.Tensor:
        return self.mlp(input) * self.gate_activation(self.gmlp(input))
