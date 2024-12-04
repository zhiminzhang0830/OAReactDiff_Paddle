import paddle
from typing import Union, Iterable
import numpy as np


class Queue:

    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def get_grad_norm(parameters: Union[paddle.Tensor, Iterable[paddle.Tensor]],
    norm_type: float=2.0) ->paddle.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return paddle.to_tensor(data=0.0)
    device = parameters[0].grad.device
    total_norm = paddle.linalg.norm(x=paddle.stack(x=[paddle.linalg.norm(x=
        p.grad.detach(), p=norm_type).to(device) for p in parameters]), p=
        norm_type)
    return total_norm
