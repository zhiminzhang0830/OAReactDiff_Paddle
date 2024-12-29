import sys
from utils import paddle_aux
import paddle
import numpy as np


class DoubleDistributionNodes:

    def __init__(self, histogram):
        histogram = paddle.to_tensor(data=histogram).astype(dtype='float32')
        histogram = histogram + 0.001
        prob = histogram / histogram.sum()
        self.idx_to_n_nodes = paddle.to_tensor(data=[[(i, j) for j in range
            (tuple(prob.shape)[1])] for i in range(tuple(prob.shape)[0])]
            ).view(-1, 2)
        self.n_nodes_to_idx = {tuple(x.tolist()): i for i, x in enumerate(
            self.idx_to_n_nodes)}
        self.prob = prob
>>>>>>        self.m = torch.distributions.Categorical(self.prob.view(-1),
            validate_args=True)
>>>>>>        self.n1_given_n2 = [torch.distributions.Categorical(prob[:, j],
            validate_args=True) for j in range(tuple(prob.shape)[1])]
>>>>>>        self.n2_given_n1 = [torch.distributions.Categorical(prob[i, :],
            validate_args=True) for i in range(tuple(prob.shape)[0])]
        entropy = self.m.entropy()
        print('Entropy of n_nodes: H[N]', entropy.item())

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        num_nodes_lig, num_nodes_pocket = self.idx_to_n_nodes[idx].T
        return num_nodes_lig, num_nodes_pocket

    def sample_conditional(self, n1=None, n2=None):
        assert (n1 is None) ^ (n2 is None
            ), 'Exactly one input argument must be None'
        m = self.n1_given_n2 if n2 is not None else self.n2_given_n1
        c = n2 if n2 is not None else n1
        return paddle.to_tensor(data=[m[i].sample() for i in c], place=c.place)

    def log_prob(self, batch_n_nodes_1, batch_n_nodes_2):
        assert len(tuple(batch_n_nodes_1.shape)) == 1
        assert len(tuple(batch_n_nodes_2.shape)) == 1
        idx = paddle.to_tensor(data=[self.n_nodes_to_idx[n1, n2] for n1, n2 in
            zip(batch_n_nodes_1.tolist(), batch_n_nodes_2.tolist())])
        log_probs = self.m.log_prob(idx)
        return log_probs.to(batch_n_nodes_1.place)

    def log_prob_n1_given_n2(self, n1, n2):
        assert len(tuple(n1.shape)) == 1
        assert len(tuple(n2.shape)) == 1
        log_probs = paddle.stack(x=[self.n1_given_n2[c].log_prob(i.cpu()) for
            i, c in zip(n1, n2)])
        return log_probs.to(n1.place)

    def log_prob_n2_given_n1(self, n2, n1):
        assert len(tuple(n2.shape)) == 1
        assert len(tuple(n1.shape)) == 1
        log_probs = paddle.stack(x=[self.n2_given_n1[c].log_prob(i.cpu()) for
            i, c in zip(n2, n1)])
        return log_probs.to(n2.place)


class SingleDistributionNodes:

    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = paddle.to_tensor(data=self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)
        self.prob = paddle.to_tensor(data=prob).astype(dtype='float32')
        entropy = paddle.sum(x=self.prob * paddle.log(x=self.prob + 1e-30))
        print('Entropy of n_nodes: H[N]', entropy.item())
>>>>>>        self.m = torch.distributions.Categorical(paddle.to_tensor(data=prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(tuple(batch_n_nodes.shape)) == 1
        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = paddle.to_tensor(data=idcs).to(batch_n_nodes.place)
        log_p = paddle.log(x=self.prob + 1e-30)
        log_p = log_p.to(batch_n_nodes.place)
        log_probs = log_p[idcs]
        return log_probs
