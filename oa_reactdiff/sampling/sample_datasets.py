import paddle
from oa_reactdiff.diffusion._node_dist import SingleDistributionNodes
from oa_reactdiff.utils import bond_analyze


@paddle.no_grad()
>>>>>>def sample_qm9(ddpm_trainer: pytorch_lightning.Trainer, nodes_dist:
    SingleDistributionNodes, bz: int, n_samples: int, n_real: int=1, n_fake:
    int=2, device: (paddle.CPUPlace, paddle.CUDAPlace, str)=str('cuda').
    replace('cuda', 'gpu')):
    n_batch = int(n_samples / bz)
    mols = []
    pos_dim = ddpm_trainer.ddpm.pos_dim
    for _ in range(n_batch):
        fragments_nodes = [nodes_dist.sample(shape=bz).to(device) for _ in
            range(n_real)]
        fragments_nodes += [paddle.ones(shape=bz).astype(dtype='int64') for
            _ in range(n_fake)]
        conditions = paddle.zeros(shape=(bz, 1))
        out_samples, out_masks = ddpm_trainer.ddpm.sample(n_samples=bz,
            fragments_nodes=fragments_nodes, conditions=conditions,
            return_frames=1, timesteps=None)
        sample_idxs = paddle.concat(x=[paddle.to_tensor(data=[0], place=
            device), paddle.cumsum(x=fragments_nodes[0], axis=0)])
        for ii in range(bz):
            _start, _end = sample_idxs[ii], sample_idxs[ii + 1]
            mols.append({'pos': out_samples[0][0][_start:_end, :pos_dim].
                detach().cpu(), 'atom': paddle.argmax(x=out_samples[0][0][
                _start:_end, pos_dim:-1].detach().cpu(), axis=1)})
    return mols
