# --- 导入和定义一些函数 ----
import paddle
import py3Dmol
import numpy as np
import os
from typing import Optional
from paddle import Tensor
from oa_reactdiff.model import o3
from oa_reactdiff.model import LEFTNet
from oa_reactdiff.model.scatter.scatter import scatter

default_float = "float32"
paddle.set_default_dtype(d=default_float) 


def remove_mean_batch(x: Tensor, indices: Optional[Tensor] = None) -> Tensor:
    """将x中的每个batch的均值去掉

    Args:
        x (tensor): input tensor.
        indices (Optional[tensor], optional): batch indices. Defaults to None.

    Returns:
        tensor: output tensor with batch mean as 0.
    """
    if indices is None:
        return x - paddle.mean(x=x, axis=0)
    mean = scatter(x, indices, dim=0, reduce="mean")
    x = x - mean[indices]
    return x


def draw_in_3dmol(mol: str, fmt: str = "xyz") -> py3Dmol.view:
    """画分子

    Args:
        mol (str): str content of molecule.
        fmt (str, optional): format. Defaults to "xyz".

    Returns:
        py3Dmol.view: output viewer
    """
    viewer = py3Dmol.view(1024, 576)
    viewer.addModel(mol, fmt)
    viewer.setStyle({"stick": {}, "sphere": {"radius": 0.36}})
    viewer.zoomTo()
    return viewer


def assemble_xyz(z: list, pos: Tensor) -> str:
    """将原子序数和位置组装成xyz格式

    Args:
        z (list): chemical elements
        pos (tensor): 3D coordinates

    Returns:
        str: xyz string
    """
    natoms = len(z)
    xyz = f"{natoms}\n\n"
    for _z, _pos in zip(z, pos.numpy()):
        xyz += f"{_z}\t" + "\t".join([str(x) for x in _pos]) + "\n"
    return xyz


num_layers = 2
hidden_channels = 8
in_hidden_channels = 4
num_radial = 4
model = LEFTNet(
    num_layers=num_layers,
    hidden_channels=hidden_channels,
    in_hidden_channels=in_hidden_channels,
    num_radial=num_radial,
    object_aware=False
)

print(f'Number of parameters {sum(p.size for p in model.parameters() if not p.stop_gradient) / 1e3:.2}K.')


h = paddle.rand(shape=[3, in_hidden_channels])
z = ['O', 'H', 'H']
pos = paddle.to_tensor([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0]
]).astype(dtype=default_float)
edge_index = paddle.to_tensor([
    [0, 0, 1, 1, 2, 2],
    [1, 2, 0, 2, 0, 1]
]).astype(dtype='int64')

_h, _pos, __ = model.forward(
    h=h,
    pos=remove_mean_batch(pos),
    edge_index=edge_index
)


rot = o3.rand_matrix()
pos_rot = paddle.matmul(x=pos, y=rot).astype(dtype=default_float)

_h_rot, _pos_rot, __ = model.forward(
    h=h,
    pos=remove_mean_batch(pos_rot),
    edge_index=edge_index
)

# 旋转后的h应该不变
print(paddle.max(x=paddle.abs(x=_h - _h_rot)))
# 旋转后的pos应该旋转
print(paddle.max(x=paddle.abs(x=paddle.matmul(x=_pos, y=rot).astype(dtype=default_float) - _pos_rot)))
print('At Cell 9, Done.')
# --- Cell 9 ---


ns = [3] + [2, 1]  # 反应物 3个原子 (H2O)，生成物 2个原子 (H2)，1个原子 (O自由基)
ntot = np.sum(ns)
mask = paddle.to_tensor(data=[0, 0, 0, 1, 1, 1])  # 用于区分反应物和生成物
z = ['O', 'H', 'H'] + ['H', 'H', 'O']
pos_react = paddle.to_tensor([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0]
]).astype(dtype=default_float)  # 方便起见，我们这里把H-O-H的角度设为90度
pos_prod = paddle.to_tensor([
    [0, 3, -0.4],
    [0, 3, 0.4],
    [0, -3, 0]
])  # 将H2和O自由基分开
pos = paddle.concat(x=[pos_react, pos_prod], axis=0)  # 拼接
h = paddle.rand(shape=[ntot, in_hidden_channels])

from oa_reactdiff.tests.model.utils import generate_full_eij, get_cut_graph_mask

edge_index = generate_full_eij(ntot)
print(edge_index)

_h, _pos, __ = model.forward(
    h=h,
    pos=remove_mean_batch(pos, mask),
    edge_index=edge_index
)

rot = o3.rand_matrix()
pos_react_rot = paddle.matmul(x=pos_react, y=rot).astype(dtype=default_float)

pos_rot = paddle.concat(
    x=[pos_react_rot, pos_prod],
    axis=0
)  # 拼接旋转过后的H2O和未旋转的H2和O自由基

_h_rot, _pos_rot, __ = model.forward(
    h=h,
    pos=remove_mean_batch(pos_rot, mask),
    edge_index=edge_index,
)

print(paddle.max(x=paddle.abs(x=_h - _h_rot)))  # 旋转后的h应该不变

_pos_rot_prime = paddle.concat(
    x=[paddle.matmul(x=_pos[:3], y=rot), _pos[3:]]
)
print(paddle.max(x=paddle.abs(x=_pos_rot_prime - _pos_rot)))  # 旋转后的pos应该旋转
print('At Cell 16, Done.')


model_oa = LEFTNet(
    num_layers=num_layers,
    hidden_channels=hidden_channels,
    in_hidden_channels=in_hidden_channels,
    num_radial=num_radial,
    object_aware=True,  # 使用object-aware模型
)


subgraph_mask = get_cut_graph_mask(edge_index, 3)  # 0-2是反应物的原子数
edge_index.T[paddle.where(subgraph_mask.squeeze() > 0)[0].squeeze()]

_h, _pos, __ = model_oa.forward(
    h=h,
    pos=remove_mean_batch(pos, mask),
    edge_index=edge_index,
    subgraph_mask=subgraph_mask
)

rot = o3.rand_matrix()
pos_react_rot = paddle.matmul(x=pos_react, y=rot).astype(dtype=default_float)

pos_rot = paddle.concat(x=[pos_react_rot, pos_prod], axis=0)

_h_rot, _pos_rot, __ = model_oa.forward(
    h=h,
    pos=remove_mean_batch(pos_rot, mask),
    edge_index=edge_index,
    subgraph_mask=subgraph_mask
)

print(paddle.max(x=paddle.abs(x=_h - _h_rot)))  # 旋转后的h应该不变
_pos_rot_prime = paddle.concat(x=[paddle.matmul(x=_pos[:3], y=rot), _pos[3:]])
print(paddle.max(x=paddle.abs(x=_pos_rot_prime - _pos_rot)))  # 旋转后的pos应该旋转
print('Cell 22, done')


from oa_reactdiff.trainer.pl_trainer import DDPMModule
from oa_reactdiff.dataset import ProcessedTS1x
from oa_reactdiff.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule
from oa_reactdiff.diffusion._normalizer import FEATURE_MAPPING
from oa_reactdiff.analyze.rmsd import batch_rmsd
from oa_reactdiff.utils.sampling_tools import assemble_sample_inputs, write_tmp_xyz

device = (
    paddle.CPUPlace()
    if not paddle.device.cuda.device_count() >= 1
    else str("cuda").replace("cuda", "gpu")
)


checkpoint = paddle.load("./pretrained-ts1x-diff.pdparams")
for key, value in checkpoint["state_dict"].items():
    checkpoint["state_dict"][key] = value.astype(paddle.get_default_dtype())

ddpm_trainer = DDPMModule(**checkpoint["hyper_parameters"])
missing_keys, unexpected_keys = ddpm_trainer.set_state_dict(checkpoint["state_dict"])
if missing_keys:
    print(
        f"There are missing keys when loading checkpoint: {missing_keys}, "
        "and corresponding parameters will be initialized by default."
    )
if unexpected_keys:
    print(
        f"There are redundant keys: {unexpected_keys}, "
        "and corresponding weights will be ignored."
    )

ddpm_trainer = ddpm_trainer.to(device)
noise_schedule: str = "polynomial_2"
timesteps: int = 150
precision: float = 1e-05
gamma_module = PredefinedNoiseSchedule(
    noise_schedule=noise_schedule, timesteps=timesteps, precision=precision
)
schedule = DiffSchedule(
    gamma_module=gamma_module, norm_values=ddpm_trainer.ddpm.norm_values
)
ddpm_trainer.ddpm.schedule = schedule
ddpm_trainer.ddpm.T = timesteps
ddpm_trainer = ddpm_trainer.to(device)
dataset = ProcessedTS1x(
    npz_path="./oa_reactdiff/data/transition1x/train.pkl",
    center=True,
    pad_fragments=0,
    device=device,
    zero_charge=False,
    remove_h=False,
    single_frag_only=False,
    swapping_react_prod=False,
    use_by_ind=True,
)
loader = paddle.io.DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=dataset.collate_fn,
)
itl = iter(loader)
idx = -1
for _ in range(4):
    representations, res = next(itl)
idx += 1
n_samples = representations[0]["size"].shape[0]
fragments_nodes = [repre["size"] for repre in representations]
conditions = paddle.to_tensor(data=[[0] for _ in range(n_samples)], place=device)
new_order_react = paddle.randperm(n=representations[0]["size"].item())

for k in ["pos", "one_hot", "charge"]:
    representations[0][k] = representations[0][k][new_order_react]
xh_fixed = [
    paddle.concat(
        x=[repre[feature_type].cast(paddle.get_default_dtype()) for feature_type in FEATURE_MAPPING],
        axis=1,
    )
    for repre in representations
]
out_samples, out_masks = ddpm_trainer.ddpm.inpaint(
    n_samples=n_samples,
    fragments_nodes=fragments_nodes,
    conditions=conditions,
    return_frames=1,
    resamplings=5,
    jump_length=5,
    timesteps=None,
    xh_fixed=xh_fixed,
    frag_fixed=[0, 2],
)

rmsds = batch_rmsd(fragments_nodes, out_samples[0], xh_fixed, idx=1)
write_tmp_xyz(
    fragments_nodes, out_samples[0], idx=[0, 1, 2], localpath="demo/inpainting"
)
rmsds = [min(1, _x) for _x in rmsds]
print([(ii, round(rmsd, 2)) for ii, rmsd in enumerate(rmsds)])
print(np.mean(rmsds))
print(np.median(rmsds))
print("Cell 33, Done")


from glob import glob
import plotly.express as px
from oa_reactdiff.analyze.rmsd import xyz2pmg, pymatgen_rmsd
from pymatgen.core import Molecule
from collections import OrderedDict


def draw_reaction(react_path: str, idx: int=0, prefix: str='gen'
    ) ->py3Dmol.view:
    """画出反应的的{反应物，过渡态，生成物}

    Args:
        react_path (str): path to the reaction.
        idx (int, optional): index for the generated reaction. Defaults to 0.
        prefix (str, optional): prefix for distinguishing true sample and generated structure.
            Defaults to "gen".

    Returns:
        py3Dmol.view: _description_
    """
    with open(f'{react_path}/{prefix}_{idx}_react.xyz', 'r') as fo:
        natoms = int(fo.readline()) * 3
    mol = f'{natoms}\n\n'
    for ii, t in enumerate(['react', 'ts', 'prod']):
        pmatg_mol = xyz2pmg(f'{react_path}/{prefix}_{idx}_{t}.xyz')
        pmatg_mol_prime = Molecule(species=pmatg_mol.atomic_numbers, coords
            =pmatg_mol.cart_coords + 8 * ii)
        mol += '\n'.join(pmatg_mol_prime.to(fmt='xyz').split('\n')[2:]) + '\n'
    viewer = py3Dmol.view(1024, 576)
    viewer.addModel(mol, 'xyz')
    viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
    viewer.zoomTo()
    return viewer


opt_ts_path = './demo/example-3/opt_ts/'
opt_ts_xyzs = glob(f'{opt_ts_path}/*ts.opt.xyz')
order_dict = {}
for xyz in opt_ts_xyzs:
    order_dict.update({int(xyz.split('/')[-1].split('.')[0]): xyz})
order_dict = OrderedDict(sorted(order_dict.items()))
opt_ts_xyzs = []
ind_dict = {}
for ii, v in enumerate(order_dict.values()):
    opt_ts_xyzs.append(v)
    ind_dict.update({ii: v})
n_ts = len(opt_ts_xyzs)
rmsd_mat = np.ones((n_ts, n_ts)) * -2.5
for ii in range(n_ts):
    for jj in range(ii + 1, n_ts):
        try:
            rmsd_mat[ii, jj] = np.log10(pymatgen_rmsd(opt_ts_xyzs[ii],
                opt_ts_xyzs[jj], ignore_chirality=True))
        except:
            print(ii, jj)
            pass
        rmsd_mat[jj, ii] = rmsd_mat[ii, jj]
from sklearn.cluster import KMeans


def reorder_matrix(matrix, n_clusters):
    row_clusters = KMeans(n_clusters=n_clusters).fit_predict(matrix)
    row_permutation = np.argsort(row_clusters)
    col_permutation = np.argsort(row_clusters)
    reordered_matrix = matrix[row_permutation][:, col_permutation]
    return reordered_matrix, row_permutation, row_clusters


n = n_ts
n_clusters = 6
reordered_matrix, row_permutation, row_clusters = reorder_matrix(rmsd_mat,
    n_clusters)
fig = px.imshow(reordered_matrix, color_continuous_scale='Oryel_r',
    range_color=[-2, -0.3])
fig.layout.font.update({'size': 18, 'family': 'Arial'})
fig.layout.update({'width': 650, 'height': 500})
fig.show()
import json
cluster_dict = {}
for ii, cluster in enumerate(row_clusters):
    cluster = str(cluster)
    if cluster not in cluster_dict:
        cluster_dict[cluster] = [ind_dict[ii]]
    else:
        cluster_dict[cluster] += [ind_dict[ii]]
cluster_dict = OrderedDict(sorted(cluster_dict.items()))
cluster_dict
print('Cell 42, Done')
xyz_path = './demo/CNOH/'
os.makedirs(xyz_path, exist_ok=True)
n_samples = 128
natm = 4
fragments_nodes = [paddle.to_tensor(data=[natm] * n_samples, place=device),
    paddle.to_tensor(data=[natm] * n_samples, place=device), paddle.
    to_tensor(data=[natm] * n_samples, place=device)]
conditions = paddle.to_tensor(data=[[0]] * n_samples, place=device)
h0 = assemble_sample_inputs(atoms=['C'] * 1 + ['O'] * 1 + ['N'] * 1 + ['H'] *
    1, device=device, n_samples=n_samples, frag_type=False)
h0 = [h0_i.cast(paddle.get_default_dtype()) for h0_i in h0]

out_samples, out_masks = ddpm_trainer.ddpm.sample(n_samples=n_samples,
    fragments_nodes=fragments_nodes, conditions=conditions, return_frames=1,
    timesteps=None, h0=h0)
write_tmp_xyz(fragments_nodes, out_samples[0], idx=[0, 1, 2], ex_ind=0,
    localpath=xyz_path)
idx = 10
assert idx < n_samples
views = draw_reaction(xyz_path, idx)

try:
    from openbabel import pybel
except:
    print('Since openbabel is not installed, we cannot execute the following code')
    exit()

from glob import glob
from pymatgen.io.xyz import XYZ
from oa_reactdiff.analyze.rmsd import pymatgen_rmsd


def xyz_to_smiles(fname: str) ->str:
    """将xyz格式的分子转换成smiles格式

    Args:
        fname (str): path to the xyz file.

    Returns:
        str: SMILES string.
    """
    mol = next(pybel.readfile('xyz', fname))
    smi = mol.write(format='can')
    return smi.split()[0].strip()


xyzfiles = glob(f'{xyz_path}/gen*_react.xyz') + glob(
    f'{xyz_path}/gen*_prod.xyz')
xyz_converter = XYZ(mol=None)
mol = xyz_converter.from_file(xyzfiles[0]).molecule
unique_mols = {xyzfiles[0]: mol}
for _xyzfile in xyzfiles:
    _mol = xyz_converter.from_file(_xyzfile).molecule
    min_rmsd = 100
    for _, mol in unique_mols.items():
        rmsd = pymatgen_rmsd(mol, _mol, ignore_chirality=True, threshold=0.5)
        min_rmsd = min(min_rmsd, rmsd)
    if min_rmsd > 0.1:
        unique_mols.update({_xyzfile: _mol})
len(unique_mols)
unique_idx = []
unique_smiles = []
idx = 0
for file in unique_mols:
    smi = xyz_to_smiles(file)
    if smi not in unique_smiles and not '.' in smi:
        unique_smiles.append(smi)
        unique_idx.append(idx)
    idx += 1
unique_idx, unique_smiles
unique_paths = {}
path_index = {}
for ii in range(n_samples):
    r_xyz = f'{xyz_path}/gen_{ii}_react.xyz'
    p_xyz = f'{xyz_path}/gen_{ii}_prod.xyz'
    path = set([xyz_to_smiles(r_xyz), xyz_to_smiles(p_xyz)])
    use = True
    for smi in path:
        if smi not in unique_smiles:
            use = False
    if not path in unique_paths.values() and len(path) > 1 and use:
        unique_paths[ii] = path
    if not (len(path) > 1 and use):
        continue
    sorted_smi = ' & '.join(list(sorted(path)))
    if sorted_smi not in path_index:
        path_index[sorted_smi] = [ii]
    else:
        path_index[sorted_smi] += [ii]
mols_in_paths = []
for k, v in unique_paths.items():
    for _v in v:
        if not _v in mols_in_paths:
            mols_in_paths.append(_v)
mols_in_paths, len(mols_in_paths)
print('All Done. Succeed!')
