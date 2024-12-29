import sys

from utils import paddle_aux
import paddle
import pickle
import numpy as np

ATOM_MAPPING = {(1): 0, (6): 1, (7): 2, (8): 3, (9): 4}
n_element = len(list(ATOM_MAPPING.keys()))


class BaseDataset(paddle.io.Dataset):

    def __init__(
        self,
        npz_path,
        center=True,
        zero_charge=False,
        device="cpu",
        remove_h=False,
        n_fragment=3,
    ) -> None:
        super().__init__()
        if ".npz" in str(npz_path):
            with np.load(npz_path, allow_pickle=True) as f:
                data = {key: val for key, val in f.items()}
        elif ".pkl" in str(npz_path):
            data = pickle.load(open(npz_path, "rb"))
        else:
            raise ValueError("data file should be either .npz or .pkl")
        self.raw_dataset = data
        self.n_samples = -1
        self.data = {}
        self.n_fragment = n_fragment
        self.remove_h = remove_h
        self.zero_charge = zero_charge
        self.center = center
        self.device = device

    def __len__(self):
        return len(self.data["size_0"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    @staticmethod
    def collate_fn(batch):
        sizes = []
        for k in batch[0].keys():
            if "size" in k:
                sizes.append(int(k.split("_")[-1]))
        n_fragment = len(sizes)
        out = [{} for _ in range(n_fragment)]
        res = {}
        for prop in batch[0].keys():
            if prop not in ["condition", "target", "rmsd", "ediff"]:
                idx = int(prop.split("_")[-1])
                _prop = prop.replace(f"_{idx}", "")
            if "size" in prop:
                out[idx][_prop] = paddle.to_tensor(
                    data=[x[prop] for x in batch], place=batch[0][prop].place
                )
            elif "mask" in prop:
                out[idx][_prop] = paddle.concat(
                    x=[
                        (i * paddle.ones(shape=len(x[prop])).astype(dtype="int64"))
                        for i, x in enumerate(batch)
                    ],
                    axis=0,
                )
            elif prop in ["condition", "target", "rmsd", "ediff"]:
                res[prop] = paddle.concat(x=[x[prop] for x in batch], axis=0)
            else:
                out[idx][_prop] = paddle.concat(x=[x[prop] for x in batch], axis=0)
        if len(list(res.keys())) == 1:
            return out, res["condition"]
        return out, res

    def patch_dummy_molecules(self, idx):
        self.data[f"size_{idx}"] = paddle.ones_like(x=self.data[f"size_0"])
        self.data[f"pos_{idx}"] = [
            paddle.to_tensor(data=[[0, 0, 0]], place=self.device)
            for _ in range(self.n_samples)
        ]
        self.data[f"one_hot_{idx}"] = [
            paddle.to_tensor(data=[0], place=self.device) for _ in range(self.n_samples)
        ]
        self.data[f"one_hot_{idx}"] = [
            paddle.nn.functional.one_hot(num_classes=n_element, x=_z).astype("int64")
            for _z in self.data[f"one_hot_{idx}"]
        ]
        if self.zero_charge:
            self.data[f"charge_{idx}"] = [
                paddle.zeros(shape=(1, 1), dtype="int64") for _ in range(self.n_samples)
            ]
        else:
            self.data[f"charge_{idx}"] = [
                paddle.ones(shape=(1, 1), dtype="int64") for _ in range(self.n_samples)
            ]
        self.data[f"mask_{idx}"] = [
            paddle.zeros(shape=(1,), dtype="int64") for _ in range(self.n_samples)
        ]

    def process_molecules(
        self, dataset_name, n_samples, idx, append_charge=None, position_key="positions"
    ):
        data = getattr(self, dataset_name)
        self.data[f"size_{idx}"] = paddle.to_tensor(
            data=data["num_atoms"], place=self.device
        )
        self.data[f"pos_{idx}"] = [
            paddle.to_tensor(
                data=data[position_key][ii][: data["num_atoms"][ii]],
                dtype="float32",
                place=self.device,
            )
            for ii in range(n_samples)
        ]
        self.data[f"one_hot_{idx}"] = [
            paddle.to_tensor(
                data=[
                    ATOM_MAPPING[_at]
                    for _at in data["charges"][ii][: data["num_atoms"][ii]]
                ],
                place=self.device,
            )
            for ii in range(n_samples)
        ]
        self.data[f"one_hot_{idx}"] = [
            paddle.nn.functional.one_hot(num_classes=n_element, x=_z).astype("int64")
            for _z in self.data[f"one_hot_{idx}"]
        ]
        if self.zero_charge:
            self.data[f"charge_{idx}"] = [
                paddle.zeros(shape=(_size, 1), dtype="int64")
                for _size in data["num_atoms"]
            ]
        elif append_charge is None:
            self.data[f"charge_{idx}"] = [
                paddle.to_tensor(
                    data=data["charges"][ii][: data["num_atoms"][ii]], place=self.device
                ).reshape([-1, 1])
                for ii in range(n_samples)
            ]
        else:
            self.data[f"charge_{idx}"] = [
                paddle.concat(
                    x=[
                        paddle.to_tensor(
                            data=data["charges"][ii][: data["num_atoms"][ii]],
                            place=self.device,
                        ).reshape([-1, 1]),
                        paddle.to_tensor(
                            data=[append_charge for _ in range(data["num_atoms"][ii])],
                            place=self.device,
                        ).reshape([-1, 1]),
                    ],
                    axis=1,
                )
                for ii in range(n_samples)
            ]
        self.data[f"mask_{idx}"] = [
            paddle.zeros(shape=(_size,), dtype="int64") for _size in data["num_atoms"]
        ]
        if self.center:
            self.data[f"pos_{idx}"] = [
                (pos - paddle.mean(x=pos, axis=0)) for pos in self.data[f"pos_{idx}"]
            ]
