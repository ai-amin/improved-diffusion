from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Added for diffraction dataset support
from pathlib import Path
import torch
import torch.nn.functional as F
try:
    import h5py  # optional dependency for diffraction data
except ImportError:  # pragma: no cover
    h5py = None


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


# ---------------------- Diffraction Dataset Support ---------------------- #
class DiffractionDataset(Dataset):
    """
    Dataset for diffraction pattern images stored in HDF5 files.

    Expected HDF5 structure:
      - dataMeas: tensor/array with shape [..., N] where last dim indexes samples
      - dataProbe: tensor/array (shared across samples)
      - dataPots: tensor/array with shape [..., N] (structure factors/potentials)

    Args:
        root (str or Path): directory containing .h5/.hdf5 files (recursively).
        resolution (int): output spatial resolution (square assumed).
        start_id (int): starting global sample index.
        end_id (int or None): exclusive end index (None means all).
        data_type (str): which component to return: 'cbed' (dataMeas), 'probe', 'pot'.
        normalize (bool): scale data to [-1,1] per sample.
        augment (bool): placeholder for future augmentations.
    """

    def __init__(
        self,
        root,
        resolution=256,
        start_id=0,
        end_id=None,
        data_type="cbed",
        normalize=True,
        augment=False,
    ):
        super().__init__()
        if h5py is None:
            raise ImportError(
                "h5py is required for DiffractionDataset. Install with `pip install h5py`."
            )
        self.root = Path(root)
        self.resolution = resolution
        self.data_type = data_type
        self.normalize = normalize
        self.augment = augment

        patterns = ("*.h5", "*.H5", "*.hdf5", "*.HDF5")
        paths = []
        for pat in patterns:
            paths.extend(self.root.rglob(pat))
        # unique & existing
        self.file_paths = sorted({p.resolve() for p in paths if p.is_file()})
        if not self.file_paths:
            raise ValueError(f"No HDF5 files (*.h5|*.hdf5) found under {self.root}")

        self.file_paths = self._filter_valid_files(self.file_paths)
        if not self.file_paths:
            raise ValueError("No valid HDF5 files with required datasets were found.")

        self.sample_indices = []  # list of tuples (file_path, local_index)
        for fp in self.file_paths:
            try:
                with h5py.File(fp, "r") as f:
                    num_samples = f["dataMeas"].shape[-1]
                    for i in range(num_samples):
                        self.sample_indices.append((fp, i))
            except Exception as e:  # pragma: no cover
                print(f"Skipping file during enumeration {fp}: {e}")

        if end_id is None:
            end_id = len(self.sample_indices)
        self.sample_indices = self.sample_indices[start_id:end_id]
        if not self.sample_indices:
            raise ValueError("No samples selected after applying start/end indices.")

    def _filter_valid_files(self, file_paths):
        required = {"dataMeas", "dataProbe", "dataPots"}
        valid = []
        for fp in file_paths:
            try:
                with h5py.File(fp, "r") as f:
                    keys = set(f.keys())
                    if required.issubset(keys):
                        valid.append(fp)
                    else:
                        print(f"Skipping file missing required keys: {fp}")
            except Exception as e:  # pragma: no cover
                print(f"Skipping corrupted file {fp}: {e}")
        return valid

    def _load_sample(self, fp, local_idx):
        with h5py.File(fp, "r") as f:
            data_meas = torch.from_numpy(f["dataMeas"][..., local_idx]).float()
            data_probe = torch.from_numpy(f["dataProbe"][...]).float()
            data_pots = torch.from_numpy(f["dataPots"][..., local_idx]).float()

        # log transform potentials for stability
        data_pots = torch.log(data_pots + 1e-6)

        # replace NaNs/Infs
        data_meas = torch.nan_to_num(data_meas)
        data_probe = torch.nan_to_num(data_probe)
        data_pots = torch.nan_to_num(data_pots)

        return data_meas, data_probe, data_pots

    def _normalize(self, tensor):
        if not self.normalize:
            return tensor
        min_v = tensor.min()
        max_v = tensor.max()
        if max_v > min_v:
            return 2 * (tensor - min_v) / (max_v - min_v) - 1
        return torch.zeros_like(tensor)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        fp, local_idx = self.sample_indices[idx]
        data_meas, data_probe, data_pots = self._load_sample(fp, local_idx)

        if self.data_type == "cbed":
            data = data_meas
        elif self.data_type == "probe":
            data = data_probe
        elif self.data_type == "pot":
            data = data_pots
        else:
            raise ValueError(f"Unknown data_type {self.data_type}")

        if data.dim() == 2:  # ensure [C,H,W]
            data = data.unsqueeze(0)

        if data.shape[-2:] != (self.resolution, self.resolution):
            data = F.interpolate(
                data.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        data = self._normalize(data)
        # If model expects 3 channels but diffraction data is single-channel, repeat.
        if data.shape[0] == 1:
            data = data.repeat(3, 1, 1)
        # placeholder for future augmentations
        return data, {}


def load_diffraction_data(
    *,
    data_dir,
    batch_size,
    image_size,
    data_type="cbed",
    start_id=0,
    end_id=None,
    deterministic=False,
):
    """Create an infinite generator over diffraction data batches.

    Returns (images, kwargs) where kwargs is currently empty dict.
    """
    dataset = DiffractionDataset(
        root=data_dir,
        resolution=image_size,
        data_type=data_type,
        start_id=start_id,
        end_id=end_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )
    while True:
        for batch, cond in loader:
            # Keep as torch.Tensor so downstream training code can call .to(device)
            # (Original image loader yields tensors via DataLoader automatic conversion.)
            yield batch, cond
