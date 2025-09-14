"""Dataset class for SUN360 datasets."""
import pathlib as pl
import random

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


class SUN360(Dataset):
    """Dataset class for SUN360 datasets."""

    def __init__(self, folder: str, mappings_folder: str) -> None:
        """Initialize a new instance of SUN360.

        Args:
            folder (str): The folder containing the images.
            mappings_folder (str): The folder containing the mappings.

        """
        self.__mean = [0.485, 0.456, 0.406]
        self.__std = [0.229, 0.224, 0.225]
        self.folder = pl.Path(folder)
        self.mappings_folder = pl.Path(mappings_folder)
        self.filenames = list(self.folder.glob("*.jpg"))
        self.gts = []
        anns_fname = self.mappings_folder / "gt.csv"
        with anns_fname.open() as f:
            lines = f.readlines()

        for lin in lines:
            stripped_lin = lin.strip().split(",")
            angles = [stripped_lin[0], float(stripped_lin[1]), float(stripped_lin[2])]
            self.gts.append(angles)

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, item: int) -> tuple:
        """Get an item from the dataset.

        Args:
            item (int): The index of the item to get.

        Returns:
            tuple: A tuple containing the image and the label

        Raises:
            ValueError: If the image could not be read.
        """
        filename = self.filenames[item]

        img_np = cv2.imread(str(filename))

        if img_np is None:
            message = f"Could not read image {filename}"
            raise ValueError(message)

        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np = img_np / 255
        chosen_rotation = random.randint(0, len(self.gts) - 1)  # noqa: S311
        lut_fname, rx, ry = self.gts[chosen_rotation]
        r = Rotation.from_euler("zxy", [0, rx, ry], degrees=True)
        rot_np = r.apply(np.array([0, 0, 1]))

        mapping = np.load(self.mappings_folder / lut_fname)

        img_np = cv2.remap(img_np, mapping, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        img_np = (img_np - self.__mean) / self.__std

        img_np = cv2.resize(img_np, (442, 221))

        img_pt = torch.from_numpy(img_np).permute(2, 0, 1).float()
        label = torch.from_numpy(rot_np).float()

        return img_pt, label
