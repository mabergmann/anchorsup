import pathlib as pl

import cv2
import hydra
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from upright_anchor import rotation


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg):
    output_folder = pl.Path(cfg.data.lut_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    train_folder = pl.Path(cfg.data.train_path)
    sample_image = cv2.imread(train_folder / "1.jpg")

    ann_lines = []

    phi = (1 + np.sqrt(5)) / 2

    n_rotations = 10000

    for i in tqdm(range(n_rotations)):
        rx = i * 360 / n_rotations
        ry = (i * 180 / phi) % 180
        ry -= 90
        rx -= 180

        r = R.from_euler("zxy", [0, rx, ry], degrees=True)
        mappings = rotation.getMapping(sample_image.copy(), r.as_matrix())
        fname = output_folder / f"{i}.npy"
        np.save(fname, mappings)
        line = f"{fname.name},{rx},{ry}\n"
        ann_lines.append(line)

    with open(str(output_folder / "gt.csv"), "w") as f:
        f.writelines(ann_lines)
