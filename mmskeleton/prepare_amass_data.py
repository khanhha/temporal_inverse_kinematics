import numpy as np
from sklearn.model_selection import train_test_split
import csv
from pathlib import Path

if __name__ == "__main__":
    amass_dir = Path('/media/F/datasets/amass/motion_data')
    cmu_sample_path = Path('/media/F/thesis/motion_capture/data/cmu_action_list.csv')
    with open(cmu_sample_path, 'rt') as file:
        lines = file.readlines()
        cmu_samples = [lline.split('|')[0] for lline in lines]
        cmu_samples = set(cmu_samples)

    all_amass_paths = [apath for apath in amass_dir.rglob('*.npz') if apath.stem.endswith('_poses')]

    def is_cmu_path(path):
        return 'CMU' in str(path) and "_".join(path.stem.split('_')[:2]) in cmu_samples

    amass_paths = [apath for apath in all_amass_paths if is_cmu_path(apath)]
    print(len(amass_paths))
    n_paths = len(amass_paths)
    train_idxs, valid_idxs = train_test_split(np.arange(n_paths), test_size=0.1)
    train_paths = [str(amass_paths[idx]) for idx in train_idxs]
    valid_paths = [str(amass_paths[idx]) for idx in valid_idxs]

    out_dir = Path('/media/F/datasets/amass/ik_model')
    with open(out_dir / 'train.csv', 'w') as file:
        for ll in train_paths:
            file.write(f'{ll}\n')
    with open(out_dir / 'valid.csv', 'w') as file:
        for ll in valid_paths:
            file.write(f'{ll}\n')
