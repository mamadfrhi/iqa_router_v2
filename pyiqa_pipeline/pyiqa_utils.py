import os
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import pyiqa


# Dataset-specific PyIQA configuration to ensure correct paths and splits are loaded
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "koniq10k": {
        "dataroot_target": "./datasets/koniq10k/512x384",
        "meta_info_file": "./datasets/meta_info/meta_info_KonIQ10kDataset.csv",
    },
    "spaq": {
        "dataroot_target": "./datasets/SPAQ/TestImage",
        "meta_info_file": "./datasets/meta_info/meta_info_SPAQDataset.csv",
    },
    "livec": {
        "dataroot_target": "./datasets/LIVEC/",
        "meta_info_file": "./datasets/meta_info/meta_info_LIVEChallengeDataset.csv",
    },
    "tid2013": {
        "dataroot_target": "./datasets/tid2013/distorted_images",
        "dataroot_ref": "./datasets/tid2013/reference_images",
        "meta_info_file": "./datasets/meta_info/meta_info_TID2013Dataset.csv",
    },
}


def load_labels_df(
    dataset: str,
    data_root: str,
    phase: str = "test",
    split_index: str = "",
    mos_normalize: bool = False,
) -> pd.DataFrame:
    def _resolve_path_case(path: str) -> str:
        if os.path.exists(path):
            return path
        dir_name, base = os.path.split(path)
        if base:
            alt = os.path.join(dir_name, base.lower())
            if alt != path and os.path.exists(alt):
                return alt
            root, ext = os.path.splitext(base)
            if ext:
                alt = os.path.join(dir_name, f"{root}{ext.lower()}")
                if alt != path and os.path.exists(alt):
                    return alt
        return path

    data_root_path = Path(data_root)
    dataset_opts = {
        "phase": phase,
        "mos_normalize": bool(mos_normalize),
    }
    if split_index:
        dataset_opts["split_index"] = split_index
    
    # Apply dataset-specific PyIQA configs to ensure correct paths and meta_info are used
    if dataset in DATASET_CONFIGS:
        for key, value in DATASET_CONFIGS[dataset].items():
            if key not in dataset_opts:  # Don't override explicitly passed options
                dataset_opts[key] = value

    ds = pyiqa.load_dataset(dataset, data_root=data_root_path, dataset_opts=dataset_opts)

    # CRITICAL: Validate that the correct split was loaded
    count = len(ds)
    if count == 0:
        raise ValueError(
            f"Dataset {dataset} returned 0 samples for phase='{phase}', split_index='{split_index}'. "
            f"This likely means the split doesn't exist or is misconfigured. "
            f"Check that the split_index is valid for the dataset."
        )
    
    # Validate that at least the first sample's paths exist (sanity check)
    if hasattr(ds, 'paths_mos') and len(ds.paths_mos) > 0:
        sample = ds.paths_mos[0]
        if len(sample) == 2:
            img_path = _resolve_path_case(str(sample[0]))
            if not os.path.exists(img_path):
                raise ValueError(
                    f"Dataset {dataset} phase='{phase}' split_index='{split_index}': "
                    f"First image path does not exist: {img_path}"
                )
        elif len(sample) == 3:
            ref_path = _resolve_path_case(str(sample[0]))
            img_path = _resolve_path_case(str(sample[1]))
            if not os.path.exists(img_path):
                raise ValueError(
                    f"Dataset {dataset} phase='{phase}' split_index='{split_index}': "
                    f"First image path does not exist: {img_path}"
                )
            if not os.path.exists(ref_path):
                raise ValueError(
                    f"Dataset {dataset} phase='{phase}' split_index='{split_index}': "
                    f"First reference path does not exist: {ref_path}"
                )

    rows = []
    for item in ds.paths_mos:
        if len(item) == 2:
            image_path = _resolve_path_case(str(item[0]))
            ref_path = ""
            mos = float(item[1])
        elif len(item) == 3:
            ref_path = _resolve_path_case(str(item[0]))
            image_path = _resolve_path_case(str(item[1]))
            mos = float(item[2])
        else:
            raise ValueError("Unsupported paths_mos format from dataset")

        rows.append({
            "image_path": str(image_path),
            "ref_path": str(ref_path),
            "mos": mos,
            "image": os.path.basename(str(image_path)),
        })

    return pd.DataFrame(rows)
