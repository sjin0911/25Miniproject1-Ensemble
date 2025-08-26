import os, re, glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# 파일명 끝 "...(F?GT_)?NNN_####"에서 NNN 추출
_ID_PAT = re.compile(r'(?:F?GT_)?(?P<mid>\d{3})_(?P<seq>\d{4})$', re.IGNORECASE)

def id_from_name_gt_rule(name: str) -> str:
    stem = Path(name).stem
    m = _ID_PAT.search(stem)
    if not m:
        raise ValueError(f"[id_from_name_gt_rule] 이름 규칙 불일치: {name}")
    return m.group("mid")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_list(txt_path):
    with open(txt_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

import re

def extract_key(fname: str):
    """
    파일명에서 '..._NNN_####.jpg' 패턴 중 NNN까지만 추출
    예: D-210801_O8103R01_003_0005.jpg -> "D-210801_O8103R01_003"
    """
    stem = Path(fname).stem
    m = re.match(r"(.+_\d{3})_\d{4}$", stem)
    if not m:
        raise ValueError(f"Unexpected filename pattern: {fname}")
    return m.group(1)

class RestoreFinetuneDataset(Dataset):
    def __init__(self, root, tasks=("rain",), split="train",
                 transform_train=None, transform_eval=None,
                 mode="train"):
        self.root = Path(root)
        self.tasks = [tasks] if isinstance(tasks, str) else list(tasks)
        self.split = split
        self.mode = mode
        self.t_train = transform_train
        self.t_eval  = transform_eval

        self.samples = []

        for task in self.tasks: 
            gt_txt = self.root / f"GT_{task}_{split}.txt"
            notgt_txt = self.root / f"notGT_{task}_{split}.txt"

            gt_files = load_list(gt_txt)
            in_files = load_list(notgt_txt)

            half = len(in_files)//2
            in_files = in_files[:half]

            gt_dir = self.root / task / split / "gt"
            in_dir = self.root / task / split / "input"

            # key -> GT path
            gt_map = {}
            for gname in gt_files:
                mid = id_from_name_gt_rule(gname)
                gt_map[mid] = gname  # 같은 mid면 마지막 GT 하나로 매칭

            # input 파일도 mid 추출해서 GT와 매칭
            for iname in in_files:
                try:
                    mid = id_from_name_gt_rule(iname)
                except ValueError:
                    continue
                if mid in gt_map:
                    gp = gt_dir / gt_map[mid]
                    ip = in_dir / iname
                    if gp.exists() and ip.exists():
                        self.samples.append((str(ip), str(gp), task, mid))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples from txt under {self.root} with tasks={self.tasks}, split={self.split}"
            )

    def __len__(self):
        return len(self.samples)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:  # gray -> 3채널
            arr = np.stack([arr]*3, axis=-1)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx):
        in_path, gt_path, task, iid = self.samples[idx]
        x = Image.open(in_path).convert("RGB")
        y = Image.open(gt_path).convert("RGB")

        meta = {
            "task": task,
            "id": iid,
            "in_path": str(in_path),
            "gt_path": str(gt_path),
        }

        if self.mode == "train" and self.t_train:
            # t_train(x_pil, y_pil) -> (x1_pil, x2_pil, y1_pil) 가정
            x1_pil, x2_pil, y1_pil = self.t_train(x, y)
            x1 = self._to_tensor(x1_pil)
            x2 = self._to_tensor(x2_pil)
            y1 = self._to_tensor(y1_pil)
            return (x1, x2), y1, meta

        x_t = self._to_tensor(x)
        y_t = self._to_tensor(y)
        if self.t_eval is not None and self.mode != "train":
            x_t, y_t = self.t_eval(x_t, y_t)
        return x_t, y_t, meta