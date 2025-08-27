# datasets_finetune/allweather_finetune.py
import os, re, random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

# 파일명 규칙:
#  D-201101_O8102DGT_001_0001.jpg  -> GT
#  D-201101_O8102D01_001_0006.jpg  -> 입력
_PAT = re.compile(
    r'^(?P<prefix>[A-Z]-\d{6})_(?P<scene>O\d{4})D(?P<tag>GT|\d{2,})_(?P<group>\d{3})_(?P<frame>\d{4})\.(jpg|png)$',
    re.IGNORECASE
)

def _scene_key_from_name(name: str):
    m = _PAT.match(name)
    if not m:
        return None, None
    d = m.groupdict()
    key = f"{d['prefix']}_{d['scene']}_{d['group']}"  # 장면 단위 키
    is_gt = (d['tag'].upper() == "GT")
    return key, is_gt

def _pair_by_same_stem(in_dir: Path, gt_dir: Path) -> List[Tuple[str, str]]:
    """input/gt 파일명이 동일 stem일 때 1:1 매칭"""
    pairs = []
    gt_map = {p.stem: str(p) for p in gt_dir.glob("*")}
    for x in in_dir.glob("*"):
        g = gt_map.get(x.stem)
        if g:
            pairs.append((str(x), g))
    return pairs

def _pair_by_scene_rule(in_dir: Path, gt_dir: Path) -> List[Tuple[str, str]]:
    """scene 규칙(DGT vs D##, scene_key 동일)으로 1:N 매칭"""
    pairs = []
    gt_map: Dict[str, str] = {}
    for g in gt_dir.glob("*"):
        key, is_gt = _scene_key_from_name(g.name)
        if key and is_gt and key not in gt_map:
            gt_map[key] = str(g)
    for x in in_dir.glob("*"):
        key, is_gt = _scene_key_from_name(x.name)
        if not key or is_gt:
            continue
        g = gt_map.get(key)
        if g:
            pairs.append((str(x), g))
    return pairs

def _build_pairs(root: Path, split: str, tasks: List[str]) -> List[Tuple[str, str, str]]:
    """
    dataset/{dust,fog,rain}/{train,val,test}/{input,gt}/*
    -> (input_path, gt_path, task) 리스트 생성
    """
    out: List[Tuple[str, str, str]] = []
    for task in tasks:
        in_dir = root / task / split / "input"
        gt_dir = root / task / split / "gt"
        if not in_dir.exists() or not gt_dir.exists():
            continue
        pairs = _pair_by_same_stem(in_dir, gt_dir)
        if len(pairs) == 0:
            pairs = _pair_by_scene_rule(in_dir, gt_dir)
        out.extend([(ip, gp, task) for (ip, gp) in pairs])
    return out

class AllWeatherDatasetFT(Dataset):
    """
    폴더 직접 스캔하는 파인튜닝용 데이터셋.
    - dataset/{dust,fog,rain}/{train,val,test}/{input,gt}/*
    - 파일명이 같으면 same-stem, 다르면 scene 규칙(DGT/D##)으로 매칭
    - __getitem__: (concat(CxHxW, input||gt), basename) 반환 (원 레포 호환)
    """
    def __init__(self, data_dir: str, split: str = "train", image_size: int = 256,
                 n_patches: int = 1, parse_patches: bool = True, tasks: List[str] = None):
        super().__init__()
        self.root = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.n = n_patches
        self.parse_patches = parse_patches
        self.tasks = tasks or ["dust", "fog", "rain"]

        self.pairs = _build_pairs(self.root, split, self.tasks)
        if len(self.pairs) == 0:
            raise RuntimeError(f"[AllWeatherFT] No pairs found: {self.root} split={split} tasks={self.tasks}")

        self.tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

        print(f"[AllWeatherFT] split={split} tasks={self.tasks} pairs={len(self.pairs)}")

    @staticmethod
    def _get_crop_params(img: Image.Image, out_hw: Tuple[int, int], n: int):
        w, h = img.size
        th, tw = out_hw
        if w == tw and h == th:
            return [0]*n, [0]*n, th, tw
        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def _n_random_crops(img: Image.Image, xs: List[int], ys: List[int], h: int, w: int):
        return [img.crop((y, x, y + w, x + h)) for x, y in zip(xs, ys)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, gp, task = self.pairs[idx]
        x_img = Image.open(ip).convert("RGB")
        y_img = Image.open(gp).convert("RGB")

        if self.parse_patches:
            xs, ys, h, w = self._get_crop_params(x_img, (self.image_size, self.image_size), self.n)
            x_crops = self._n_random_crops(x_img, xs, ys, h, w)
            y_crops = self._n_random_crops(y_img, xs, ys, h, w)
            outs = [torch.cat([self.tf(xc), self.tf(yc)], dim=0) for xc, yc in zip(x_crops, y_crops)]
            return torch.stack(outs, dim=0), os.path.basename(ip)
        else:
            # whole-image: 16배수 리사이즈(원 레포 호환)
            wd_new, ht_new = x_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new)); ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new)); wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0)); ht_new = int(16 * np.ceil(ht_new / 16.0))
            x_img = x_img.resize((wd_new, ht_new), Image.Resampling.LANCZOS)
            y_img = y_img.resize((wd_new, ht_new), Image.Resampling.LANCZOS)
            return torch.cat([T.ToTensor()(x_img), T.ToTensor()(y_img)], dim=0), os.path.basename(ip)

class AllWeather:
    """
    원 AllWeather 인터페이스 유지(get_loaders).
    config.data.data_dir = ".../dataset"
    """
    def __init__(self, config):
        self.config = config
        self.transforms = T.Compose([T.ToTensor()])

    def get_loaders(self, parse_patches=True):
        # 우리 데이터: train/val/test 모두 동일 폴더 규칙
        data_dir = self.config.data.data_dir
        img_size = self.config.data.image_size
        tasks = getattr(self.config.data, "tasks", ["dust", "fog", "rain"])

        train_ds = AllWeatherDatasetFT(
            data_dir, split="train", image_size=img_size,
            n_patches=self.config.training.patch_n,
            parse_patches=parse_patches, tasks=tasks
        )
        val_ds = AllWeatherDatasetFT(
            data_dir, split="val", image_size=img_size,
            n_patches=self.config.training.patch_n,
            parse_patches=parse_patches, tasks=tasks
        )

        # whole-image 모드에선 배치=1 강제(원 코드 호환)
        if not parse_patches:
            self.config.training.batch_size = 1
            if not hasattr(self.config, "sampling"):
                import argparse as _ap
                self.config.sampling = _ap.Namespace(batch_size=1, last_only=True)
            else:
                self.config.sampling.batch_size = 1

        num_workers = getattr(self.config.data, "num_workers", 0)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=parse_patches
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=getattr(self.config.sampling, "batch_size", 1),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader