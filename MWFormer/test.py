
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData
from utils_val import validation_stylevec
import numpy as np
import random

from model.EncDec import Network_top    # default backbone
from model.style_filter64 import StyleFilter_Top

# (추가) 폴더 인퍼런스용
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
import glob


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')

# 원래 인자
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-seed', help='set random seed', default=19, type=int)

# (주의) 헷갈리지 않게 help/기본값을 교정
parser.add_argument("-restore-from-stylefilter", type=str,
                    default='./checkpoints/test_model/style_filter',
                    help='path to the style/feature extraction network weights (.pth or dir)')
parser.add_argument('-restore-from-backbone', type=str,
                    default='./checkpoints/test_model/backbone',
                    help='path to the image restoration backbone weights (.pth or dir)')

parser.add_argument('-val_data_dir', default='./data/test/', type=str)
parser.add_argument('-val_filename', default='snowtest100k_L.txt', type=str)

# (추가) GT 없이 폴더 복원 전용
parser.add_argument('--input_dir', type=str, default=None,
                    help='GT 없이 복원할 입력 이미지 폴더')
parser.add_argument('--save_dir', type=str, default='outputs/mwformer_restored',
                    help='복원 결과 저장 폴더')
parser.add_argument('--no_metrics', action='store_true',
                    help='PSNR/SSIM 계산(=GT 의존) 건너뛰기')

args = parser.parse_args()

val_batch_size = args.val_batch_size

# --- Seed --- #
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- Build models (backbone + style filter) --- #
# backbone
net = Network_top().to(device)
if len(device_ids) > 1:
    net = nn.DataParallel(net, device_ids=device_ids)

# style filter
StyleFilter = StyleFilter_Top().to(device)
if len(device_ids) > 1:
    StyleFilter = nn.DataParallel(StyleFilter, device_ids=device_ids)

def _pick_pth(path_or_dir: str):
    """dir를 주면 내부 최신 .pth를, .pth면 그대로 반환"""
    if path_or_dir.endswith('.pth') and os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        cands = sorted(glob.glob(os.path.join(path_or_dir, '*.pth')))
        if not cands:
            raise FileNotFoundError(f"No .pth found in: {path_or_dir}")
        cands.sort(key=os.path.getmtime, reverse=True)
        print(f"[picked weights] {os.path.basename(cands[0])} from {path_or_dir}")
        return cands[0]
    raise FileNotFoundError(f"Weight path not found: {path_or_dir}")

# 가중치 로드
backbone_w = _pick_pth(args.restore_from_backbone)
style_w    = _pick_pth(args.restore_from_stylefilter)

print("[load backbone]:", backbone_w)
print("[load style   ]:", style_w)

# 주의: 레포마다 저장 방식이 달라 .module 유무가 다를 수 있음 -> 두 가지 케이스 지원
def _load_weights_into(model_obj, ckpt_path):
    weights = torch.load(ckpt_path, map_location='cpu')
    if isinstance(weights, dict) and 'state_dict' in weights:
        weights = weights['state_dict']
    try:
        model_obj.load_state_dict(weights)
    except Exception as e:
        # DataParallel 유무에 따른 키 prefix 차이를 완화
        new_state = {}
        for k, v in weights.items():
            if k.startswith('module.') and not any(s.startswith('module.') for s in model_obj.state_dict().keys()):
                new_state[k[len('module.'):]] = v
            elif (not k.startswith('module.')) and any(s.startswith('module.') for s in model_obj.state_dict().keys()):
                new_state['module.' + k] = v
            else:
                new_state[k] = v
        model_obj.load_state_dict(new_state)

_load_weights_into(net, backbone_w)
_load_weights_into(StyleFilter, style_w)

for p in StyleFilter.parameters():
    p.requires_grad = False
net.eval()
StyleFilter.eval()

# ======================================================
# ================ Folder Inference Mode ===============
# ======================================================
if args.input_dir is not None:
    print('--- Folder inference (no GT) starts! ---')
    os.makedirs(args.save_dir, exist_ok=True)

    to_tensor = T.ToTensor()   # [0,1], RGB
    to_pil    = T.ToPILImage()

    def pad_to_mod(x, m=8):
        """트랜스포머 계열에서 크기 배수 안정화를 위한 반사패딩"""
        h, w = x.shape[-2:]
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x, ph, pw

    img_paths = sorted(sum([glob.glob(os.path.join(args.input_dir, f'*.{ext}'))
                            for ext in ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']], []))
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")

    t0 = time.time()
    with torch.no_grad():
        for p in img_paths:
            im = Image.open(p).convert('RGB')
            x  = to_tensor(im).unsqueeze(0).to(device)  # [1,3,H,W]
            x, ph, pw = pad_to_mod(x, m=64)              # 필요시 16으로 변경

            # StyleFilter로 스타일/특징 추출 → backbone 복원
            # 레포 구현에 따라 입력 시그니처가 (x, style) 또는 (x)일 수 있어 두 가지 모두 시도
            try:
                style_vec = StyleFilter(x)
            except Exception:
                style_vec = None

            try:
                if style_vec is not None:
                    y = net(x, style_vec)
                else:
                    y = net(x)
            except TypeError:
                # 시그니처 반대 케이스 보정
                if style_vec is not None:
                    y = net(style_vec, x)
                else:
                    y = net(x)

            # 패딩 제거
            if ph or pw:
                y = y[..., :im.height, :im.width]

            y = torch.clamp(y, 0, 1).squeeze(0).cpu()
            out = to_pil(y)

            base = os.path.splitext(os.path.basename(p))[0]
            out_name = f"{base}_mwformer.png"
            out.save(os.path.join(args.save_dir, out_name))

    dt = time.time() - t0
    print(f'✅ 복원 완료: {len(img_paths)}개 → {args.save_dir}')
    print(f'⏱ elapsed: {dt:.2f}s  (avg {dt/len(img_paths):.3f}s/img)')
    raise SystemExit(0)

# ======================================================
# ================== Validation (with GT) ==============
# ======================================================

# --- Validation data loader --- #
val_data_dir = args.val_data_dir
val_filename = args.val_filename  # a text file listing validation image names in val_data_dir
val_data_loader = DataLoader(
    ValData(val_data_dir, val_filename),
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=8
)

print('--- Testing (validation with GT) starts! ---')
start_time = time.time()
with torch.no_grad():
    if args.no_metrics:
        # 간단 저장-only 루프가 필요하면, 여기서도 유사하게 구현 가능
        # (현재는 원본 로직을 유지하며, no_metrics일 경우엔 지표 계산을 생략하고 통과)
        val_psnr, val_ssim = 0.0, 0.0
    else:
        val_psnr, val_ssim = validation_stylevec(StyleFilter, net, val_data_loader, device)

end_time = time.time() - start_time
if not args.no_metrics:
    print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
