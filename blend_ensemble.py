# blend_ensemble_patch_batch_multi.py
# - 단일 파일: --model1 --model2 [--model3] [--model4] --gt
# - 배치:      --m1_dir --m2_dir [--m3_dir] [--m4_dir] --gt_dir
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2, os, argparse, random, glob, csv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ----------------------
# Utils
# ----------------------
def imread_u8_rgb(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def resize_like(src, ref):
    h, w = ref.shape[:2]
    return src if src.shape[:2]==(h,w) else cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

def to_tensor_u8(img_u8):
    return torch.from_numpy(img_u8.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)  # 1x3xHxW

def to_u8_rgb(t):
    if t.dim()==4: t = t[0]
    arr = t.permute(1,2,0).clamp(0,1).cpu().numpy()
    return (arr*255 + 0.5).astype(np.uint8)

# ----------------------
# 1x1 Soft Blending (N models)
# ----------------------
class Blend1x1N(nn.Module):
    """
    N개 모델 입력(각 3채널)을 받아 픽셀별 softmax 가중치(M개)를 예측하고
    가중합으로 출력 이미지를 생성.
    """
    def __init__(self, num_models: int):
        super().__init__()
        assert 2 <= num_models <= 4, "num_models must be 2~4"
        in_ch = 3 * num_models
        out_ch = num_models
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.num_models = num_models

    def forward(self, models):  # models: list of tensors [B,3,H,W]
        x = torch.cat(models, dim=1)                 # (B, 3M, H, W)
        logits = self.gate(x)                        # (B, M,  H, W)
        w = torch.softmax(logits, dim=1)             # 가중치 합=1
        # 가중합
        y = 0.0
        for i in range(self.num_models):
            y = y + w[:, i:i+1] * models[i]
        return y, w

# ----------------------
# Patch cropper (N개 모델 공통 크롭)
# ----------------------
def random_crop_multi(models, gt, crop_size):
    """
    models: list of tensors [1,3,H,W], gt: [1,3,H,W]
    return: (models_patched_list, gt_patch)
    """
    _,_,H,W = gt.shape
    if H < crop_size or W < crop_size:
        raise ValueError("crop_size too large")
    y = random.randint(0, H - crop_size)
    x = random.randint(0, W - crop_size)
    models_p = [m[:,:,y:y+crop_size, x:x+crop_size] for m in models]
    gt_p = gt[:,:,y:y+crop_size, x:x+crop_size]
    return models_p, gt_p

# ----------------------
# 파일 매칭 (키: 211015_031)
# ----------------------
def key_date_group(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parts = stem.split('_')
    if len(parts) < 3: return None
    date = parts[0].split('-')[-1]
    group = parts[2]
    return f"{date}_{group}"

def build_index(folder, exts=(".png",".jpg",".jpeg")):
    index = {}
    if not folder: return index
    for ext in exts:
        for p in glob.glob(os.path.join(folder, f"*{ext}")):
            k = key_date_group(p)
            if k: index.setdefault(k, []).append(p)
    return index

# ----------------------
# 1장 처리(학습→추론→저장)
# ----------------------
def process_one(model_paths, gt_path, outdir, epochs, patch_size, iters_per_epoch, lr, device):
    os.makedirs(outdir, exist_ok=True)

    # 로드 & 정렬
    gt_u8 = imread_u8_rgb(gt_path)
    models_u8 = [resize_like(imread_u8_rgb(p), gt_u8) for p in model_paths]
    gt = to_tensor_u8(gt_u8).to(device)
    models = [to_tensor_u8(img).to(device) for img in models_u8]

    # 모델/옵티마이저
    M = len(models)
    net = Blend1x1N(num_models=M).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    # 손실: L1
    def loss_fn(pred, tgt): return torch.mean(torch.abs(pred - tgt))

    # 학습
    for ep in range(1, epochs+1):
        net.train(); tot=0.0
        for _ in range(iters_per_epoch):
            ms, gtp = random_crop_multi(models, gt, patch_size)
            opt.zero_grad()
            y, _ = net(ms)
            loss = loss_fn(y, gtp)
            loss.backward(); opt.step()
            tot += loss.item()
        if ep % max(1, epochs//10) == 0 or ep == 1:
            print(f"[{os.path.basename(model_paths[0])}][Ep {ep}] loss={tot/iters_per_epoch:.6f}")

    # 추론(full)
    net.eval()
    with torch.no_grad():
        y_full, _ = net(models)
    out = to_u8_rgb(y_full)

    # 지표
    psnr_val = psnr(gt_u8, out, data_range=255)
    ssim_val = ssim(gt_u8, out, data_range=255, channel_axis=2)

    # 저장 (첫 번째 모델 이름 기준)
    stem = os.path.splitext(os.path.basename(model_paths[0]))[0]
    out_img = os.path.join(outdir, f"{stem}_blend_patch.png")
    cv2.imwrite(out_img, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    return psnr_val, ssim_val, out_img

# ----------------------
# 메인
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    # 단일 파일
    ap.add_argument("--model1"); ap.add_argument("--model2")
    ap.add_argument("--model3"); ap.add_argument("--model4")
    ap.add_argument("--gt")
    # 배치 폴더
    ap.add_argument("--m1_dir"); ap.add_argument("--m2_dir")
    ap.add_argument("--m3_dir"); ap.add_argument("--m4_dir")
    ap.add_argument("--gt_dir")
    # 공통
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--iters_per_epoch", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # ---- 단일 파일 모드 ----
    single_models = [p for p in [args.model1, args.model2, args.model3, args.model4] if p]
    if single_models and args.gt:
        if len(single_models) < 2:
            raise ValueError("단일 파일 모드에서는 최소 2개의 --model* 경로가 필요합니다.")
        psnr_val, ssim_val, out_img = process_one(
            single_models, args.gt, args.outdir,
            args.epochs, args.patch_size, args.iters_per_epoch, args.lr, device
        )
        print(f"[완료] PSNR={psnr_val:.3f} dB, SSIM={ssim_val:.4f}")
        print(f" - 결과: {out_img}")
        return

    # ---- 배치 모드 ----
    if not (args.m1_dir and args.m2_dir and args.gt_dir):
        raise ValueError("배치: --m1_dir --m2_dir --gt_dir(필수), 필요 시 --m3_dir --m4_dir 추가")

    # 폴더 인덱스
    idxs = []
    idx_m1 = build_index(args.m1_dir); idxs.append(idx_m1)
    idx_m2 = build_index(args.m2_dir); idxs.append(idx_m2)
    if args.m3_dir: idx_m3 = build_index(args.m3_dir); idxs.append(idx_m3)
    if args.m4_dir: idx_m4 = build_index(args.m4_dir); idxs.append(idx_m4)
    idx_gt = build_index(args.gt_dir)

    # 키 교집합
    keysets = [set(d.keys()) for d in idxs] + [set(idx_gt.keys())]
    keys = sorted(set.intersection(*keysets))
    print(f"[배치] 매칭된 키 개수: {len(keys)} (모델 {len(idxs)}개)")

    rows = []
    for k in keys:
        model_paths = [sorted(d[k])[0] for d in idxs]
        gt_path = sorted(idx_gt[k])[0]
        try:
            psnr_val, ssim_val, out_img = process_one(
                model_paths, gt_path, args.outdir,
                args.epochs, args.patch_size, args.iters_per_epoch, args.lr, device
            )
            print(f"[OK] {k} → PSNR={psnr_val:.3f}, SSIM={ssim_val:.4f}")
            rows.append([k] + model_paths + [gt_path, out_img, psnr_val, ssim_val])
        except Exception as e:
            print(f"[SKIP] {k}: {e}")

    if args.csv and rows:
        headers = ["key"] + [f"model{i+1}" for i in range(len(idxs))] + ["gt", "out", "psnr", "ssim"]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(headers); w.writerows(rows)
        print(f"[요약 저장] {args.csv}")

if __name__ == "__main__":
    main()
