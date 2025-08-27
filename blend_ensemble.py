# blend_ensemble_patch_batch.py
# - 단일 파일: 기존과 동일
# - 배치 처리: --m1_dir --m2_dir --gt_dir 사용 (여러 장 자동 매칭)
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
    if img_bgr is None:
        raise FileNotFoundError(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def resize_like(src, ref):
    h, w = ref.shape[:2]
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA) if src.shape[:2] != (h, w) else src

def to_tensor_u8(img_u8):
    return torch.from_numpy(img_u8.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)  # 1x3xHxW

def to_u8_rgb(t):
    if t.dim()==4: t=t[0]
    arr = t.permute(1,2,0).clamp(0,1).cpu().numpy()
    return (arr*255 + 0.5).astype(np.uint8)

# ----------------------
# 모델: 1x1 CNN 게이팅
# ----------------------
class Blend1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Conv2d(6,2,kernel_size=1,bias=True)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
    def forward(self,m1,m2):
        x = torch.cat([m1,m2],dim=1)      # (B,6,H,W)
        w = torch.softmax(self.gate(x),1) # (B,2,H,W)
        y = w[:,0:1]*m1 + w[:,1:2]*m2
        return y,w

# ----------------------
# Patch cropper
# ----------------------
def random_crop(m1,m2,gt,crop_size):
    _,_,H,W = gt.shape
    if H<crop_size or W<crop_size: raise ValueError("crop_size too large")
    y = random.randint(0,H-crop_size)
    x = random.randint(0,W-crop_size)
    return (m1[:,:,y:y+crop_size,x:x+crop_size],
            m2[:,:,y:y+crop_size,x:x+crop_size],
            gt[:,:,y:y+crop_size,x:x+crop_size])

# ----------------------
# 파일 매칭 유틸
# ----------------------
def key_date_group(path):
    # 예) D-211015_O9120F01_031_0033_mwformer.png -> "211015_031"
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parts = stem.split('_')
    if len(parts) < 3: return None
    date = parts[0].split('-')[-1]  # D-211015 -> 211015
    group = parts[2]                # 031
    return f"{date}_{group}"

def build_index(folder, exts=(".png",".jpg",".jpeg")):
    index = {}
    for ext in exts:
        for p in glob.glob(os.path.join(folder, f"*{ext}")):
            k = key_date_group(p)
            if k: index.setdefault(k, []).append(p)
    return index

# ----------------------
# 1장 처리(학습→추론→저장)
# ----------------------
def process_one(m1_path, m2_path, gt_path, outdir, epochs, patch_size, iters_per_epoch, lr, device):
    os.makedirs(outdir, exist_ok=True)

    # 로드 & 정렬
    m1_u8, m2_u8, gt_u8 = imread_u8_rgb(m1_path), imread_u8_rgb(m2_path), imread_u8_rgb(gt_path)
    m1_u8, m2_u8 = resize_like(m1_u8, gt_u8), resize_like(m2_u8, gt_u8)
    m1, m2, gt = to_tensor_u8(m1_u8).to(device), to_tensor_u8(m2_u8).to(device), to_tensor_u8(gt_u8).to(device)

    # 모델/옵티마이저
    model = Blend1x1().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # 손실: L1
    def loss_fn(pred,gt): return torch.mean(torch.abs(pred-gt))

    # 학습
    for ep in range(1, epochs+1):
        model.train(); total=0.0
        for _ in range(iters_per_epoch):
            m1p,m2p,gtp = random_crop(m1,m2,gt,patch_size)
            opt.zero_grad()
            y,_ = model(m1p,m2p)
            loss = loss_fn(y,gtp)
            loss.backward(); opt.step()
            total += loss.item()
        if ep%max(1,epochs//10)==0 or ep==1:
            print(f"[{os.path.basename(m1_path)}][Ep {ep}] loss={total/iters_per_epoch:.6f}")

    # 추론
    model.eval()
    with torch.no_grad():
        y_full,_ = model(m1,m2)
    out = to_u8_rgb(y_full)

    # 지표
    psnr_val = psnr(gt_u8, out, data_range=255)
    ssim_val = ssim(gt_u8, out, data_range=255, channel_axis=2)

    # 저장
    stem = os.path.splitext(os.path.basename(m1_path))[0]
    out_img  = os.path.join(outdir, f"{stem}_blend_patch.png")
    cv2.imwrite(out_img, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    return psnr_val, ssim_val, out_img

# ----------------------
# 메인
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    # 단일 파일
    ap.add_argument("--model1")
    ap.add_argument("--model2")
    ap.add_argument("--gt")
    # 배치 폴더
    ap.add_argument("--m1_dir")
    ap.add_argument("--m2_dir")
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

    # 단일 파일 모드
    if args.model1 and args.model2 and args.gt:
        psnr_val, ssim_val, out_img = process_one(
            args.model1, args.model2, args.gt, args.outdir,
            args.epochs, args.patch_size, args.iters_per_epoch, args.lr, device
        )
        print(f"[완료] PSNR={psnr_val:.3f} dB, SSIM={ssim_val:.4f}")
        print(f" - 결과: {out_img}")
        return

    # 배치 모드
    if not (args.m1_dir and args.m2_dir and args.gt_dir):
        raise ValueError("배치 처리는 --m1_dir --m2_dir --gt_dir 가 모두 필요합니다.")

    idx_m1 = build_index(args.m1_dir)
    idx_m2 = build_index(args.m2_dir)
    idx_gt = build_index(args.gt_dir)
    keys = sorted(set(idx_m1.keys()) & set(idx_m2.keys()) & set(idx_gt.keys()))
    print(f"[배치] 매칭된 키 개수: {len(keys)}")

    rows = []
    for k in keys:
        m1_path = sorted(idx_m1[k])[0]
        m2_path = sorted(idx_m2[k])[0]
        gt_path = sorted(idx_gt[k])[0]
        try:
            psnr_val, ssim_val, out_img = process_one(
                m1_path, m2_path, gt_path, args.outdir,
                args.epochs, args.patch_size, args.iters_per_epoch, args.lr, device
            )
            print(f"[OK] {k} → PSNR={psnr_val:.3f}, SSIM={ssim_val:.4f}")
            rows.append([k, m1_path, m2_path, gt_path, out_img, psnr_val, ssim_val])
        except Exception as e:
            print(f"[SKIP] {k}: {e}")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["key","model1","model2","gt","out","psnr","ssim"]); w.writerows(rows)
        print(f"[요약 저장] {args.csv}")

if __name__ == "__main__":
    main()
