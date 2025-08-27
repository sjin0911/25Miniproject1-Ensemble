import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2, os, argparse, random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ----------------------
# Utils
# ----------------------
def imread_u8_rgb(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
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
    return (arr*255).astype(np.uint8)

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
    return m1[:,:,y:y+crop_size,x:x+crop_size], \
           m2[:,:,y:y+crop_size,x:x+crop_size], \
           gt[:,:,y:y+crop_size,x:x+crop_size]

# ----------------------
# 학습
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", required=True)
    ap.add_argument("--model2", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--iters_per_epoch", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 로드
    m1_u8, m2_u8, gt_u8 = imread_u8_rgb(args.model1), imread_u8_rgb(args.model2), imread_u8_rgb(args.gt)
    m1_u8, m2_u8 = resize_like(m1_u8,gt_u8), resize_like(m2_u8,gt_u8)
    m1, m2, gt = to_tensor_u8(m1_u8).to(device), to_tensor_u8(m2_u8).to(device), to_tensor_u8(gt_u8).to(device)

    model = Blend1x1().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # 손실: L1
    def loss_fn(pred,gt): return torch.mean(torch.abs(pred-gt))

    # 학습 루프
    for ep in range(1,args.epochs+1):
        model.train(); total=0
        for it in range(args.iters_per_epoch):
            m1p,m2p,gtp = random_crop(m1,m2,gt,args.patch_size)
            opt.zero_grad()
            y,w = model(m1p,m2p)
            loss = loss_fn(y,gtp)
            loss.backward(); opt.step()
            total += loss.item()
        if ep%50==0 or ep==1:
            print(f"[Ep {ep}] loss={total/args.iters_per_epoch:.6f}")

    # 최종 결과 full image 예측
    model.eval()
    with torch.no_grad():
        y_full,_ = model(m1,m2)
    out = to_u8_rgb(y_full)
    psnr_val = psnr(gt_u8,out,data_range=255)
    ssim_val = ssim(gt_u8,out,data_range=255,channel_axis=2)
    cv2.imwrite(os.path.join(args.outdir,"blend_patch_result.png"),
                cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
    print(f"[완료] PSNR={psnr_val:.3f} dB, SSIM={ssim_val:.4f}")

if __name__=="__main__":
    main()
