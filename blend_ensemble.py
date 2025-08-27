# 1x1 CNN Soft Blending (Model1/Model2 -> Softmax weights -> Weighted Sum)
# 사용 예:
#   python blend_ensemble.py \
#       --model1 /path/to/m1.png \
#       --model2 /path/to/m2.png \
#       --gt     /path/to/gt.png \
#       --outdir ./out_blend \
#       --epochs 200 --lr 1e-2 --smooth 0

import os, argparse, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def imread_u8_rgb(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None: raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def resize_like(src, ref):
    h, w = ref.shape[:2]
    if src.shape[:2] == (h, w): return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

def to_tensor_u8_rgb(img_u8):
    # uint8 [0,255] HxWx3 -> float32 [0,1] 1x3xHxW
    x = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    return x

def to_u8_rgb(img_f):
    # 1x3xHxW or 3xHxW -> HxWx3 uint8
    if img_f.dim()==4: img_f = img_f[0]
    img = img_f.permute(1,2,0).clamp(0,1).detach().cpu().numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)

def colorize_weights(w):  # w: 1x2xHxW, sum=1
    # 가중치 시각화: (R=모델1, G=모델2)
    w = w.detach().cpu()[0]
    r = (w[0]*255.0).clamp(0,255).byte().numpy()
    g = (w[1]*255.0).clamp(0,255).byte().numpy()
    b = np.zeros_like(r)
    vis = np.stack([r,g,b], axis=-1)
    return vis

# 간단한 SSIM loss (채널별 평균 사용)
def ssim_loss(pred, gt, C1=0.01**2, C2=0.03**2):
    # pred, gt: 1x3xHxW in [0,1]
    mu_x = torch.mean(pred, dim=(2,3), keepdim=True)
    mu_y = torch.mean(gt,   dim=(2,3), keepdim=True)
    sigma_x = torch.var(pred, dim=(2,3), keepdim=True)
    sigma_y = torch.var(gt,   dim=(2,3), keepdim=True)
    sigma_xy = torch.mean((pred - mu_x) * (gt - mu_y), dim=(2,3), keepdim=True)

    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + 1e-8)  # 1x3x1x1
    # SSIM은 높을수록 좋으므로 loss는 1-평균
    return 1.0 - ssim_map.mean()

# =========================
# 1x1 Softmax Blender
# =========================
class Blend1x1(nn.Module):
    """
    입력: (B, 6, H, W) = [m1_rgb(3), m2_rgb(3)]
    1x1 conv -> (B, 2, H, W) 로짓 -> softmax -> 가중치
    최종 출력: sum_i w_i * x_i (i=2개의 모델), 채널별 동일 가중치 적용
    """
    def __init__(self):
        super().__init__()
        # 모델별 가중치 스칼라를 채널별 공유하려면, 먼저 1x1로 2채널 로짓을 만들고
        # 각 RGB 채널에 동일 가중치를 적용한다.
        self.gate = nn.Conv2d(6, 2, kernel_size=1, bias=True)
        # 초기화: 균등 혼합에 가깝게
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, m1, m2):
        x = torch.cat([m1, m2], dim=1)         # (B,6,H,W)
        logits = self.gate(x)                  # (B,2,H,W)
        weights = torch.softmax(logits, dim=1) # 합=1
        # 각 모델 RGB에 동일 가중치를 적용
        y = weights[:,0:1] * m1 + weights[:,1:1+1] * m2
        return y, weights

# =========================
# Train per-image (few epochs)
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", required=True)
    ap.add_argument("--model2", required=True)
    ap.add_argument("--gt",     required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr",     type=float, default=1e-2)
    ap.add_argument("--smooth", type=int, default=0, help="가중치 맵 medianBlur 커널(0이면 미사용)")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # --- Load
    m1_u8 = imread_u8_rgb(args.model1)
    m2_u8 = imread_u8_rgb(args.model2)
    gt_u8 = imread_u8_rgb(args.gt)

    # --- Align to GT size
    m1_u8 = resize_like(m1_u8, gt_u8)
    m2_u8 = resize_like(m2_u8, gt_u8)

    # --- To tensors [0,1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = to_tensor_u8_rgb(m1_u8).to(device)
    m2 = to_tensor_u8_rgb(m2_u8).to(device)
    gt = to_tensor_u8_rgb(gt_u8).to(device)

    # --- Model & Optim
    model = Blend1x1().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # --- Loss: L1 + SSIM (가벼운 조합)
    def loss_fn(pred, gt):
        l1 = torch.mean(torch.abs(pred - gt))
        s  = ssim_loss(pred, gt)
        return l1 + 0.2 * s, l1.item(), s.item()

    # --- Train few epochs (per-image fitting)
    best = {"loss": 1e9, "y": None, "w": None}
    for ep in range(1, args.epochs+1):
        model.train()
        opt.zero_grad()
        y, w = model(m1, m2)
        loss, l1_v, s_v = loss_fn(y, gt)
        loss.backward(); opt.step()

        # Early keep
        if loss.item() < best["loss"]:
            best = {"loss": loss.item(), "y": y.detach().clone(), "w": w.detach().clone()}

        if ep % max(1, args.epochs//10) == 0 or ep == args.epochs:
            print(f"[Ep {ep:4d}] loss={loss.item():.6f} (L1={l1_v:.6f}, SSIMloss={s_v:.6f})")

    # --- Best output
    y_best = best["y"]
    w_best = best["w"]
    out_u8 = to_u8_rgb(y_best)

    # --- (선택) weight map smoothing
    if args.smooth and args.smooth >= 3:
        k = args.smooth if args.smooth % 2 == 1 else args.smooth + 1
        w_map = w_best.detach().cpu().numpy()[0,1]  # 모델2 가중치 (HxW)
        w_u8  = (np.clip(w_map, 0, 1) * 255).astype(np.uint8)
        w_u8  = cv2.medianBlur(w_u8, k)
        w_s   = torch.from_numpy(w_u8.astype(np.float32)/255.0)[None,None].to(device)
        w_pair= torch.cat([1.0 - w_s, w_s], dim=1) # (1,2,H,W)
        # 스무딩 가중치로 다시 합성
        y_sm = w_pair[:,0:1] * m1 + w_pair[:,1:2] * m2
        out_u8 = to_u8_rgb(y_sm)

    # --- Metrics
    gt_aligned = gt_u8  # 이미 GT 기준 정렬
    psnr_val = psnr(gt_aligned, out_u8, data_range=255)
    ssim_val = ssim(gt_aligned, out_u8, data_range=255, channel_axis=2)

    # --- Save
    cv2.imwrite(os.path.join(args.outdir, "blend_result.png"),
                cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "m1_aligned.png"),
                cv2.cvtColor(m1_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "m2_aligned.png"),
                cv2.cvtColor(m2_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "gt_aligned.png"),
                cv2.cvtColor(gt_aligned, cv2.COLOR_RGB2BGR))

    w_vis = colorize_weights(w_best)  # R=m1, G=m2
    cv2.imwrite(os.path.join(args.outdir, "weights_vis.png"),
                cv2.cvtColor(w_vis, cv2.COLOR_RGB2BGR))

    print(f"[블렌딩 완료] PSNR={psnr_val:.3f} dB, SSIM={ssim_val:.4f}")
    print(f" - 결과 : {os.path.join(args.outdir, 'blend_result.png')}")
    print(f" - 가중치맵 시각화 : {os.path.join(args.outdir, 'weights_vis.png')}")

if __name__ == "__main__":
    main()
