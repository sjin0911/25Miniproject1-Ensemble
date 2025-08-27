import os, argparse, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ----------------------
# Utils
# ----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def imread_u8_rgb(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def resize_like(src, ref):
    h, w = ref.shape[:2]
    if src.shape[:2] == (h, w): return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

def to_tensor_u8(img_u8):
    # uint8 [0,255] HxWx3 -> float32 [0,1] 1x3xHxW
    return torch.from_numpy(img_u8.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)

def to_u8_rgb(t):
    if t.dim()==4: t=t[0]
    arr = t.permute(1,2,0).clamp(0,1).cpu().numpy()
    return (arr*255.0 + 0.5).astype(np.uint8)

# ----------------------
# 1x1 Blender (with Temperature)
# ----------------------
class Blend1x1(nn.Module):
    """
    입력: (B,6,H,W) = [m1_rgb(3), m2_rgb(3)]
    1x1 conv -> (B,2,H,W) logits; softmax(logits/T)
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Conv2d(6,2,kernel_size=1,bias=True)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, m1, m2, temperature=1.0):
        x = torch.cat([m1, m2], dim=1)         # (B,6,H,W)
        logits = self.gate(x)                  # (B,2,H,W)
        weights = torch.softmax(logits / max(1e-6, float(temperature)), dim=1)
        y = weights[:,0:1]*m1 + weights[:,1:2]*m2
        return y, weights

# ----------------------
# Random Patch Sampler
# ----------------------
def random_crop(m1,m2,gt,crop):
    _,_,H,W = gt.shape
    if H<crop or W<crop: raise ValueError("patch_size too large for this image")
    y = random.randint(0, H - crop)
    x = random.randint(0, W - crop)
    return (m1[:,:,y:y+crop,x:x+crop],
            m2[:,:,y:y+crop,x:x+crop],
            gt[:,:,y:y+crop,x:x+crop])

# ----------------------
# Losses
# ----------------------
# Lightweight SSIM (global stats) – 충분히 효과적이며 빠름
def ssim_loss(pred, gt, C1=0.01**2, C2=0.03**2):
    mu_x = torch.mean(pred, dim=(2,3), keepdim=True)
    mu_y = torch.mean(gt,   dim=(2,3), keepdim=True)
    sigma_x = torch.var(pred, dim=(2,3), keepdim=True)
    sigma_y = torch.var(gt,   dim=(2,3), keepdim=True)
    sigma_xy = torch.mean((pred - mu_x) * (gt - mu_y), dim=(2,3), keepdim=True)
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + 1e-8)  # 1x3x1x1
    return 1.0 - ssim_map.mean()

class VGGPerceptual(nn.Module):
    """
    VGG16 features 기반 Perceptual Loss.
    입력은 [0,1] → ImageNet 정규화 후 conv1_2, conv2_2, conv3_3 feature L1 합산.
    """
    def __init__(self, layers=('relu1_2','relu2_2','relu3_3'), weight=(1.0, 0.5, 0.25)):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.layers = layers
        self.weight = weight
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def encode(self, x):
        # x: [0,1]
        x = (x - self.mean) / self.std
        feats = {}
        idx_map = {'relu1_2':3, 'relu2_2':8, 'relu3_3':15}  # VGG16 feature indices
        cur = x
        for i,layer in enumerate(self.vgg):
            cur = layer(cur)
            # capture after ReLU layers of interest
            for k,idx in idx_map.items():
                if i == idx:
                    feats[k] = cur
        return feats

    def forward(self, x, y):
        fx = self.encode(x); fy = self.encode(y)
        loss = 0.0
        for i,layer in enumerate(self.layers):
            loss = loss + self.weight[i]*torch.mean(torch.abs(fx[layer]-fy[layer]))
        return loss

def build_lpips(device):
    try:
        import lpips
        net = lpips.LPIPS(net='vgg').to(device).eval()
        for p in net.parameters(): p.requires_grad=False
        return net
    except Exception as e:
        print("[경고] lpips 임포트 실패 → LPIPS 미사용:", e)
        return None

# ----------------------
# Train
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    # paths
    ap.add_argument("--model1", required=True)
    ap.add_argument("--model2", required=True)
    ap.add_argument("--gt",     required=True)
    ap.add_argument("--outdir", required=True)
    # training
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--iters_per_epoch", type=int, default=80)
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    # losses
    ap.add_argument("--w_l1", type=float, default=1.0)
    ap.add_argument("--w_ssim", type=float, default=0.5)
    ap.add_argument("--w_vgg", type=float, default=0.1)
    ap.add_argument("--w_lpips", type=float, default=0.0, help=">0이면 LPIPS 추가 (pip install lpips 필요)")
    # softmax temperature
    ap.add_argument("--temperature", type=float, default=0.7, help="<1 → 샤프, >1 → 스무스")
    # smoothing (optional vis)
    ap.add_argument("--smooth_weights", type=int, default=0, help="가중치맵 medianBlur 커널(홀수). 0=미사용")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load & align
    m1_u8 = imread_u8_rgb(args.model1)
    m2_u8 = imread_u8_rgb(args.model2)
    gt_u8 = imread_u8_rgb(args.gt)
    m1_u8 = resize_like(m1_u8, gt_u8)
    m2_u8 = resize_like(m2_u8, gt_u8)

    m1 = to_tensor_u8(m1_u8).to(device)
    m2 = to_tensor_u8(m2_u8).to(device)
    gt = to_tensor_u8(gt_u8).to(device)

    # model & opt
    model = Blend1x1().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # losses
    perc = VGGPerceptual().to(device)
    lpips_net = build_lpips(device) if args.w_lpips > 0 else None

    def loss_fn(pred, tgt):
        l_total = 0.0
        l1 = torch.mean(torch.abs(pred - tgt))
        l_total += args.w_l1 * l1

        if args.w_ssim > 0:
            ls = ssim_loss(pred, tgt)
            l_total += args.w_ssim * ls
        else:
            ls = torch.tensor(0.0, device=pred.device)

        if args.w_vgg > 0:
            lv = perc(pred, tgt)
            l_total += args.w_vgg * lv
        else:
            lv = torch.tensor(0.0, device=pred.device)

        if lpips_net is not None and args.w_lpips > 0:
            # LPIPS expects [-1,1]; convert from [0,1]
            p = pred*2.0 - 1.0
            t = tgt*2.0 - 1.0
            ll = lpips_net(p, t).mean()
            l_total += args.w_lpips * ll
        else:
            ll = torch.tensor(0.0, device=pred.device)

        return l_total, (l1.item(), ls.item(), lv.item(), ll.item())

    # train with random patches
    for ep in range(1, args.epochs+1):
        model.train()
        tot = 0.0; l1m=lsm=lvm=llm=0.0
        for _ in range(args.iters_per_epoch):
            m1p, m2p, gtp = random_crop(m1, m2, gt, args.patch_size)
            opt.zero_grad()
            y, w = model(m1p, m2p, temperature=args.temperature)
            loss, parts = loss_fn(y, gtp)
            loss.backward(); opt.step()
            tot += loss.item(); l1m += parts[0]; lsm += parts[1]; lvm += parts[2]; llm += parts[3]
        if ep % max(1,args.epochs//10)==0 or ep==1:
            n = float(args.iters_per_epoch)
            print(f"[Ep {ep}] total={tot/n:.6f} | L1={l1m/n:.6f} SSIM={lsm/n:.6f} VGG={lvm/n:.6f} LPIPS={llm/n:.6f} | T={args.temperature}")

    # inference on full image
    model.eval()
    with torch.no_grad():
        y_full, w_full = model(m1, m2, temperature=args.temperature)
    out_u8 = to_u8_rgb(y_full)

    # optional: smooth weight map for a second-pass recomposition
    if args.smooth_weights and args.smooth_weights >= 3:
        k = args.smooth_weights if args.smooth_weights % 2 == 1 else args.smooth_weights + 1
        w2 = w_full[0,1].clamp(0,1).cpu().numpy()
        w2_u8 = (w2*255).astype(np.uint8)
        w2_u8 = cv2.medianBlur(w2_u8, k)
        w2_s  = torch.from_numpy(w2_u8.astype(np.float32)/255.).to(device)[None,None]
        w_pair= torch.cat([1.0 - w2_s, w2_s], dim=1)
        with torch.no_grad():
            y_sm = w_pair[:,0:1]*m1 + w_pair[:,1:2]*m2
        out_u8 = to_u8_rgb(y_sm)

    # metrics & save
    psnr_val = psnr(gt_u8, out_u8, data_range=255)
    ssim_val = ssim(gt_u8, out_u8, data_range=255, channel_axis=2)

    cv2.imwrite(os.path.join(args.outdir, "blend_patch_result.png"),
                cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "m1_aligned.png"),
                cv2.cvtColor(m1_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "m2_aligned.png"),
                cv2.cvtColor(m2_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "gt_aligned.png"),
                cv2.cvtColor(gt_u8, cv2.COLOR_RGB2BGR))

    # weight visualization (R=m1, G=m2)
    w = w_full[0].detach().cpu()
    r = (w[0]*255).clamp(0,255).byte().numpy()
    g = (w[1]*255).clamp(0,255).byte().numpy()
    vis = np.stack([r,g,np.zeros_like(r)], axis=-1)
    cv2.imwrite(os.path.join(args.outdir, "weights_vis.png"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"[블렌딩 패치 완료] PSNR={psnr_val:.3f} dB, SSIM={ssim_val:.4f}")
    print(f" - 결과 : {os.path.join(args.outdir, 'blend_patch_result.png')}")
    print(f" - 가중치맵 : {os.path.join(args.outdir, 'weights_vis.png')}")
    print(f" - T(temperature)={args.temperature}, patch={args.patch_size}")

if __name__=="__main__":
    main()
