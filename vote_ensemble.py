# vote_ensemble_multi.py
# - 단일 파일: --model1 --model2 [--model3] [--model4]
# - 배치 처리: --m1_dir --m2_dir [--m3_dir] [--m4_dir]
import os, argparse, glob, csv
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def imread_u8_rgb(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def resize_like(src, ref):
    h, w = ref.shape[:2]
    return src if src.shape[:2] == (h, w) else cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

# --------- 핵심: N-모델 보팅 ----------
def compute_vote_multi(models, gt, smooth_kernel=0):
    """
    models: list of HxWx3 uint8 RGB (길이 M, 2<=M<=4)
    gt:     HxWx3 uint8 RGB
    return: result(HxWx3 u8), winner(HxW int in [0,M-1])
    """
    M = len(models)
    gti = gt.astype(np.int16)
    # (H,W,M): 각 모델과 GT의 채널합 L1 거리
    dists = []
    for m in models:
        d = np.sum(np.abs(m.astype(np.int16) - gti), axis=2)
        dists.append(d)
    D = np.stack(dists, axis=2)           # HxWxM
    winner = np.argmin(D, axis=2).astype(np.uint8)  # HxW

    # (선택) 스무딩: winner 맵에 medianBlur 적용(모드 필터 비슷한 효과)
    if smooth_kernel and smooth_kernel >= 3:
        k = smooth_kernel if smooth_kernel % 2 == 1 else smooth_kernel + 1
        winner = cv2.medianBlur(winner, ksize=k)

    # 우승 모델 픽셀 채택
    H, W = winner.shape
    result = np.zeros_like(gt)
    for idx, m in enumerate(models):
        mask = (winner == idx)[..., None]
        result = np.where(mask, m, result)
    return result.astype(np.uint8), winner

def colorize_winner(winner):
    """
    winner: HxW uint8 (0..3)
    색상: 0=Red(m1), 1=Green(m2), 2=Blue(m3), 3=Yellow(m4)
    """
    H, W = winner.shape
    vis = np.zeros((H,W,3), dtype=np.uint8)
    vis[winner==0] = (255,  64,  64)   # m1
    vis[winner==1] = ( 64, 255,  64)   # m2
    vis[winner==2] = ( 64,  64, 255)   # m3
    vis[winner==3] = (255, 255,  64)   # m4
    return vis

# ---- 파일명 키 추출(예: 211015_031) ----
def key_date_group(name):
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    parts = stem.split('_')
    if len(parts) < 3: return None
    date = parts[0].split('-')[-1]  # "D-211015" -> "211015"
    group = parts[2]                # "031"
    return f"{date}_{group}"

def build_index(folder, exts=(".png",".jpg",".jpeg")):
    index = {}
    if not folder: return index
    for ext in exts:
        for p in glob.glob(os.path.join(folder, f"*{ext}")):
            k = key_date_group(p)
            if k: index.setdefault(k, []).append(p)
    return index

def process_one(model_paths, gt_path, outdir, smooth):
    """
    model_paths: list[str] (len>=2)
    """
    gt = imread_u8_rgb(gt_path)
    models = [resize_like(imread_u8_rgb(p), gt) for p in model_paths]
    result, winner = compute_vote_multi(models, gt, smooth_kernel=smooth)

    psnr_res = psnr(gt, result, data_range=255)
    ssim_res = ssim(gt, result, data_range=255, channel_axis=2)

    # 저장명: model1 파일명을 기준(stem)으로
    stem = os.path.splitext(os.path.basename(model_paths[0]))[0]
    out_img  = os.path.join(outdir, f"{stem}_vote.png")
    out_mask = os.path.join(outdir, f"{stem}_vote_mask.png")
    cv2.imwrite(out_img,  cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_mask, cv2.cvtColor(colorize_winner(winner), cv2.COLOR_RGB2BGR))

    # (선택) 정렬본 저장
    for i, m in enumerate(models, 1):
        cv2.imwrite(os.path.join(outdir, f"{stem}_m{i}_align.png"), cv2.cvtColor(m, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(outdir, f"{stem}_gt_align.png"), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))

    return psnr_res, ssim_res, out_img

def main():
    ap = argparse.ArgumentParser()
    # 단일 파일
    ap.add_argument("--model1")
    ap.add_argument("--model2")
    ap.add_argument("--model3")
    ap.add_argument("--model4")
    ap.add_argument("--gt")
    # 배치 폴더
    ap.add_argument("--m1_dir")
    ap.add_argument("--m2_dir")
    ap.add_argument("--m3_dir")
    ap.add_argument("--m4_dir")
    ap.add_argument("--gt_dir")
    # 공통
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -------- 단일 파일 모드 --------
    single_models = [p for p in [args.model1, args.model2, args.model3, args.model4] if p]
    if single_models and args.gt:
        if len(single_models) < 2:
            raise ValueError("단일 파일 모드에서는 최소 2개의 --model* 경로가 필요합니다.")
        psnr_res, ssim_res, out_img = process_one(single_models, args.gt, args.outdir, args.smooth)
        print(f"[보팅 완료] PSNR={psnr_res:.3f} dB, SSIM={ssim_res:.4f}")
        print(f" - 결과: {out_img}")
        return

    # -------- 배치 모드 --------
    if not (args.m1_dir and args.m2_dir and args.gt_dir):
        raise ValueError("배치 모드: --m1_dir --m2_dir --gt_dir(필수), 필요 시 --m3_dir --m4_dir 추가")

    idxs = []
    idx_m1 = build_index(args.m1_dir); idxs.append(idx_m1)
    idx_m2 = build_index(args.m2_dir); idxs.append(idx_m2)
    if args.m3_dir: idx_m3 = build_index(args.m3_dir); idxs.append(idx_m3)
    if args.m4_dir: idx_m4 = build_index(args.m4_dir); idxs.append(idx_m4)
    idx_gt = build_index(args.gt_dir)

    # 교집합 키
    keysets = [set(d.keys()) for d in idxs] + [set(idx_gt.keys())]
    keys = sorted(set.intersection(*keysets))
    print(f"[배치] 매칭된 키 개수: {len(keys)} (모델 {len(idxs)}개)")

    rows = []
    for k in keys:
        model_paths = [sorted(d[k])[0] for d in idxs]
        gt_path = sorted(idx_gt[k])[0]
        try:
            psnr_res, ssim_res, out_img = process_one(model_paths, gt_path, args.outdir, args.smooth)
            print(f"[OK] {k} → PSNR={psnr_res:.3f}, SSIM={ssim_res:.4f}")
            row = [k] + model_paths + [gt_path, out_img, psnr_res, ssim_res]
            rows.append(row)
        except Exception as e:
            print(f"[SKIP] {k}: {e}")

    if args.csv and rows:
        # CSV 헤더는 모델 개수에 따라 가변
        headers = ["key"] + [f"model{i+1}" for i in range(len(idxs))] + ["gt", "out", "psnr", "ssim"]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(headers); w.writerows(rows)
        print(f"[요약 저장] {args.csv}")

if __name__ == "__main__":
    main()
