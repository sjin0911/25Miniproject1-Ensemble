# vote_ensemble.py
# - 단일 파일: 기존과 동일
# - 배치 처리: --m1_dir --m2_dir --gt_dir 사용 (여러 장 자동 매칭)
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
    if src.shape[:2] == (h, w):
        return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

def compute_vote(m1, m2, gt, smooth_kernel=0):
    m1i = m1.astype(np.int16)
    m2i = m2.astype(np.int16)
    gti = gt.astype(np.int16)
    d1 = np.sum(np.abs(m1i - gti), axis=2)
    d2 = np.sum(np.abs(m2i - gti), axis=2)
    mask = d2 < d1
    if smooth_kernel and smooth_kernel >= 3:
        k = smooth_kernel if smooth_kernel % 2 == 1 else smooth_kernel + 1
        mask_u8 = (mask.astype(np.uint8) * 255)
        mask_u8 = cv2.medianBlur(mask_u8, ksize=k)
        mask = mask_u8 >= 128
    result = np.where(mask[..., None], m2, m1).astype(np.uint8)
    return result, mask

def colorize_mask(mask):
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[~mask] = (255, 64, 64)   # m1
    vis[ mask] = ( 64,255, 64)   # m2
    return vis

# ---- 매칭 키 함수들 ----
def key_date_group(name):
    # ex) D-211015_O9120F01_031_0033_mwformer.png -> "211015_031"
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    date = parts[0].split('-')[-1]  # "211015"
    group = parts[2]                # "031"
    return f"{date}_{group}"

def key_prefix3(name):
    # 처음 3개의 토큰을 키로 (충돌 가능성 ↑, 필요시 사용)
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    return '_'.join(parts[:3])

MATCHERS = {
    "date_group": key_date_group,
    "prefix3": key_prefix3,
}

def build_index(folder, exts=(".png",".jpg",".jpeg"), matcher="date_group"):
    fn_to_key = MATCHERS.get(matcher, key_date_group)
    index = {}
    for ext in exts:
        for p in glob.glob(os.path.join(folder, f"*{ext}")):
            k = fn_to_key(p)
            if k is None: 
                continue
            index.setdefault(k, []).append(p)
    return index

def process_one(m1_path, m2_path, gt_path, outdir, smooth):
    m1 = imread_u8_rgb(m1_path)
    m2 = imread_u8_rgb(m2_path)
    gt = imread_u8_rgb(gt_path)
    m1 = resize_like(m1, gt)
    m2 = resize_like(m2, gt)
    result, mask = compute_vote(m1, m2, gt, smooth_kernel=smooth)
    psnr_res = psnr(gt, result, data_range=255)
    ssim_res = ssim(gt, result, data_range=255, channel_axis=2)

    # 파일명 구성: 모델1 파일명(확장자 제거) + "_vote.png"
    stem = os.path.splitext(os.path.basename(m1_path))[0]
    out_img = os.path.join(outdir, f"{stem}_vote.png")
    out_mask = os.path.join(outdir, f"{stem}_vote_mask.png")
    out_m1 = os.path.join(outdir, f"{stem}_m1_align.png")
    out_m2 = os.path.join(outdir, f"{stem}_m2_align.png")
    out_gt = os.path.join(outdir, f"{stem}_gt_align.png")

    cv2.imwrite(out_img, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_mask, cv2.cvtColor(colorize_mask(mask), cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_m1, cv2.cvtColor(m1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_m2, cv2.cvtColor(m2, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_gt, cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
    return psnr_res, ssim_res, out_img

def main():
    ap = argparse.ArgumentParser()
    # 단일 파일 모드
    ap.add_argument("--model1", help="모델1 결과 이미지 경로")
    ap.add_argument("--model2", help="모델2 결과 이미지 경로")
    ap.add_argument("--gt",     help="GT(정답) 이미지 경로")
    # 배치 모드
    ap.add_argument("--m1_dir", help="모델1 결과 폴더")
    ap.add_argument("--m2_dir", help="모델2 결과 폴더")
    ap.add_argument("--gt_dir", help="GT 폴더")
    ap.add_argument("--match_mode", default="date_group", choices=list(MATCHERS.keys()),
                    help="파일명 매칭 규칙(기본: date_group)")
    ap.add_argument("--outdir", required=True, help="출력 폴더(단일/배치 공통)")
    ap.add_argument("--smooth", type=int, default=0, help="마스크 medianBlur 커널(홀수, 0이면 미사용)")
    ap.add_argument("--csv", default="", help="배치 결과 요약 CSV 경로(선택)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- 단일 파일 모드 ----
    if args.model1 and args.model2 and args.gt:
        psnr_res, ssim_res, out_img = process_one(args.model1, args.model2, args.gt, args.outdir, args.smooth)
        print(f"[보팅 완료] PSNR={psnr_res:.3f} dB, SSIM={ssim_res:.4f}")
        print(f" - 결과: {out_img}")
        return

    # ---- 배치 모드 ----
    if not (args.m1_dir and args.m2_dir and args.gt_dir):
        raise ValueError("배치 처리 시 --m1_dir --m2_dir --gt_dir 를 모두 지정하세요.")

    idx_m1 = build_index(args.m1_dir, matcher=args.match_mode)
    idx_m2 = build_index(args.m2_dir, matcher=args.match_mode)
    idx_gt = build_index(args.gt_dir, matcher=args.match_mode)

    keys = sorted(set(idx_m1.keys()) & set(idx_m2.keys()) & set(idx_gt.keys()))
    print(f"[배치] 매칭된 키 개수: {len(keys)} (mode={args.match_mode})")

    rows = []
    for k in keys:
        # 각 키에 해당하는 파일이 여러 개인 경우, 사전순 첫 번째를 사용
        m1_path = sorted(idx_m1[k])[0]
        m2_path = sorted(idx_m2[k])[0]
        gt_path = sorted(idx_gt[k])[0]
        try:
            psnr_res, ssim_res, out_img = process_one(m1_path, m2_path, gt_path, args.outdir, args.smooth)
            print(f"[OK] {k} → PSNR={psnr_res:.3f}, SSIM={ssim_res:.4f}")
            rows.append([k, m1_path, m2_path, gt_path, out_img, psnr_res, ssim_res])
        except Exception as e:
            print(f"[SKIP] {k}: {e}")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key","model1","model2","gt","out","psnr","ssim"])
            w.writerows(rows)
        print(f"[요약 저장] {args.csv}")

if __name__ == "__main__":
    main()
