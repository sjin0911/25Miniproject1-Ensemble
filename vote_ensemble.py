# vote_ensemble.py
# 픽셀별 하드 보팅(모델1 vs 모델2) → GT에 더 가까운 픽셀 선택
# 사용 예:
#   python vote_ensemble.py \
#       --model1 /path/to/m1.png \
#       --model2 /path/to/m2.png \
#       --gt     /path/to/gt.png \
#       --outdir ./out_vote

import os, argparse
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def imread_u8_rgb(path):
    """[0,255] uint8 RGB로 로드 (OpenCV 기본 BGR이므로 변환)"""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def resize_like(src, ref):
    """src를 ref(H,W)에 맞춤 (inter_area: 축소/확대 모두 무난)"""
    h, w = ref.shape[:2]
    if src.shape[:2] == (h, w):
        return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

def compute_vote(m1, m2, gt, smooth_kernel=0):
    """
    m1, m2, gt : HxWx3 uint8 RGB
    반환: result(HxWx3 uint8), mask(HxW bool)  # mask=True는 m2가 선택된 픽셀
    """
    # 안전 계산을 위해 int16로 변환(음수 허용)
    m1i = m1.astype(np.int16)
    m2i = m2.astype(np.int16)
    gti = gt.astype(np.int16)

    # 채널 합 L1 거리 (픽셀마다 3채널 절대값 합)
    d1 = np.sum(np.abs(m1i - gti), axis=2)  # HxW
    d2 = np.sum(np.abs(m2i - gti), axis=2)  # HxW

    # m2가 더 가깝다면 True, 아니면 False(=m1)
    mask = d2 < d1

    # (선택) 마스크 스무딩: 작은 잡영역 완화
    if smooth_kernel and smooth_kernel >= 3:
        k = smooth_kernel if smooth_kernel % 2 == 1 else smooth_kernel + 1
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.medianBlur(mask_u8, ksize=k)
        mask = mask_u8 >= 128

    # 보팅 결과 합성
    result = np.where(mask[..., None], m2, m1).astype(np.uint8)
    return result, mask

def colorize_mask(mask):
    """
    이진 마스크를 시각화(RGB): m2 선택=초록, m1 선택=빨강
    """
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[~mask] = (255, 64, 64)   # m1 선택: R
    vis[ mask] = (64, 255, 64)   # m2 선택: G
    return vis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model1", required=True, help="모델1 결과 이미지 경로")
    ap.add_argument("--model2", required=True, help="모델2 결과 이미지 경로")
    ap.add_argument("--gt",     required=True, help="GT(정답) 이미지 경로")
    ap.add_argument("--outdir", required=True, help="출력 폴더")
    ap.add_argument("--smooth", type=int, default=0, help="마스크 medianBlur 커널(홀수, 0이면 미사용)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    m1 = imread_u8_rgb(args.model1)
    m2 = imread_u8_rgb(args.model2)
    gt = imread_u8_rgb(args.gt)

    # 해상도 통일: (권장) GT 크기에 맞춤
    m1 = resize_like(m1, gt)
    m2 = resize_like(m2, gt)

    result, mask = compute_vote(m1, m2, gt, smooth_kernel=args.smooth)

    # 품질 지표(선택): PSNR/SSIM
    psnr_res = psnr(gt, result, data_range=255)
    ssim_res = ssim(gt, result, data_range=255, channel_axis=2)

    # 저장 (RGB→BGR로 변환하여 OpenCV로 저장)
    res_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.outdir, "vote_result.png"), res_bgr)

    mask_vis = colorize_mask(mask)
    cv2.imwrite(os.path.join(args.outdir, "vote_mask_vis.png"),
                cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

    # 참고용: 개별 입력도 정렬 저장
    cv2.imwrite(os.path.join(args.outdir, "m1_aligned.png"),
                cv2.cvtColor(m1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "m2_aligned.png"),
                cv2.cvtColor(m2, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "gt_aligned.png"),
                cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))

    print(f"[보팅 완료] PSNR={psnr_res:.3f} dB, SSIM={ssim_res:.4f}")
    print(f" - 결과: {os.path.join(args.outdir, 'vote_result.png')}")
    print(f" - 선택맵 시각화: {os.path.join(args.outdir, 'vote_mask_vis.png')}")

if __name__ == "__main__":
    main()
