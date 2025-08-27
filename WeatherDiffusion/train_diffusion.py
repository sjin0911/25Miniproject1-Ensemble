# train_diffusion.py  (교체본)
import argparse, os, yaml, torch, numpy as np
import torch.backends.cudnn as cudnn
import datasets_finetune as datasets
import models_finetune as models
from models_finetune import DenoisingDiffusion, DiffusiveRestoration

def dict2namespace(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns

def parse_args_and_config():
    p = argparse.ArgumentParser(description="Training Patch-Based Denoising Diffusion Models")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--sampling_timesteps", type=int, default=25)
    p.add_argument("--image_folder", type=str, default="results/images/")
    p.add_argument("--seed", type=int, default=61)
    args = p.parse_args()

    # ✅ 파일이 그대로 주어지면 그 경로 사용, 아니면 후보 폴더에서 탐색
    cand = args.config
    search_roots = ["", "configs", "configs_finetune"]
    if os.path.isabs(cand) or os.path.isfile(cand):
        cfg_path = cand
    else:
        cfg_path = None
        for root in search_roots:
            path = os.path.join(root, cand) if root else cand
            if os.path.isfile(path):
                cfg_path = path
                break
    if cfg_path is None:
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return args, dict2namespace(cfg)

def main():
    args, config = parse_args_and_config()

    # device & seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")
    config.device = device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Colab/로더 안전값 (로더 만들기 전에)
    if hasattr(config, "data"):
        config.data.num_workers = 0

    # dataset 인스턴스 (로더 생성은 models 쪽에서 get_loaders() 호출)
    print(f"=> using dataset '{config.data.dataset}'")
    DATASET = datasets.__dict__[config.data.dataset](config)  # 우리 finetune AllWeather로 매핑되어야 함

    # diffusion 모델 (finetune 버전이 __init__에서 로드/EMA/검증까지 처리)
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)

    # 학습 시작
    diffusion.train(DATASET)

if __name__ == "__main__":
    main()
