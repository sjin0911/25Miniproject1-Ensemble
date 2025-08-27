# models_finetune/ddm_finetune.py
import os, time, shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torch.serialization import safe_globals

import utils
from utils.optimize import get_optimizer
from models.unet import DiffusionUNet


# ===== helpers =====
def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def _unwrap_state_dict(obj):
    if hasattr(obj, "module"):
        obj = obj.module
    return obj.state_dict() if hasattr(obj, "state_dict") else obj

def _save_ckpt(save_dir, tag, model, ema_model, optimizer, scheduler,
               epoch, step, config, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "step": int(step),
        "model": _unwrap_state_dict(model),
        "ema": _unwrap_state_dict(ema_model) if ema_model is not None else None,
        "optim": optimizer.state_dict() if optimizer is not None else None,
        "sched": scheduler.state_dict() if scheduler is not None else None,
        "config": config,  # PyTorch 2.6에서 로드 시 safe_globals 허용 필요
    }
    path = os.path.join(save_dir, f"{tag}.pth.tar")
    torch.save(ckpt, path)
    shutil.copy2(path, os.path.join(save_dir, "latest.pth.tar"))
    if is_best:
        shutil.copy2(path, os.path.join(save_dir, "best_ssim.pth.tar"))
    print(f"[ckpt] saved: {path}  (latest{' & best' if is_best else ''})")


# ===== beta schedule =====
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x): return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# ===== loss =====
def noise_estimation_loss(model, x0, t, e, b):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    out = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float())
    return (e - out).square().sum(dim=(1, 2, 3)).mean(dim=0)


# ===== EMA =====
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}
    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()
    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[name].data = (1. - self.mu) * p.data + self.mu * self.shadow[name].data
    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, p in module.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name].data)
    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner = module.module
            copy = type(inner)(inner.config).to(inner.config.device)
            copy.load_state_dict(inner.state_dict())
            copy = nn.DataParallel(copy)
        else:
            copy = type(module)(module.config).to(module.config.device)
            copy.load_state_dict(module.state_dict())
        self.ema(copy)
        return copy
    def state_dict(self): return self.shadow
    def load_state_dict(self, sd): self.shadow = sd


# ===== state_dict flexible loader (prefix auto-fix) =====
def _flex_load_state_dict(model, sd, strict=True):
    """
    state_dict 키와 모델(DataParallel 여부)의 불일치를 자동 보정해서 로드.
    - ckpt 키에 'module.'가 없는데 모델이 DP인 경우 → model.module에 로드
    - ckpt 키에 'module.'가 있는데 모델이 non-DP인 경우 → prefix 제거 후 로드
    - 그 외는 가능한 순서대로 시도
    """
    def has_module_prefix(d):
        for k in d.keys():
            return k.startswith("module.")
        return False

    # 그대로 먼저 시도
    try:
        model.load_state_dict(sd, strict=strict)
        return
    except RuntimeError:
        pass

    # DP면 module에 직접 시도
    if isinstance(model, nn.DataParallel):
        try:
            model.module.load_state_dict(sd, strict=strict)
            return
        except RuntimeError:
            pass

    # prefix 보정
    sd_has_module = has_module_prefix(sd)
    if sd_has_module and not isinstance(model, nn.DataParallel):
        # ckpt: module.*, model: non-DP → prefix 제거
        sd_fixed = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd_fixed, strict=strict)
        return
    if (not sd_has_module) and isinstance(model, nn.DataParallel):
        # ckpt: no module., model: DP → module에 로드
        model.module.load_state_dict(sd, strict=strict)
        return

    # 마지막 안전장치
    model.load_state_dict(sd, strict=False)


# ===== main class =====
class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config).to(self.device)
        self.model = nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        # AdamW 등 utils.optimize 설정 사용
        self.optimizer = get_optimizer(self.config, self.model.parameters())
        self.scheduler = None  # 필요 시 붙이기

        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        # PyTorch 2.6: 안전 로더 허용(ckpt에 argparse.Namespace 포함)
        try:
            ckpt = utils.logging.load_checkpoint(load_path, None)
        except Exception:
            with safe_globals([argparse.Namespace]):
                ckpt = torch.load(load_path, map_location="cpu")

        # 다양한 포맷 호환
        state_dict = ckpt.get("ema") or ckpt.get("model") or ckpt.get("state_dict")
        optim_sd   = ckpt.get("optim") or ckpt.get("optimizer")
        ema_sd     = ckpt.get("ema")   or ckpt.get("ema_helper")

        self.start_epoch = int(ckpt.get("epoch", 0))
        self.step        = int(ckpt.get("step", 0))

        # ✅ 접두사 자동 보정 로드
        if state_dict is not None:
            _flex_load_state_dict(self.model, state_dict, strict=True)

        if optim_sd is not None and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(optim_sd)
            except Exception as e:
                print(f"[warn] optimizer load skipped: {e}")

        if ema_sd is not None:
            try:
                self.ema_helper.load_state_dict(ema_sd)
            except Exception as e:
                print(f"[warn] ema_helper load skipped: {e}")

        if ema:
            self.ema_helper.ema(self.model)

        print(f"=> loaded checkpoint '{load_path}' (epoch {self.start_epoch}, step {self.step})")

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders(parse_patches=True)

        # 저장 폴더 결정
        save_dir = getattr(self.args, "save_dir", None)
        if save_dir is None:
            ds_name = getattr(self.config.data, "dataset", "allweather").lower()
            save_dir = os.path.join("results", "finetune", ds_name)
        os.makedirs(save_dir, exist_ok=True)

        # resume(선택)
        if isinstance(self.args.resume, str) and os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        snap_freq = int(getattr(self.config.training, "snapshot_freq", 1))
        val_freq  = int(getattr(self.config.training, "validation_freq", 1))
        best_ssim = -1.0

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0.0

            self.model.train()
            for i, (x, y) in enumerate(train_loader):
                # (B, N, C, H, W) → (B*N, C, H, W)
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)

                data_time += time.time() - data_start
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,), device=self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = noise_estimation_loss(self.model, x, t, e, b)

                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item():.6f}, data time: {data_time / (i+1):.4f}s")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    try: self.scheduler.step()
                    except: pass
                self.ema_helper.update(self.model)
                data_start = time.time()

                # 스텝 단위 검증/샘플(선택)
                if self.step % val_freq == 0:
                    self.model.eval()
                    try:
                        self.sample_validation_patches(val_loader, self.step)
                    except Exception as e:
                        print(f"[warn] validation sampling skipped: {e}")
                    self.model.train()

            # (옵션) 에폭 단위 validate → best 갱신
            val_ssim = None
            if hasattr(self, "validate"):
                try:
                    log = self.validate(val_loader)  # {'ssim': ...} 형태 기대
                    if isinstance(log, dict) and "ssim" in log:
                        val_ssim = float(log["ssim"])
                        if val_ssim > best_ssim:
                            best_ssim = val_ssim
                except Exception as e:
                    print(f"[warn] validate() skipped: {e}")

            # 에폭 스냅샷 저장
            tag = f"epoch_{epoch + 1:03d}"
            is_best = (val_ssim is not None and val_ssim >= best_ssim)
            _save_ckpt(
                save_dir=save_dir,
                tag=tag,
                model=self.model,
                ema_model=self.ema_helper.ema_copy(self.model) if hasattr(self, "ema_helper") else None,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                step=self.step,
                config=self.config,
                is_best=is_best
            )

    # ===== sampling =====
    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs, _ = utils.sampling.generalized_steps_overlapping(
                x, x_cond, seq, self.model, self.betas, eta=0., corners=patch_locs, p_size=patch_size
            )
        else:
            xs, _ = utils.sampling.generalized_steps(
                x, x_cond, seq, self.model, self.betas, eta=0.
            )
        return xs[-1] if last else xs

    def sample_validation_patches(self, val_loader, step):
        # 저장 폴더 생성
        image_folder = os.path.join(self.args.image_folder, f"{self.config.data.dataset}{self.config.data.image_size}")
        step_dir = os.path.join(image_folder, str(step))
        os.makedirs(step_dir, exist_ok=True)

        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = data_transform(x[:, :3, :, :].to(self.device))
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(step_dir, f"{i}_cond.png"))
                utils.logging.save_image(x[i],      os.path.join(step_dir, f"{i}.png"))
