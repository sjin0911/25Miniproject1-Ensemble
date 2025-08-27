# utils/__init__.py
import importlib, sys

# 우선 적용할 서브모듈 이름들(레포 상황 맞춰 추가/삭제)
CANDIDATES = [
    "optimize",   # 옵티마이저 (AdamW 추가한 곳)
    "logging",    # 로깅 유틸
    "tta",        # test-time augmentation (있다면)
    "metrics",    # 지표 모듈(있다면)
    "sampling",   # 샘플링 유틸(있다면)
]

# utils_finetune가 있으면 그걸 우선, 없으면 기본 utils 서브모듈
def _load(name: str):
    # finetune 우선
    try:
        mod = importlib.import_module(f"utils_finetune.{name}")
        print(f"[utils] {name} -> FINETUNE")
    except Exception:
        # 기본 utils 서브모듈(동일 패키지 내)
        mod = importlib.import_module(f".{name}", __name__)
        print(f"[utils] {name} -> DEFAULT")
    # 'import utils.name' 과 'from utils import name' 둘 다 먹게 등록
    setattr(sys.modules[__name__], name, mod)
    sys.modules[f"{__name__}.{name}"] = mod
    return mod

# 로드 실행
_loaded = {name: _load(name) for name in CANDIDATES if name}

# 외부로 노출
__all__ = list(_loaded.keys())
