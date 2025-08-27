# datasets/__init__.py  ← 전체 교체

# AllWeather: finetune 버전이 있으면 그걸 우선 사용
try:
    from datasets_finetune.allweather_finetune import AllWeather as _AllWeather
    print("[datasets] AllWeather -> FINETUNE version")
except Exception:
    from .allweather import AllWeather as _AllWeather
    print("[datasets] AllWeather -> DEFAULT version")

# 나머지 로더도 필요하면 같은 방식으로 finetune 우선
try:
    from datasets_finetune.outdoorrain_finetune import OutdoorRain as _OutdoorRain
except Exception:
    from .outdoorrain import OutdoorRain as _OutdoorRain

try:
    from datasets_finetune.raindrop_finetune import RainDrop as _RainDrop
except Exception:
    from .raindrop import RainDrop as _RainDrop

try:
    from datasets_finetune.snow100k_finetune import Snow100K as _Snow100K
except Exception:
    from .snow100k import Snow100K as _Snow100K

# 외부로 노출되는 심볼
AllWeather  = _AllWeather
OutdoorRain = _OutdoorRain
RainDrop    = _RainDrop
Snow100K    = _Snow100K

__all__ = ["AllWeather", "OutdoorRain", "RainDrop", "Snow100K"]
