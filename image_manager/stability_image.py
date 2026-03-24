from typing import Optional, Union, Dict, Any
from .base import BaseImageStrategy
from .schemas import validate_model_kwargs, ImageModel, ModelKwargsType

class StabilityImageStrategy(BaseImageStrategy):
    LEVEL_MAP = {
        1: "sdxl-turbo",                  # 초고속 렌더링
        2: "sdxl-turbo",
        3: "stable-diffusion-xl-1024-v1-0", # [기본값] 표준 고화질 SDXL
        4: "sd3-medium",                  # 최신 고화질 모델 (SD3)
        5: "sd3-large",                   # 최상위 모델 배정
    }

    def generate(self, prompt: str, model_val: str, level: int, options: Optional[Union[ModelKwargsType, Dict[str, Any]]] = None) -> str:
        # 모델명 결정: 유저가 직접 값을 넣었다면 그 값, 아니면 레벨 맵핑에서 가져옴
        model_name = model_val if model_val else self.LEVEL_MAP.get(level, self.LEVEL_MAP[3])
            
        kwargs = self.parse_options(options)
            
        kwargs = validate_model_kwargs(ImageModel.STABLE_DIFFUSION.value, kwargs)

        negative_prompt = kwargs.get("negative_prompt", "")
        guidance_scale = kwargs.get("guidance_scale", 7.0)
        
        # 레벨 5 이상인 경우, 명시되지 않았다면 디퓨전 스텝 수를 최대로 끌어올림 (고화질 연산)
        num_inference_steps = kwargs.get("num_inference_steps", None)
        if not model_val:
            if level >= 5 and num_inference_steps is None:
                num_inference_steps = 50 # 풀스텝(비용/시간 증가, 화질 상승)
            elif num_inference_steps is None:
                num_inference_steps = 30 # 표준 스텝
                
        print(f"[시스템] Stability 모델({model_name})로 이미지를 요청합니다. (레벨:{level}, CFG:{guidance_scale}, Steps:{num_inference_steps})")
        return "https://picsum.photos/1024/1024"
