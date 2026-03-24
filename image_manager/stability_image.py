from .base import BaseImageStrategy

class StabilityImageStrategy(BaseImageStrategy):
    LEVEL_MAP = {
        1: "sdxl-turbo",                  # 초고속 렌더링
        2: "sdxl-turbo",
        3: "stable-diffusion-xl-1024-v1-0", # [기본값] 표준 고화질 SDXL
        4: "sd3-medium",                  # 최신 고화질 모델 (SD3)
        5: "sd3-large",                   # 최상위 모델 배정
    }

    def generate(self, prompt: str, model_val: str, level: int, size: str, **kwargs) -> str:
        # 모델명 결정: 유저가 직접 값을 넣었다면 그 값, 아니면 레벨 맵핑에서 가져옴
        model_name = model_val if model_val else self.LEVEL_MAP.get(level, self.LEVEL_MAP[3])
            
        negative_prompt = kwargs.get("negative_prompt", "")
        cfg_scale = kwargs.get("cfg_scale", 7.0)
        
        # 레벨 5 이상인 경우, 명시되지 않았다면 디퓨전 스텝 수를 최대로 끌어올림 (고화질 연산)
        steps = kwargs.get("steps", None)
        if not model_val:
            if level >= 5 and steps is None:
                steps = 50 # 풀스텝(비용/시간 증가, 화질 상승)
            elif steps is None:
                steps = 30 # 표준 스텝
                
        print(f"[시스템] Stability 모델({model_name})로 이미지를 요청합니다. (레벨:{level}, CFG:{cfg_scale}, Steps:{steps})")
        return "https://picsum.photos/1024/1024"
