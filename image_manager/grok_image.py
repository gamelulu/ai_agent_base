import os
from .base import BaseImageStrategy

class GrokImageStrategy(BaseImageStrategy):
    # 1~5 레벨별 권장/기본 모델 명시적 매핑
    LEVEL_MAP = {
        1: "grok-2-vision", # 가벼운 비전 모델 등 (템플릿)
        2: "grok-2-vision",
        3: "grok-2-image",  # [기본값] 표준 Flux 기반 모델
        4: "grok-imagine",  # 고급 모델 (예: flux pro 기반 등 가칭)
        5: "grok-imagine",  # 차세대 최고화질 엔진
    }

    def validate(self, model_val: str):
        if model_val and "dall-e" in model_val:
            raise ValueError(f"xAI(Grok) 제공자에 '{model_val}' 계열 모델을 달 수 없습니다.")

    def generate(self, prompt: str, model_val: str, level: int, size: str, **kwargs) -> str:
        try:
            from openai import OpenAI
            # xAI API는 OpenAI 클라이언트 문법을 100% 호환 적용합니다.
            client = OpenAI(
                api_key=os.environ.get("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        except ImportError:
            raise ImportError("Grok 이미지를 생성하기 위해 openai 패키지가 필요합니다.")
            
        # 모델명 결정: 유저가 직접 값을 넣었다면 그 값, 아니면 레벨 맵핑에서 가져옴
        model_name = model_val if model_val else self.LEVEL_MAP.get(level, self.LEVEL_MAP[3])
        
        # Grok API 역시 OpenAI SDK를 사용하므로 호환되는 확장 옵션(디테일 등) 자동 활성화
        model_quality = kwargs.get("quality", None)
        if not model_val:
            if level >= 5 and model_quality is None:
                model_quality = "hd"
            elif model_quality is None:
                model_quality = "standard"
        
        # API 호출 기본 파라미터 구성
        api_kwargs = {"model": model_name, "prompt": prompt, "n": 1, "size": size}
        if model_quality: api_kwargs["quality"] = model_quality
        
        try:
            response = client.images.generate(**api_kwargs)
            return response.data[0].url
        except Exception as e:
            print(f"\n[xAI Grok 이미지 생성 오류] {e}")
            return "https://picsum.photos/1024/1024"
