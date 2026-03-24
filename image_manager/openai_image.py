from typing import Optional, Union, Dict, Any
from pydantic import BaseModel
from .base import BaseImageStrategy
from .schemas import validate_model_kwargs, ImageModel, ModelKwargsType

class OpenAIImageStrategy(BaseImageStrategy):
    # 1~5 레벨별 권장/기본 모델 명시적 매핑
    LEVEL_MAP = {
        1: "dall-e-2",  # 빠른 생성 (프로토타입)
        2: "dall-e-2",  # 기본 생성
        3: "dall-e-3",  # [기본값] 표준 고품질 모델
        4: "dall-e-3",  # 프롬프트 상세 반영
        5: "dall-e-3",  # 압도적인 디테일 (HD 퀄리티 강제)
    }

    def validate(self, model_val: str):
        if model_val and "grok" in model_val:
            raise ValueError(f"OpenAI 제공자에 '{model_val}' 계열 모델을 사용/할당할 수 없습니다.")

    def generate(self, prompt: str, model_val: str, level: int, options: Optional[Union[ModelKwargsType, Dict[str, Any]]] = None) -> str:
        try:
            from openai import OpenAI
            client = OpenAI() # .env의 OPENAI_API_KEY 자동 매핑
        except ImportError:
            raise ImportError("OpenAI 이미지를 생성하기 위해 openai 패키지가 필요합니다.\n설치: uv add openai")

        # 모델명 결정: 유저가 직접 값을 넣었다면 그 값, 아니면 레벨 맵핑에서 가져옴
        model_name = model_val if model_val else self.LEVEL_MAP.get(level, self.LEVEL_MAP[3])

        kwargs = self.parse_options(options)
        api_size = kwargs.get("size", "1024x1024")
        
        # DALL-E 3 전용 옵션 검증
        if model_name == "dall-e-3":
            kwargs = validate_model_kwargs(ImageModel.OPENAI_DALLE3.value, kwargs)

        model_quality = kwargs.get("quality", None)
        style = kwargs.get("style", None)

        if not model_val:
            if level >= 5 and model_quality is None:
                model_quality = "hd"
            elif model_quality is None:
                model_quality = "standard"

        # 2. API 호출 기본 파라미터 구성
        api_kwargs = {"model": model_name, "prompt": prompt, "n": 1, "size": api_size}

        # DALL-E 3 전용 옵션들 적용
        if model_name == "dall-e-3":
            if model_quality: api_kwargs["quality"] = model_quality
            if style: api_kwargs["style"] = style

        try:
            response = client.images.generate(**api_kwargs)
            return response.data[0].url
        except Exception as e:
            print(f"\n[OpenAI 이미지 생성 오류] {e}")
            return "https://picsum.photos/1024/1024"
