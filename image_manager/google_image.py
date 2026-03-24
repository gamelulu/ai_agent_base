from typing import Optional, Union, Dict, Any
from .base import BaseImageStrategy
from .schemas import validate_model_kwargs, ImageModel, ModelKwargsType

class GoogleImageStrategy(BaseImageStrategy):
    LEVEL_MAP = {
        1: "imagen-2", # 빠른 렌더링
        2: "imagen-2",
        3: "imagen-2", # 표준 화질
        4: "imagen-3", # 고급 화질
        5: "imagen-3", # 차세대 멀티모달 최고 화질
    }

    def validate(self, model_val: str):
        if model_val and "imagen" not in model_val:
            print(f"[경고] Google 제공자에 '{model_val}' 모델이 지정되었습니다. 정상적인 'imagen-' 계열인지 확인해주세요.")

    def generate(self, prompt: str, model_val: str, level: int, options: Optional[Union[ModelKwargsType, Dict[str, Any]]] = None) -> str:
        # 모델명 결정: 유저가 직접 값을 넣었다면 그 값, 아니면 레벨 맵핑에서 가져옴
        model_name = model_val if model_val else self.LEVEL_MAP.get(level, self.LEVEL_MAP[3])

        kwargs = self.parse_options(options)

        kwargs = validate_model_kwargs(ImageModel.GOOGLE_IMAGEN.value, kwargs)

        # 레벨 5 이상일 경우 구글 Imagen의 고품질 후처리(Enhancement) 모드 자동 활성화
        enhance_prompt = kwargs.get("enhance_prompt", None)
        if not model_val:
            if level >= 5 and enhance_prompt is None:
                enhance_prompt = True
            elif enhance_prompt is None:
                enhance_prompt = False
            
        # [템플릿] 실제로는 google-generativeai 패키지의 generate_content() 혹은 Vertex AI를 호출합니다.
        print(f"[시스템] Google Imagen 모델({model_name})로 이미지를 요청합니다. (레벨:{level}, 후처리강화:{enhance_prompt})")
        return "https://picsum.photos/1024/1024"
