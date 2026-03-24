from typing import Union, Dict
from ai_enums import ImageProvider
from .base import BaseImageStrategy
from .openai_image import OpenAIImageStrategy
from .grok_image import GrokImageStrategy
from .stability_image import StabilityImageStrategy
from .thirdparty_image import ThirdPartyImageStrategy
from .google_image import GoogleImageStrategy

class ImageManager:
    """
    이미지 생성 모델 및 옵션을 관리하는 매니저 클래스.
    이 파일로 분리됨으로써 그래프 작성 유저는 복잡한 API 호출 구조를 몰라도 됩니다.
    
    [전략 패턴 적용]
    새로운 이미지 생성 엔진을 추가하려면 BaseImageStrategy를 상속받아 만들고 
    _STRATEGIES 레지스트리에 꽂아만 주면 됩니다!
    
    ==========================================================================================
    💡 [주요 이미지 생성 모델 가이드 - 레벨(Level) 1~5]
    
    1. Level 1 (빠르고 저렴한 프로토타이핑)
       - OpenAI: DALL-E 2 (저해상도) / Stability: Stable Diffusion v1.5 / SDXL Turbo 
       
    2. Level 2 (중간 화질, 적절한 속도)
       - OpenAI: DALL-E 2 (1024x1024) / 미드저니(Midjourney): V5
       
    3. Level 3 (표준 고품질 - 기본값)
       - OpenAI: DALL-E 3 (standard) / NovelAI, 나노바나나 등 (애니/파생형 특화) / Google: Imagen 2
       
    4. Level 4 (예술성, 극 사실주의 등 전문 영역)
       - Stability: SD 3 Medium / Grok: grok-2-image
       
    5. Level 5 (현존 최고 화질 및 차세대 멀티모달)
       - OpenAI: DALL-E 3 (hd 품질) / Google: Imagen 3
    ==========================================================================================
    
    💡 [주요 모델별 전용 옵션(프로퍼티) 가이드] (`kwargs`를 통해 자유롭게 전달)
    [OpenAI DALL-E 3 전용]
    - style (str): "vivid", "natural" / quality (str): "standard", "hd" 
    
    [Stable Diffusion 전용]
    - negative_prompt (str) / guidance_scale (float)
    
    [Midjourney 전용]
    - aspect_ratio (str) / stylize (int)
    ==========================================================================================
    """
    
    # 팩토리 전략 레지스트리
    _STRATEGIES: Dict[str, BaseImageStrategy] = {
        ImageProvider.OPENAI.value: OpenAIImageStrategy(),
        ImageProvider.XAI.value: GrokImageStrategy(),
        ImageProvider.GROK.value: GrokImageStrategy(),
        ImageProvider.STABILITY.value: StabilityImageStrategy(),
        ImageProvider.MIDJOURNEY.value: ThirdPartyImageStrategy(),
        ImageProvider.NOVELAI.value: ThirdPartyImageStrategy(),
        ImageProvider.NANOBANANA.value: ThirdPartyImageStrategy(),
        ImageProvider.GOOGLE.value: GoogleImageStrategy(),
    }

    @classmethod
    def create_image(cls, prompt: str, provider: Union[str, ImageProvider] = ImageProvider.OPENAI, model: str = None, level: int = 3, size: str = "1024x1024", **kwargs):
        """
        주어진 옵션으로 이미지를 생성하고 URL을 반환합니다.
        
        Args:
            prompt (str): 이미지 생성을 위한 프롬프트
            provider (Union[str, ImageProvider]): 주요 제공자 
            model (Union[str, ImageModel]): 강제로 지정할 특정 모델명
            level (int): 1~5 단계의 모델 수준 구분 (model 인자가 없을때 작동)
            size (str): 해상도 (e.g. "1024x1024" (default))
            kwargs: 각 모델별 전용(특수) 파라미터들 (style, negative_prompt 등)
        """
        provider_val = provider.value if hasattr(provider, "value") else str(provider).lower()
        model_val = model.value if hasattr(model, "value") else str(model) if model else None
        
        # 1. 팩토리에서 핸들러 가져오기
        strategy = cls._STRATEGIES.get(provider_val)
        if not strategy:
            raise ValueError(f"현재 지원하지 않는 이미지 provider입니다: '{provider_val}'")
            
        # 2. 입력된 Model과 Provider 상성 1차 검증 (보안벽)
        strategy.validate(model_val)
        
        # 3. 객체 및 렌더링 호출
        return strategy.generate(prompt, model_val, level, size, **kwargs)
