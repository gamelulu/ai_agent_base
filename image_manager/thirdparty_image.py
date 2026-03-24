from .base import BaseImageStrategy

class ThirdPartyImageStrategy(BaseImageStrategy):
    """Midjourney, NovelAI, Nanobanana 등을 위한 범용 서드파티 통합 핸들러 템플릿"""
    
    LEVEL_MAP = {
        1: "midjourney-v5",
        2: "midjourney-v5",
        3: "midjourney-v5.2",  # [기본값]
        4: "midjourney-v6",    # 4단계 이상 고화질
        5: "midjourney-v6",
    }
    
    def generate(self, prompt: str, model_val: str, level: int, size: str, **kwargs) -> str:
        # 모델명 결정
        model_name = model_val if model_val else self.LEVEL_MAP.get(level, self.LEVEL_MAP[3])
            
        aspect_ratio = kwargs.get("aspect_ratio", "--ar 1:1")
        stylize = kwargs.get("stylize", 100)
        
        # 레벨 5 이상인 경우, 미드저니 등에서 고품질 플래그(--q 2 등)를 자동 적용
        quality_flag = kwargs.get("quality_flag", None)
        if not model_val:
            if level >= 5 and quality_flag is None:
                quality_flag = "--q 2" # 렌더링 타임 두배, 디테일 극대화
            elif quality_flag is None:
                quality_flag = "--q 1"
        
        print(f"[시스템] 서드파티({model_name})에서 이미지를 요청합니다. (비율:{aspect_ratio}, 스타일:{stylize}, 화질:{quality_flag})")
        return "https://picsum.photos/1024/1024"
