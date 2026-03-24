class BaseImageStrategy:
    """이미지 생성 로직을 분리하는 전략 패턴 기본 클래스"""
    
    # 1~5 레벨별 권장/기본 모델 명시적 매핑 (개별 전략에서 오버라이드)
    LEVEL_MAP = {
        1: "default-level-1",
        2: "default-level-2",
        3: "default-level-3",
        4: "default-level-4",
        5: "default-level-5",
    }
    
    def validate(self, model_val: str):
        pass
        
    def generate(self, prompt: str, model_val: str, level: int, size: str, **kwargs) -> str:
        raise NotImplementedError("하위 클래스에서 generate 메서드를 구현해야 합니다.")
