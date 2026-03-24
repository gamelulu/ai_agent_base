class BaseLLMStrategy:
    """LLM 객체 생성 및 검증을 위한 전략 패턴(Strategy Pattern) 기본 클래스"""
    
    def validate(self, model_val: str):
        pass
        
    def create(self, model_val: str, temperature: float):
        raise NotImplementedError("하위 클래스에서 create 메서드를 구현해야 합니다.")
