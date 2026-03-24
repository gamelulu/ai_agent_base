from .base import BaseLLMStrategy

class AnthropicLLMStrategy(BaseLLMStrategy):
    def validate(self, model_val: str):
        if not model_val.startswith("claude"):
            raise ValueError(f"Anthropic 제공자에는 'claude' 계열 모델만 사용할 수 있습니다. (입력됨: '{model_val}')")
            
    def create(self, model_val: str, temperature: float):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_val, temperature=temperature)
        except ImportError as e:
            raise ImportError("Anthropic 모델을 사용하기 위해 패키지 설치가 필요합니다:\nuv add langchain-anthropic") from e
