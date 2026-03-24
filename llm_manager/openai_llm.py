from .base import BaseLLMStrategy

class OpenAILLMStrategy(BaseLLMStrategy):
    def validate(self, model_val: str):
        if model_val.startswith(("claude", "gemini", "llama", "mixtral")):
            raise ValueError(f"OpenAI 제공자에 '{model_val}' 모델을 사용할 수 없습니다.")
            
    def create(self, model_val: str, temperature: float):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_val, temperature=temperature)
        except ImportError as e:
            raise ImportError("OpenAI 모델을 사용하기 위해 패키지 설치가 필요합니다:\nuv add langchain-openai") from e
