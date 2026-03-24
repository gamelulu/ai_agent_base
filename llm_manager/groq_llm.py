from .base import BaseLLMStrategy

class GroqLLMStrategy(BaseLLMStrategy):
    def validate(self, model_val: str):
        if model_val.startswith(("gpt", "claude", "gemini", "o1", "o3")):
             raise ValueError(f"Groq 제공자에 타사의 상용 모델('{model_val}')을 할당할 수 없습니다. 오픈소스 모델만 지원됩니다.")
             
    def create(self, model_val: str, temperature: float):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model_name=model_val, temperature=temperature)
        except ImportError as e:
            raise ImportError("Groq 모델을 사용하기 위해 패키지 설치가 필요합니다:\nuv add langchain-groq") from e
