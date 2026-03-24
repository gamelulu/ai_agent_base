from .base import BaseLLMStrategy

class GoogleLLMStrategy(BaseLLMStrategy):
    def validate(self, model_val: str):
        if not model_val.startswith("gemini"):
            raise ValueError(f"Google 제공자에는 'gemini' 계열 모델만 사용할 수 있습니다. (입력됨: '{model_val}')")
            
    def create(self, model_val: str, temperature: float):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_val, temperature=temperature)
        except ImportError as e:
            raise ImportError("Google 모델을 사용하기 위해 패키지 설치가 필요합니다:\nuv add langchain-google-genai") from e
