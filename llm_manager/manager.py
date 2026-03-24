from typing import Union, Dict
from ai_enums import LLMProvider, LLMModel
from .base import BaseLLMStrategy
from .openai_llm import OpenAILLMStrategy
from .anthropic_llm import AnthropicLLMStrategy
from .google_llm import GoogleLLMStrategy
from .groq_llm import GroqLLMStrategy

class LLMManager:
    """
    LLM 선택과 초기화를 전담하는 매니저 클래스.
    이 파일로 분리됨으로써 그래프 작성 유저는 LangChain의 내장 코드를 몰라도 됩니다.
    
    [전략 패턴 적용]
    새로운 LLM 제공자를 추가하려면 BaseLLMStrategy를 상속받은 새 클래스를 만들고 
    _STRATEGIES 딕셔너리에 등록해주기만 하면 됩니다! (OCP - 개방 폐쇄 원칙 준수)
    
    ==========================================================================================
    💡 [노드별 LLM 선택 가이드라인] - 상황에 맞는 최고의 모델을 노드 데코레이터에 할당하세요!
    
    1. OpenAI (대표 모델: gpt-3.5-turbo, gpt-4o, gpt-4o-mini)
       - 추천 용도: 전천후 범용 모델, 일상적인 대화, 안정적인 JSON 출력, API 함수 호출(Tool Calling).
       - 특징: 가장 대중적이고 호출 속도와 성능 밸런스가 뛰어나며, 에러가 적음. (전역 기본값으로 쓰기 좋음)
       
    2. Anthropic (대표 모델: claude-3-5-sonnet-20240620, claude-3-opus-20240229)
       - 추천 용도: 프로그래밍(코딩), 심도 깊은 글쓰기, 기계적이지 않고 자연스러운 한국어 문장 작성.
       - 특징: 코딩 실력이 현존 최고 수준으로 평가받으며, 어투가 매우 지능적이고 인간적임.
       
    3. Google / Gemini (대표 모델: gemini-1.5-pro, gemini-1.5-flash)
       - 추천 용도: 방대한 PDF 문서 분석, 긴 대화 로그 요약 분석, 영상/이미지 등 멀티모달 처리.
       - 특징: 최대 200만 토큰에 달하는 어마어마한 컨텍스트 창을 보유하여 한 번에 책 단위로 읽음.
       
    4. Groq (오픈소스 초고속 추론 - 대표 모델: llama3-70b-8192, mixtral-8x7b-32768)
       - 추천 용도: 실시간 음성 대화 봇, 초고속 번역, 아주 단순하고 반복적인 분류형 작업.
       - 특징: 구글/오픈AI의 서버를 거치지 않고 특수 반도체(LPU)를 통해 초고속 텍스트 생성 속도 자랑.
    ==========================================================================================
    """
    _instance = None
    
    # 팩토리 전략 레지스트리
    _STRATEGIES: Dict[str, BaseLLMStrategy] = {
        LLMProvider.OPENAI.value: OpenAILLMStrategy(),
        LLMProvider.ANTHROPIC.value: AnthropicLLMStrategy(),
        LLMProvider.GOOGLE.value: GoogleLLMStrategy(),
        "gemini": GoogleLLMStrategy(), # 하위 호환 매핑
        LLMProvider.GROQ.value: GroqLLMStrategy(),
    }
    
    @classmethod
    def configure(cls, provider: Union[str, LLMProvider] = LLMProvider.OPENAI, model: Union[str, LLMModel] = LLMModel.GPT_3_5_TURBO, temperature=0):
        """앱 초기화 시점에 단 한 번 호출되어 시스템 전역 기본 LLM을 세팅합니다."""
        cls._instance = cls.create_llm(provider, model, temperature)
            
    @classmethod
    def create_llm(cls, provider: Union[str, LLMProvider] = LLMProvider.OPENAI, model: Union[str, LLMModel] = LLMModel.GPT_3_5_TURBO, temperature=0):
        """명시된 파라미터로 즉시 LLM 객체를 생성합니다! (노드별 개별 덮어쓰기 설정용)"""
        # Enum이 들어왔을 경우 문자열(.value)로 안전하게 변환
        provider_val = provider.value if hasattr(provider, "value") else str(provider).lower()
        model_val = model.value if hasattr(model, "value") else str(model)
        
        # 1. 팩토리에서 전략 핸들러 가져오기
        strategy = cls._STRATEGIES.get(provider_val)
        if not strategy:
            raise ValueError(f"현재 지원하지 않는 LLM 제공자입니다: '{provider_val}'")
            
        # 2. 강력한 Model 크로스체크 검증 수행
        strategy.validate(model_val)
        
        # 3. 객체 생성 통과!
        return strategy.create(model_val, temperature)

    @classmethod
    def get_llm(cls):
        """현재 세팅된 기본 LLM 객체를 가져옵니다."""
        if cls._instance is None:
            cls.configure() # 초기화 안 하고 불렀을 땐 안전하게 기본값 사용
        return cls._instance
