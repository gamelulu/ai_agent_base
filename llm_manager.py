class LLMManager:
    """
    LLM 선택과 초기화를 전담하는 매니저 클래스.
    이 파일로 분리됨으로써 그래프 작성 유저는 LangChain의 내장 코드를 몰라도 됩니다.
    
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
       - 특징: 최대 200만 토큰에 달하는 어마어마한 컨텍스트(기억력/문맥) 창을 보유하여 한 번에 책 단위로 읽음.
       
    4. Groq (오픈소스 초고속 추론 - 대표 모델: llama3-70b-8192, mixtral-8x7b-32768)
       - 추천 용도: 실시간 음성 대화 봇, 초고속 번역, 아주 단순하고 반복적인 분류형 작업.
       - 특징: 구글/오픈AI의 서버를 거치지 않고 특수 반도체(LPU)를 통해 미친듯한 초고속 텍스트 생성 속도 자랑.
    ==========================================================================================
    """
    _instance = None
    
    @classmethod
    def configure(cls, provider="openai", model="gpt-3.5-turbo", temperature=0):
        """앱 초기화 시점에 단 한 번 호출되어 시스템 전역 기본 LLM을 세팅합니다."""
        cls._instance = cls.create_llm(provider, model, temperature)
            
    @classmethod
    def create_llm(cls, provider="openai", model="gpt-3.5-turbo", temperature=0):
        """명시된 파라미터로 즉시 LLM 객체를 생성합니다! (노드별 개별 덮어쓰기 설정용)"""
        provider = provider.lower()
        
        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=model, temperature=temperature)
                
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(model=model, temperature=temperature)
                
            elif provider in ["google", "gemini"]:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(model=model, temperature=temperature)
                
            elif provider == "groq":
                from langchain_groq import ChatGroq
                return ChatGroq(model_name=model, temperature=temperature)
                
            else:
                raise ValueError(f"현재 지원하지 않는 LLM 제공자입니다: '{provider}'")
                
        except ImportError as e:
            raise ImportError(f"[{provider}] 모델을 사용하기 위해 패키지 설치가 필요합니다. 터미널에 다음을 입력하세요:\n"
                              f"uv add langchain-{provider if provider != 'gemini' else 'google-genai'}") from e

    @classmethod
    def get_llm(cls):
        """설정된 전역 LLM 객체를 반환합니다."""
        if cls._instance is None:
            cls.configure()
        return cls._instance
