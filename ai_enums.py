from enum import Enum

# =======================================================================
# [LLM 텍스트 모델 관련 설정]
# =======================================================================
class LLMProvider(str, Enum):
    """LLM API 제공자"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"

class LLMModel(str, Enum):
    """주요 LLM 텍스트 생성 모델 목록"""
    # OpenAI
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    
    # Anthropic
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    
    # Google (Gemini)
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    
    # Groq (오픈소스 초고속 추론 특화)
    LLAMA_3_70B = "llama3-70b-8192"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"

# =======================================================================
# [이미지 생성 모델 관련 설정]
# =======================================================================
class ImageProvider(str, Enum):
    """이미지 API 제공자"""
    OPENAI = "openai"
    XAI = "xai"
    GROK = "grok"
    STABILITY = "stability"
    MIDJOURNEY = "midjourney"
    NOVELAI = "novelai"
    NANOBANANA = "nanobanana" # 파생 서비스 등
    GOOGLE = "google"         # Google Imagen 시리즈
