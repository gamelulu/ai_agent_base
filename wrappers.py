import inspect
from functools import wraps
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from state import GraphState
from llm_manager import LLMManager

def _get_user_kwargs(func, state: GraphState):
    sig = inspect.signature(func)
    kwargs = {}
    if "messages" in sig.parameters:
        kwargs["messages"] = state.get("messages", [])
    if "state" in sig.parameters:
        kwargs["state"] = state
    return kwargs

def _resolve_llm(provider, model, temperature):
    if provider or model or temperature is not None:
        return LLMManager.create_llm(
            provider=provider or "openai",
            model=model or "gpt-3.5-turbo",
            temperature=temperature if temperature is not None else 0
        )
    return LLMManager.get_llm()


def handle_input(func):
    """사용자의 입력을 텍스트로 받아 HumanMessage로 변환해주는 래퍼"""
    @wraps(func)
    def wrapper(state: GraphState):
        kwargs = _get_user_kwargs(func, state)
        result = func(**kwargs) if kwargs else func()
        
        if isinstance(result, dict): return result
        if isinstance(result, list): return {"messages": result}
            
        if isinstance(result, str):
            result = HumanMessage(content=result)
        elif not isinstance(result, BaseMessage):
            result = HumanMessage(content=str(result))
            
        return {"messages": [result]}
    return wrapper


def handle_chat(func=None, *, provider=None, model=None, temperature=None, stream=True):
    """
    일반 LLM 챗봇 전담 래퍼.
    스트리밍(stream)이 기본적으로 켜져 있어 글자가 타이핑되듯 출력됩니다!
    """
    if func is None:
        def decorator(f):
            return handle_chat(f, provider=provider, model=model, temperature=temperature, stream=stream)
        return decorator

    @wraps(func)
    def wrapper(state: GraphState):
        kwargs = _get_user_kwargs(func, state)
        processed_data = func(**kwargs) if kwargs else func()
        
        llm = _resolve_llm(provider, model, temperature)
        used_model = getattr(llm, "model_name", getattr(llm, "model", "default"))
            
        if isinstance(processed_data, BaseMessage) and not isinstance(processed_data, HumanMessage):
             return {"messages": [processed_data]}
             
        target_messages = processed_data if processed_data is not None else state.get("messages", [])
        
        # [핵심 로직] 스트리밍 처리 분기
        if stream and hasattr(llm, "stream"):
            print(f"AI [{used_model}]: ", end="", flush=True)
            chunks = []
            
            for chunk in llm.stream(target_messages):
                print(chunk.content, end="", flush=True)
                chunks.append(chunk)
            print()
            
            if chunks:
                response = chunks[0]
                for c in chunks[1:]:
                    response += c
            else:
                response = AIMessage(content="")
                
        else:
            response = llm.invoke(target_messages)
            print(f"AI [{used_model}]: {response.content}")
        
        return {"messages": [response]}
        
    return wrapper


def handle_image(func):
    """
    이미지 렌더링 노드 보조 래퍼.
    유저가 직접 작성한 API(DALL-E 등) 호출 함수의 반환값(URL)을 마크다운 포맷으로 깔끔하게 래핑합니다.
    (텍스트 챗봇과 달리 텍스트 언어모델을 중복 호출하지 않습니다)
    """
    @wraps(func)
    def wrapper(state: GraphState):
        kwargs = _get_user_kwargs(func, state)
        # 1. 유저의 함수(이미지 생성)가 실제로 실행됨 -> 이미지 URL 문자열 반환
        result = func(**kwargs) if kwargs else func()
        
        # 2. 반환 타입에 따른 시스템 예외 처리
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"messages": result}
        if isinstance(result, BaseMessage) and not isinstance(result, HumanMessage):
             return {"messages": [result]}
             
        # 3. [고유 후처리] URL 문자열이 정상 반환되었다면 마크다운 이미지 태그로 변환
        if isinstance(result, str) and result.startswith("http"):
            final_message = AIMessage(content=f"![생성된 이미지]({result})")
        else:
            final_message = AIMessage(content=f"[그림 생성기]: {result}")
            
        print(final_message.content)
        return {"messages": [final_message]}
        
    return wrapper


def handle_voice(func):
    """
    음성 렌더링 노드 보조 래퍼.
    유저가 반환한 음성 파일의 경로/URL을 보기 좋게 래핑합니다.
    """
    @wraps(func)
    def wrapper(state: GraphState):
        kwargs = _get_user_kwargs(func, state)
        result = func(**kwargs) if kwargs else func()
        
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"messages": result}
        if isinstance(result, BaseMessage) and not isinstance(result, HumanMessage):
             return {"messages": [result]}
             
        final_message = AIMessage(content=f"[음성 생성기]: {result}")
        print(final_message.content)
        return {"messages": [final_message]}
        
    return wrapper
