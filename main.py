import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from state import GraphState
from wrappers import handle_input, handle_chat, handle_image
from llm_manager import LLMManager
from image_manager import ImageManager
from ai_enums import LLMProvider, LLMModel, ImageProvider

load_dotenv()


# =======================================================================
# [앱 초기화(Configuration) 부분]
# =======================================================================
# 앱 전반적으로 깔려있을 "디폴트 기본 모델"을 하나 할당합니다 (비용이 싼 모델 권장)
LLMManager.configure(provider=LLMProvider.OPENAI, model=LLMModel.GPT_3_5_TURBO, temperature=0)


# =======================================================================
# [사용자 로직 작성] 
# =======================================================================

@handle_input
def dictation_node():
    """사용자 입력 노드"""
    print("-" * 30)
    return input("사용자: ")


@handle_chat
def chatbot_node(messages):
    """가벼운 대화를 쳐내는 기본 비서 노드 (빠른 디폴트 할당됨)"""
    return messages


# 이 부분이 핵심! 이 노드에만 고급 모델을 덮어씌워서 실행시킵니다.
@handle_chat(provider=LLMProvider.OPENAI, model=LLMModel.GPT_4O, temperature=0.8)
def creative_writer_node(messages):
    """
    창의적인 글을 작성하는 전문가 노드.
    상단의 `chatbot_node`와 똑같은 함수 구조임에도 불구하고, 
    데코레이터의 환경변수 덕택에 이 친구는 'gpt-4o' 두뇌로 실행됩니다!
    """
    # (선택) 프롬프트 조작이 필요하면 여기에 심기도 함
    return messages


@handle_image
def drawing_node(messages):
    """이미지 렌더링 로직 (ImageManager 활용)"""
    last_text = messages[-1].content
    print(f"[시스템] AI 화가가 '{last_text}' (을)를 기반으로 실제 그림을 그리는 중입니다...")
    
    # ImageManager를 호출하여 최고 수준 모델 사용 예시 작성
    image_url = ImageManager.create_image(
        prompt=last_text,
        provider=ImageProvider.OPENAI,
        level=5,
        size="1024x1024",
        style="vivid"
    )
    
    return image_url

# =======================================================================


def should_continue(state: GraphState):
    """다음 노드를 결정하는 엣지 단의 라우팅 함수"""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, HumanMessage):
        content = last_message.content.strip().lower()
        if content in ["종료", "끝", "quit", "exit", "q"]:
            return END
        elif "그려" in content or "이미지" in content:
            return "drawer"
        elif "창의적" in content or "소설" in content or "gpt-4" in content:
            # 창의적인 요청이 들어오면 전문가 노드(creative_writer)로 토스합니다.
            return "creative_writer"
            
    return "chatbot"


def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("user_input", dictation_node)
    
    # 두 개의 각기 다른 브레인(LLM)을 가진 채팅 노드가 등록됩니다.
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("creative_writer", creative_writer_node)
    
    workflow.add_node("drawer", drawing_node)
    
    workflow.add_edge(START, "user_input")
    workflow.add_conditional_edges("user_input", should_continue, {
        END: END, 
        "chatbot": "chatbot",
        "creative_writer": "creative_writer",
        "drawer": "drawer"
    })
    workflow.add_edge("chatbot", "user_input")
    workflow.add_edge("creative_writer", "user_input")
    workflow.add_edge("drawer", "user_input")
    
    return workflow.compile()


if __name__ == "__main__":
    print("=== 멀티 LLM 브레인 스위칭 에이전트 ===")
    print("('소설 써봐'라고 치면 강력한 GPT-4 머신이 가동됩니다.)")
    print("('그려줘'라고 입력하면 이미지 봇이 작동합니다.)")
    print("('종료', 'quit', 'q' 입력 시 종료됩니다.)\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[경고] OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.\n")
    
    # 그래프 실행
    app = build_graph()
    app.invoke({"messages": []}, {"recursion_limit": 100})
