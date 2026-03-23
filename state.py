from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# 프레임워크 사용자/개발자 모두가 공유하는 상태 데이터 스키마 정의
class GraphState(TypedDict):
    # add_messages 리듀서를 사용하여 메시지 목록이 누적되도록 합니다.
    messages: Annotated[list[AnyMessage], add_messages]
