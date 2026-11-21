from typing import Annotated, TypedDict,List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ResearchState(TypedDict):
    question: str

    # Analysis
    analysis: str

    # Research plan
    research_plan: List[str]

    # Research results
    research_results: List[dict]

    # Generated answer
    answer: str

    # Evaluation feedback
    feedback: str

    # Evaluation status
    approved: bool

    # Number of attempts
    iteration: int

    # Conversation history
    messages: Annotated[List[BaseMessage], add_messages]


## LangGraph passes state from one node to another. A node typically returns only the fields it wants to update, rather than manually passing all the data around.