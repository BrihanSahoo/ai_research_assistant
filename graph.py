from nodes import (
    analyzer_node,
    planer_node,
    researcher_node,
    generator_node,
    evaluator_node,
    save_response_node,
    rag_retriver_node
)

from state import ResearchState
from langgraph.graph import START,StateGraph,END
from langgraph.checkpoint.memory import InMemorySaver


# Add nodes
workflow = StateGraph(ResearchState)
workflow.add_node("analyzer",analyzer_node)
workflow.add_node("planner",planer_node)
workflow.add_node("researcher",researcher_node)
workflow.add_node("generator",generator_node)
workflow.add_node("evaluator",evaluator_node)
workflow.add_node("save_response", save_response_node)
workflow.add_node("rag_retriever",rag_retriver_node)

# Add edges
workflow.add_edge(START, "analyzer")
workflow.add_edge(
    "analyzer",
    "rag_retriever"
)

workflow.add_edge(
    "rag_retriever",
    "planner"
)
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "generator")
workflow.add_edge("generator", "evaluator")
workflow.add_edge("save_response", END) # Finish after saving the response



def route_after_evaluation(state:ResearchState):
    MAX_ITERATIONS = 3
    if state['approved']:
        return "finish"
    if state['iteration']>=MAX_ITERATIONS:
        return "finish"
    return "retry"


workflow.add_conditional_edges(
    "evaluator",                        # Decision making node
    route_after_evaluation,             # Routing function
    {
        "finish" :"save_response",      # Save the response
        "retry": "researcher"           # Start research again
    }
)

checkpointer = InMemorySaver()

app = workflow.compile(
    checkpointer=checkpointer
)