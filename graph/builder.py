from functools import partial
from langgraph.graph import StateGraph, END
from graph.state import DocChatState
from graph.nodes import planner, retriever, generator


def build_graph(search_engine=None):
    """Build and compile the DocChat RAG pipeline graph."""
    workflow = StateGraph(DocChatState)

    workflow.add_node("planner", planner)
    workflow.add_node("retriever", partial(retriever, search_engine=search_engine))
    workflow.add_node("generator", generator)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()
