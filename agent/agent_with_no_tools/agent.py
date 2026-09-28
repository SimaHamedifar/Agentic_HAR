from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from config import PROMPT_TEXT

class State(TypedDict):
    event: str
    event_sequence: str
    detected_activity: str

def build_agent(llm):
    def activity_detector(state: State) -> dict:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEXT)
        chain = prompt | llm
        response = chain.invoke({
            "event_sequence": state["event_sequence"], 
            "event": state.get("event", '')
        })
        return {**state, "detected_activity": response.content}

    workflow = StateGraph(State)
    workflow.add_node("activity_detector", activity_detector)
    workflow.add_edge(START, "activity_detector")
    workflow.add_edge("activity_detector", END)
    
    return workflow.compile()
