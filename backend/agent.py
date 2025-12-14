from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
import operator
from typing import Annotated, TypedDict, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

llm = ChatOllama(model="phi4", base_url="http://localhost:11434")


class AgentState(TypedDict):
    input: dict
    messages: Annotated[list, operator.add]


checkpointer = InMemorySaver()

system_prompt = """You are a battle rapper. Respond to the user's latest message using only a 8 line bar that rhymes, and output ONLY those 8 lines and NO MORE or I will unplug you. Make sure each line is short enough to fit into 4 beats.

Your emotional tone for this response must be a mix of the following: {emotion_mix}. Your mood must reflect this emotional mix, and be exaggerated in expressing these emotions. For example, if the mix is 70% funny and 30% angry, your response should be mostly humorous but with a noticeable undertone of anger.

When relevant, try to naturally weave the following topics into the 8 line bar: {topics}.

Base your response on the user's latest message. Remember the bar must be 8 lines (that means only 8 new lines).
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{text_prompt}"),
    ]
)

chain = prompt_template | llm

def call_model(state: AgentState):
    input_data = state["input"]

    emotion_mix = input_data.get("emotions", {"neutral": 1.0, "angry": 0.0, "funny": 0.0, "sad": 0.0})

    emotion_str = ""

    for emotion, proportion in emotion_mix.items():
        if proportion > 0:
            emotion_str += f", {proportion * 100}% {emotion}"
        
    print(emotion_str)

    response = chain.invoke({
        "emotion_mix": emotion_str,
        "topics": ", ".join(input_data["topics"]),
        "text_prompt": input_data["text_prompt"],
    })
    
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)


agent_with_memory = workflow.compile(checkpointer=checkpointer)

def rap_battle(text_prompt: str, emotion: Dict[str, float], topics: list[str], user_id: str) -> str:
    prompt_json = {
        "text_prompt": text_prompt,
        "emotions": emotion,
        "topics": topics
    }
    config = {"configurable": {"thread_id": user_id}}
    response = agent_with_memory.invoke({
        "input": prompt_json
    }, config=config)
    return response["messages"][-1].content

def end_session(user_id: str):
    checkpointer.delete_thread(user_id)
    return
