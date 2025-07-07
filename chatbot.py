import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import json

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_tavily import TavilySearch

from IPython.display import Image, display

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# <=====================- Initialize the model -=====================>

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

tool = TavilySearch(max_results=2)
tools = [tool]

llm_with_tools = llm.bind_tools(tools)


# <=====================- Initialize LangGraph -=====================>

# State class; add_messages annotation defines how this state key should be updated
class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """Node that runs the tools requested in the last AIMessage"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if messages:
            message = messages[-1]
        else:
            raise ValueError("No messages found in the input")

        outputs = []

        for tool_call in message.tool_calls:
            tool_name = tool_call["name"]
            tool = self.tools_by_name.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")

            # Call the tool with the arguments provided in the tool call
            tool_result = tool.invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


# def chatbot(state: State):
#     return {"messages": [llm.invoke(state["messages"])]}

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if last message has tool calls
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif state.get("messages", []):
        ai_message = state.get("messages", [])[-1]
    else:
        raise ValueError("State does not contain messages")
    
    # Return the correct path
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# <=====================- Build the Graph -=====================>

workflow = StateGraph(state_schema=State)

tool_node = BasicToolNode(tools=[tool])

# Node name: Chatbot, Action: call the chatbot function
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", tool_node)

# Add entry point
workflow.add_edge(START, "chatbot")

tool_node = BasicToolNode(tools=[tool])
workflow.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
workflow.add_edge("tools", "chatbot")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Visualize graph
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("Graph image saved as graph.png")


config = {"configurable": {"thread_id": "1"}}


# <=====================- Run the chatbot -=====================>

def stream_graph_updates(user_input: str):
    for chunk, metadata in graph.stream(
        {"messages" : [{"role": "user", "content": user_input}]},
        config,
        stream_mode="messages",
    ):
        print(chunk.content, end="", flush=True)
    
    print("\n")
    # capture the latest state snapshot
    snapshot = graph.get_state(config)
    # write plaintext snapshot to file

    # print(snapshot)
    with open("chat_history_snapshot.txt", "w") as f:
        for msg in snapshot.values["messages"]:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            f.write(f"{role}: {content}\n")


while True:
    try:
        user_input = input("Query: ")
        if user_input in ["q", "quit"]:
            print("Done.")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print("Error:", e)
