import os
import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from IPython.display import Image, display

# LangGraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages

# LangChain imports
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, trim_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# <=====================- Initialize the model -=====================>

# llm = ChatOllama(
#     model="llama3.2",
#     temperature=0,
# )

# https://platform.openai.com/settings/organization/usage
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

tool = TavilySearch(max_results=2)
tools = [tool]

llm_with_tools = llm.bind_tools(tools)


# <=====================- Initialize LangGraph -=====================>

# State class; add_messages annotation defines how this state key should be updated
class State(TypedDict):
    messages: Annotated[list, add_messages]


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Only use search tools when the user explicitly requests web search; "
            "otherwise, simply return a response without filling out anything into the tool schema."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

def chatbot(state: State):
    messages = prompt_template.format_messages(messages=state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": response}


# class BasicToolNode:
#     """Node that runs the tools requested in the last AIMessage"""

#     def __init__(self, tools: list) -> None:
#         self.tools_by_name = {tool.name: tool for tool in tools}

#     def __call__(self, inputs: dict):
#         messages = inputs.get("messages", [])
#         if messages:
#             message = messages[-1]
#         else:
#             raise ValueError("No messages found in the input")

#         outputs = []

#         for tool_call in message.tool_calls:
#             args = tool_call.get("args", {})
#             tool_name = tool_call.get("name", {})

#             if tool_name == "tavily_search":
#                 query_arg = args.get("query", "")
#                 if not isinstance(query_arg, str) or not query_arg.strip():
#                     continue
            
#             tool = self.tools_by_name.get(tool_name)
#             if not tool:
#                 raise ValueError(f"Tool {tool_name} not found")

#             # Call the tool with the arguments provided in the tool call
#             tool_result = tool.invoke(args)
#             outputs.append(
#                 ToolMessage(
#                     content=json.dumps(tool_result, default=str),
#                     name=tool_call["name"],
#                     tool_call_id=tool_call["id"],
#                 )
#             )
#         return {"messages": outputs}


# def route_tools(state: State):
#     """
#     Use in the conditional_edge to route to the ToolNode if last message has tool calls
#     """
#     if isinstance(state, list):
#         ai_message = state[-1]
#     elif state.get("messages", []):
#         ai_message = state.get("messages", [])[-1]
#     else:
#         raise ValueError("State does not contain messages")
    
#     # Return the correct path
#     if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
#         return "tools"
#     return END

# <=====================- Build the Graph -=====================>

workflow = StateGraph(state_schema=State)

# tool_node = BasicToolNode(tools=[tool])
tool_node = ToolNode(tools=[tool])

# Node name: Chatbot | Action: call the chatbot function
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", tool_node)

# Add entry point
workflow.add_edge(START, "chatbot")

# Call tools conditionally
workflow.add_conditional_edges(
    "chatbot", 
    tools_condition)

# Return tool output when called
workflow.add_edge("tools", "chatbot")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Visualize graph
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("Graph image saved as graph.png")


# <=====================- Run the chatbot -=====================>

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    user_msg = {"role": "user", "content": user_input}
    messages = [ user_msg ]
    for chunk, metadata in graph.stream(
        {"messages": messages},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, (HumanMessage, AIMessage)):
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
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            f.write(f"{role}: {content}\n\n")


while True:
    try:
        user_input = input("Query: ")
        if user_input in ["q", "quit"]:
            print("Done.")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print("\n Error: \n", e)


# messing with time
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state


print(to_replay.next)
print(to_replay.config)


for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()



# History management

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
