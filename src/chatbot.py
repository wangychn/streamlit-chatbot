import os
from dotenv import load_dotenv
from typing import List, Any, TypedDict, Annotated

# Langchain imports
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Langgraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class chatbot:

    def __init__(
            self, 
            llm: ChatOllama | ChatOpenAI, 
            tools: List[Any],
            prompt_template: ChatPromptTemplate
        ):
        """
        Initializes the key functions for the chatbot
        - llm: LangChain llm object
        - tools: list of tool instances (e.g. TavilySearch)
        - prompt_template: a ChatPromptTemplate
        """

        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm.bind_tools(tools)
        self.prompt_template = prompt_template

    
    def initialize_nodes(self) -> None:
        """
        Initializes all nodes that will be used in the graph
        """

        # Define all the states and nodes that will be used
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def _chat_node(state: State):
            messages = self.prompt_template.format_messages(messages=state["messages"])
            response = self.llm.invoke(messages)
            return {"messages": response}
        
        # Initialize the workflow and nodes
        self.workflow = StateGraph(state_schema=State)
        self.tool_node = ToolNode(tools=list(self.tools.values()))
        
        # Add the nodes to the graph
        self.workflow.add_node("chatbot", _chat_node)
        self.workflow.add_node("tools",  self.tool_node)


    def build_graph(self) -> None:
        """
        Builds the graph with the initialized nodes; call this after `initialize_nodes()`
        """

        # Entry point
        self.workflow.add_edge(START, "chatbot")

        # Conditional tool call
        self.workflow.add_conditional_edges(
            "chatbot",
            tools_condition
        )

        # Returning tool output
        self.workflow.add_edge("tools", "chatbot")

        self.memory = MemorySaver()
        self.graph = self.workflow.compile(checkpointer=self.memory)

        # Optional: Visualize the graph
        with open("chatbot_graph.png", "wb") as f:
            f.write(self.graph.get_graph().draw_mermaid_png())
        print("Graph image saved as graph.png")


    def invoke(self, user_input):
        """
        Takes a single input message and call the graph, streaming the output
        """

        user_msg = {"role": "user", "content": user_input}
        messages = [ user_msg ]
        config = {"configurable": {"thread_id": "1"}}
        return self.graph.stream(
            {"messages": messages},
            config,
            stream_mode="messages",
        )





