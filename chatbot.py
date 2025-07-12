import os
from dotenv import load_dotenv
from typing import List, Any

# Langchain imports
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Langgraph imports
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.message import add_messages

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

        self.tools = {tool.name for tool in tools}
        self.llm = llm.bind_tools(tools)
        prompt_template = prompt_template


    def build_graph(self):
        pass

    
    def initialize_nodes(self):
        """
        Initializes all nodes that will be used in the graph
        """

        pass

    def run(self):
        """
        Starts the chatbot look
        TODO: Determine how this will work with the streamlit frontend, and if this is needed
        """
        pass

    def _stream_graph_updates(self, user_input: str):
        """
        Used to power the run function to start updates
        TODO: Determine how this will work with the streamlit frontend, and if this is needed
        """




