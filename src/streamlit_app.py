import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.retrievers import ArxivRetriever
import os
from dotenv import load_dotenv
from chatbot import chatbot

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "pr-ample-infix-5"

st.title("Quickstart App")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# # Initialize the chatbot
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


# Initialize Arxiv retriever
retriever = ArxivRetriever(
    load_max_docs=2,
    get_full_documents=True,
)


# Initialize Arxiv retriever as a tool
@tool(response_format="content_and_artifact")
def retrieve(query : str):
    """Retrieve relevant documents from Arxiv and serialize them for the assistant."""
    retrieved_docs = retriever.invoke(query)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Function to initialize the chatbot
@st.cache_resource
def initialize_chatbot():
    prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Only use the retrieve tool when the user mentions the need to investigate papers; "
            "otherwise, simply return a response without filling out anything into the tool schema."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

    chatbot_instance = chatbot(llm, [retrieve], prompt_template)
    chatbot_instance.initialize_nodes()
    chatbot_instance.build_graph()
    return chatbot_instance

# Generator function for extracting content from stream
def stream_content_only(stream):
    for chunk, metadata in stream:
        # print("*" * 100)
        # print(metadata)
        # print("*" * 100)
        # print(chunk)
        if hasattr(chunk, "content") and metadata['langgraph_node'] == 'chatbot':
            yield chunk.content


# <=======================- Actual App Displays -==========================>

if __name__ == "__main__":
    # Initialize the chatbot instance
    cb = initialize_chatbot()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    # Display every chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Takes the user input and generate it; disabled while streaming, but block still accessible
    if user_input := st.chat_input("What is up?", disabled=st.session_state.streaming) or st.session_state.streaming:

        # (1) Add user message; Display user message; turn on streaming
        if not st.session_state.streaming:
            with st.chat_message("human"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "human", "content": user_input}) 

            st.session_state.streaming = True
            st.rerun()

        # (2) Stream the chat response, and display it
        with st.chat_message("assistant"):
            stream = cb.invoke(st.session_state.messages[-1]['content'])
            full_response = st.write_stream(stream_content_only(stream))

        # (3) Turn off streaming and rerender
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.streaming = False
        st.rerun()
            




            


