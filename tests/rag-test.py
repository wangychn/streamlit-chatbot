import os
os.environ["USER_AGENT"] = "rag-testing"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bs4
import tiktoken
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

# Ollama Imports
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# <===========================- MISC. FUNCTIONS -==============================>

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string. For prompt budgeting"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# <===========================- INDEXING -==============================>

# LOADING THE BLOG
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

blog_docs = loader.load()

# DECLARE THE EMBEDDING
embd = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# MAKE SPLITTER
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# MAKE SPLITS
splits = text_splitter.split_documents(blog_docs)


# <===========================- RETREIVAL -==============================>

# Creating Chroma vector store
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embd,)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


docs = retriever.invoke("What is Task Decomposition?")


# <===========================- GENERATION -==============================>


# Prompt
template = """Answer the question based only on the following context;
if you do not know the information, just say so.

{context}

Question begins now: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Instantiate the local Ollama LLM
llm = OllamaLLM(model="llama3.2")

# Build a RetrievalQA chain with your custom prompt and retriever
combine_docs_chain = prompt | llm

# Let the chain handle retrieval; only pass the question

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print("Source:", doc.metadata.get("source", "N/A"))
    print("Content:\n", doc.page_content)
    print("-" * 40)

answer = combine_docs_chain.invoke({
    "context" : docs,
    "question": "What is Task Decomposition?"
    
})

print("Answer:", answer)