
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import ArxivRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama.llms import OllamaLLM
from langsmith import traceable
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "pr-ample-infix-5"

retriever = ArxivRetriever(
    load_max_docs=2,
    get_ful_documents=True,
)

llm = OllamaLLM(model="llama3.2")


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)


def format_docs(docs):
    context =  "\n\n".join(doc.page_content for doc in docs)
    # print(context)
    return context

@traceable(name="paperRetrievalRun")
def rag(question):

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

def main():
    question = "What is Deep Q-Learning in Reinforcement Learning?"
    result = rag(question)
    print(result)

if __name__ == "__main__":
    main()
