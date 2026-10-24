from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

loader = PyPDFLoader("../pdf/medical_report.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splitted_data = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="all-minilm")

vector_store = InMemoryVectorStore.from_documents(
    documents= splitted_data,
    embedding=embeddings
)

# create a agent = llm , tools, prompt

llm = ChatGroq(model="openai/gpt-oss-20b")

@tool
def retriver_tool(query :str):
    """Retrive document relavant information form knowledge base"""
    docs = vector_store.similarity_search(query=query, k=4)
    context = ""
    for doc in docs:
        context = doc.page_content + "\n\n"

    return context

system_propmt = """You are a helpful assistant that answers question using retrived context
                    My knowledge base consists of the details from the uploaded documents.
                    Always use `retriver_tool` tool for question requiring external knowledge."""

memory = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[retriver_tool],
    system_prompt=system_propmt,
    checkpointer=memory 
)