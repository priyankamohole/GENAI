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
import streamlit as st


if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
    # Load document
    loader = PyPDFLoader("../pdf/medical_report.pdf")
    docs = loader.load()

    # spilt document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splitted_data = splitter.split_documents(docs)

    # (converting doc into numberical format and store it on DB)embedding with vector database
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

    #   memory for agent to remember chat
    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retriver_tool],
        system_prompt=system_propmt,
        checkpointer=memory 
    )

    st.session_state.file_uploaded = True
    st.session_state.agent = agent
# while True:
#     query = input("User :")
#     if query.lower in ["q", "quit", "bye"]:
#          break

#     response = agent.invoke({"messages":[{"role": "user", "content":query}]},
#                             { "configurable":{"thread_id":"1bjh"}})
#     result = response["messages"][-1].content
#     print("AI :",result)


# uploade ui

if not st.session_state.file_uploaded:
    uploaded = st.file_uploader(label = "Select PDF Files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "../pdf/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()

# chat ui
if st.session_state.file_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything relatef to uploaded documents...")
    if query :
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role":"user","content":query})
       
        response = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"configurable": {"thread_id": "1"}}
                )
        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer) 
        st.session_state.messages.append({"role":"ai","content":answer})