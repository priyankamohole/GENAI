from dotenv import load_dotenv
from langgraph.checkpoint import memory
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm=ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(model=llm , 
                     tools=[search.run],
                     system_prompt="you are an agent and can search on google ",
                     checkpointer=st.session_state.memory   
                     )

st.subheader("QuickAnswer : Get answer faster than you thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything ...")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    response = agent.invoke({"messages": [{"role": "user", "content": query}]},
                                 {"configurable": {"thread_id": "1"}} ,
                                configurable = "1")
    
    answer = response["messages"][-1].content
    st.chat_message("ai").markdown(answer)  
    st.session_state.history.append({"role": "ai", "content": answer})  

# while True:
#     query = input("Ask anything ...")
#     if query.lower() in ["exit","q" , "quit"]:
#         print("Exiting the agent...")
#         break
    
#     print( "AI:", response["messages"][-1].content)