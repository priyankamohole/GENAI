from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm=ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory =MemorySaver()

agent = create_agent(model=llm , 
                     tools=[search.run],
                     system_prompt="you are an agent and can search on google ",
                     checkpointer=memory
                     )

st.subheader("QuickAnswer : Get answer faster than you thought")

query = st.chat_input("Ask anything ...")
if query:
    st.chat_message("user").markdown(query)
    response = agent.invoke({"messages": [{"role": "user", "content": query}]},
                                 {"configurable": {"thread_id": "1"}} ,
                                configurable = memory)

    answer = response["messages"][-1].content
    st.chat_message("ai").markdown(answer)  

# while True:
#     query = input("Ask anything ...")
#     if query.lower() in ["exit","q" , "quit"]:
#         print("Exiting the agent...")
#         break
    
#     print( "AI:", response["messages"][-1].content)