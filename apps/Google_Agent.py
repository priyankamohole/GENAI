from dotenv import load_dotenv
from langgraph.checkpoint import memory


from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
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

    response = agent.stream({"messages": [{"role": "user", "content": query}]},
                                {"configurable": {"thread_id": "1"}} ,
                                stream_mode="messages"
                                )

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)
    # answer = response["messages"][-1].content
    # st.chat_message("ai").markdown(answer)  
        st.session_state.history.append({"role": "ai", "content": message})  

# while True:
#     query = input("Ask anything ...")
#     if query.lower() in ["exit","q" , "quit"]:
#         print("Exiting the agent...")
#         break
    
#     print( "AI:", response["messages"][-1].content)