from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq

llm=ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

agent = create_agent(model=llm , 
                     tools=[search.run],
                     system_prompt="you are an agent and can search on google ")

while True:
    query = input("Ask anything ...")
    if query.lower() in ["exit","q" , "quit"]:
        print("Exiting the agent...")
        break
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print( "AI:", response["messages"][-1].content)