from dotenv import load_dotenv
load_dotenv()

# llm , tools , agent , system prompt , DB

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

print("DB connected successfully")