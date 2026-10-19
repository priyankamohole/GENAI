from dotenv import load_dotenv
load_dotenv()

# llm , tools , agent , system prompt , DB

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

db = SQLDatabase.from_uri("sqlite:///my_tasks.db")
db.run(
    """ CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT ,
            status TEXT CHECK (status IN ('pending' , 'in_process', 'completed')) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
"""
)

print("DB TABLE connected successfully")