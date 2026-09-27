from dotenv import load_dotenv

load_dotenv() 

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

# question = "What is capital of india?"
# try :
#     result = llm.invoke(question)
#     print(result.content[0]["text"])

# except Exception as e:
#     print(f"An error occurred: {e}")


while True:
    query = input("User : ")

    if query.lower() in ["exit", "quit", "q"]:
        print("Exiting the chat.")
        break;
    result = llm.invoke(query)
    print("Bot : ", result.content[0]["text"] ,"\n")