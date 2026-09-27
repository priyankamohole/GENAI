from dotenv import load_dotenv

load_dotenv() 

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

# question = "What is capital of india?"
# try :
#     result = llm.invoke(question)
#     print(result.content[0]["text"])

# except Exception as e:
#     print(f"An error occurred: {e}")


# while True:
#     query = input("User : ")

#     if query.lower() in ["exit", "quit", "q"]:
#         print("Exiting the chat.")
#         break;
#     result = llm.invoke(query)
#     print("Bot : ", result.content[0]["text"] ,"\n")

st.title("AI Q&A Bot")
st.markdown("Ask any question and get an answer from the bot.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask me anything...")
if query:
    # print(query)
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("User").markdown(query)
    try :
        res = llm.invoke(query)
        st.session_state.messages.append({"role": "bot", "content": res.content[0]["text"]})
        st.chat_message("Bot").markdown(res.content[0]["text"])
    except Exception as e:

        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):

            st.error(
                "⚠️ Gemini free-tier quota has been reached. "
                "Please wait until the quota resets and try again."
            )

        else:

            st.error(f"An error occurred: {e}")
    