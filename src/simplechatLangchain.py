import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
# 1. Import the Langfuse Callback Handler
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# 1. Initialize our free Groq model
model = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.7,
        api_key=GROQ_API_KEY
    )
# 2. Update our prompt to allow a list of past messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, friendly AI coach."),
    # This placeholder is where LangChain will inject the back-and-forth history
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
# 3. Create the base chain
chain = prompt | model
# 4. Create a dictionary to hold multi-user session histories in memory
session_store = {}
# 3. Initialize the handler
langfuse_handler = CallbackHandler()
def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]
# 5. Wrap our chain with memory management capabilities
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
# 6. Let's create an interactive chat loop in the terminal!
print("Chatbot initialized! Type 'exit' to quit.\n")
config = {"configurable": {"session_id": "learning_session_1"}}
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    print("AI: ", end="")
    # Note: We can still stream chunks even when wrapped in memory!
    for chunk in chain_with_memory.stream({"input": user_input}, config={"callbacks": [langfuse_handler], "session_id": "learning_session_1"}):
        print(chunk.content, end="", flush=True)
    print("\n")