import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Import the Langfuse Callback Handler
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# 2. Set your Langfuse API Keys (or set them as system environment variables)
#os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-your-public-key"
#os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-your-secret-key"
#os.environ["LANGFUSE_BASE_URL"] = "https://cloud.langfuse.com" # Default for US/Global Cloud

# 3. Initialize the handler
langfuse_handler = CallbackHandler()

# 4. Standard free Groq Chain Setup
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, api_key=GROQ_API_KEY)
prompt = ChatPromptTemplate.from_template("Explain the difference between open-source and closed-source software in 2 sentences.")
chain = prompt | model | StrOutputParser()

# 5. Execute while passing the handler inside the config dictionary!
print("Streaming response while tracking with Langfuse:\n")

# Notice we pass the handler inside the config block: config={"callbacks": [langfuse_handler]}
for chunk in chain.stream({"topic": "software"}, config={"callbacks": [langfuse_handler]}):
    print(chunk, end="", flush=True)

print("\n\nExecution complete. Check your Langfuse Cloud Dashboard!")