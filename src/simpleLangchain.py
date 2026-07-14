


import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY: {GROQ_API_KEY}")

def initialize_groq():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")
def initialize_model():
    # 1. Initialize Groq with a highly capable free model
    model = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.7,
        api_key=GROQ_API_KEY
    )
    return model
def create_prompt():
    # 2. Define the template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a witty assistant. Give a 1-sentence explanation of the concept provided."),
        ("user", "{concept}")
    ])
    return prompt
def create_simple_parser():
    # 3. Output parser
    parser = StrOutputParser()
    return parser
def create_chain(prompt, model, parser):
    # 4. Chain them up via LCEL
    chain = prompt | model | parser
    return chain
def invoke_chain(chain, concept):
    # 5. Invoke the chain with a concept
    response = chain.invoke({"concept": concept})
    return response

model = initialize_model()
prompt = create_prompt()
parser = create_simple_parser()
chain = create_chain(prompt, model, parser)
response = invoke_chain(chain, "LCEL")

print("Streaming response: ", end="")
for chunk in chain.stream({"concept": "API integration"}):
    # Print each piece immediately without a newline, flushing the buffer
    sys.stdout.write(chunk)
    sys.stdout.flush()

print() # New line at the end

print(response)

