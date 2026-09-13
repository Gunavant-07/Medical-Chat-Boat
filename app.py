from flask import Flask, request, jsonify, render_template
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("Openai_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_huggingface_embeddings()

index_name = "medical-chat-boat"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs={"k": 3})


llm = ChatOpenAI(
    model="gpt-5.6-luna",
    api_key=os.getenv("Openai_API_KEY"),
    temperature=0.4,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User Input: {input}")
    response = rag_chain.invoke({"input": input})
    print(f"Response: {response['answer']}")
    return str(response['answer'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7008, debug=True)
    