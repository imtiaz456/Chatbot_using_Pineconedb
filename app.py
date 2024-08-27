import os
import openai
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader  # to load documents like pdf etc
from langchain.text_splitter import RecursiveCharacterTextSplitter  # to split text into chunks
from langchain.embeddings.openai import OpenAIEmbeddings  # to convert these chunks into vector form
from langchain.vectorstores import Pinecone  # here we need a db to store these vectors
from langchain.llms import OpenAI  # then we will need to use llm models
from langchain.vectorstores import Cassandra

from dotenv import load_dotenv
import streamlit as st
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

load_dotenv()

# Streamlit UI for PDF upload
st.title("Document Q&A")
pdf_path = st.file_uploader("Upload a PDF file", type=["pdf"])
raw_text = ""
if pdf_path:
    pdfreader = PyPDFDirectoryLoader(pdf_path)
    for i, page in enumerate(pdfreader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

# Initialize Cassandra connection
cloud_config = {
    'secure_connect_bundle': os.getenv("SECURE_CONNECT_BUNDLE_PATH")  # Path to secure connect bundle
}
auth_provider = PlainTextAuthProvider(os.getenv("ASTRA_DB_CLIENT_ID"), os.getenv("ASTRA_DB_CLIENT_SECRET"))
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

# Initialize LLM and Embeddings
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Set up Cassandra Vector Store
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",  # Replace hyphen with an underscore
    session=session,
    keyspace=os.getenv("ASTRA_DB_KEYSPACE"),
)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)
astra_vector_store.add_texts(texts[:50])

# Streamlit UI for Q&A
first_question = True

while True:
    if first_question:
        query_text = st.text_input("Enter your question (or type 'quit' to exit):").strip()
    else:
        query_text = st.text_input("What's your next question (or type 'quit' to exit):").strip()
    if query_text.lower() == "quit":
        break
    if query_text == "":
        continue
    first_question = False
    st.write(f"\nQuestion: \"{query_text}\"")
    answer = astra_vector_store.query(query_text, llm=llm).strip()
    st.write(f"Answer: \"{answer}\"\n")
    st.write("FIRST DOCUMENTS BY RELEVANCE:")
    # Uncomment and modify below lines if you want to display documents by relevance
    # for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
    #     st.write(f"    [{score:0.4f}] \"{doc.page_content[:84]}...\"")
