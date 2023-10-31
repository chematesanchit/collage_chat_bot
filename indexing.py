import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

# os.environ["OPENAI_API_KEY"] = "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"

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# Set persist directory
persist_directory = 'db'

loader = PyPDFLoader(r'C:\Users\sanchit\Downloads\GPT-Kotak-main\docs\data.pdf')

pages = loader.load_and_split()
chunks = pages


embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=8)

# Split documents and generate embeddings
Local_docs_split = text_splitter.split_documents(loader)


# Create Chroma instances and persist embeddings
LocalDB = Chroma.from_documents(Local_docs_split, embeddings, persist_directory=os.path.join(persist_directory, 'Local'))
LocalDB.persist()


