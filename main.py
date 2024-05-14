# Importing Libraries
import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File
from fastapi import HTTPException
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler



import os
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.document_loaders import TextLoader

from pinecone import Pinecone

from langchain.text_splitter import RecursiveCharacterTextSplitter



load_dotenv() # Loading Enviroment Variables
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY')) # Setting Pine Cone API Key
embed_model = LlamaCppEmbeddings(model_path='models/llama-2-7b-chat.gguf.q4_0.bin',verbose=False) # 


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = -1  
n_batch = 2048  


llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=False,  
)



app = FastAPI() # Creating An Intance of Fast API

@app.get("/")
def read_root():
    return {'response': "API RUNNING"}


@app.post("/add_embeddings_on_pinecone/")
def add_embeddings_on_pinecone( files: List[UploadFile] = File(...)):
    # embeddings = LlamaCppEmbeddings(model_path='models/llama-2-7b-chat.gguf.q4_0.bin',verbose=False)
    # print('files')
    # loader = TextLoader(files[0])
    # doc = loader.load()

    # print(doc)
    return {'response':"YOMAN"}

