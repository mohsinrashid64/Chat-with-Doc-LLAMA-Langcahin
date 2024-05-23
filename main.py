# Importing Libraries
import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File
from fastapi import HTTPException
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from _utils.documents import get_chunks

from llama_cpp import Llama




load_dotenv() # Loading Enviroment Variables
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY')) # Setting Pine Cone API Key
# embed_model = LlamaCppEmbeddings(model_path='models/llama-2-7b-chat.gguf.q4_0.bin',verbose=False) 



# embed_model = HuggingFaceEmbeddings(
#     # model_name = "sentence-transformers/all-mpnet-base-v2",
#     model_name = 'hkunlp/instructor-large',
#     model_kwargs = {'device': 'cpu'},
#     encode_kwargs = {'normalize_embeddings': False},

# )

llm_embed = Llama(model_path = "models/llama-2-7b-chat.gguf.q4_0.bin",verbose=False,embedding=True)


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = -1  
n_batch = 2048  

llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.gguf.q4_0.bin",
    # n_gpu_layers=n_gpu_layers,
    # n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=False,  
)



app = FastAPI() # Creating An Intance of Fast API


@app.get("/")
def read_root():
    return {'response': "API RUNNING"}


@app.post("/add_embeddings_on_pinecone/")
async def add_embeddings_on_pinecone(use_case_id:str, files: List[UploadFile] = File(...)):

    try:
        chunks, files_not_supported = await get_chunks(files)
        pc_v =  PineconeVectorStore(index_name=use_case_id,pinecone_api_key=os.environ.get('PINECONE_API_KEY'),embedding=embed_model)
        pc_v.add_documents(chunks)

        print(files_not_supported)
        return {"response":'Embedding Added Succesfully'}
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Yo Man Internal Server Error Man")
    


@app.post("/query")
async def query(use_case_id:str, query_:str):
    try:
        pc_v =  PineconeVectorStore(index_name=use_case_id,pinecone_api_key=os.environ.get('PINECONE_API_KEY'),embedding=embed_model)
        retriever=pc_v.as_retriever()
        chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        response = chain.run(query_)
        return {'response': response }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=e)



@app.get("/delete_file")
async def delete_file(use_case_id:str, file_name:str):
    try:
        index = pc.Index(use_case_id, pool_threads = 32)
        doc_ids = sum([ids for ids in index.list()], [])
        index_data  = index.fetch(doc_ids)
        doc_ids_to_delete = [doc_id for doc_id in doc_ids if index_data.vectors[doc_id].metadata['file_name'] == file_name]
        index.delete(doc_ids_to_delete)
        return {'response': f"File '{file_name}' sucessfully deleted"}


    except Exception as e:
        raise HTTPException(status_code=404, detail="!!!FILE NAME NOT FOUND!!!")