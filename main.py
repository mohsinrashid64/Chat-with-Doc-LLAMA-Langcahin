# Importing Libraries
import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi import HTTPException
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from _utils.documents import get_chunks
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Ollama
from _utils.classes.rag_chain import Ragchain




load_dotenv() 
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

embed_model = OllamaEmbeddings(
    model="llama3",
    base_url='http://127.0.0.1:11434'
)


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = -1  
n_batch = 2048  

llm = Ollama(model="llama3", base_url="http://127.0.0.1:11434")


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
    
@app.get('/chat')
def chat(user_id:str, use_case_id:str, question:str):
    try:
        rag_chain = Ragchain(user_id, use_case_id, embed_model, llm)
        response = rag_chain.get_rag_chain(use_case_id).invoke({"input": f'{question}'}, {'configurable': {f'session_id': f'{user_id}'}})
        # response = str(response)
        print('RESPONSE_X',response['answer'])
        return {'response': response['answer']}
    
    except Exception as err:
        raise HTTPException(status_code=500, detail="Error that is Internal Server Hah")
