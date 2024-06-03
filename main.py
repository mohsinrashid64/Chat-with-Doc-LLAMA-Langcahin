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
    

class Singleton:
    _instances = {}

    def __new__(cls, user_id, use_case_id):
        if user_id not in cls._instances:
            cls._instances[user_id] = super().__new__(cls)
            cls._instances[user_id].init(user_id, use_case_id)
        return cls._instances[user_id]
    
    def init(self, user_id, use_case_id):
        print('X_INIT_TRIGGERED_X')
        self.user_id = user_id
        self.use_case_id = use_case_id
        self.conditions = []
        self.store = {}


    def init_rag_chain(self, index_name: str):
        # Initialize RAG chain
        self.pc_v = PineconeVectorStore(index_name=index_name, pinecone_api_key=os.environ.get('PINECONE_API_KEY'), embedding=embed_model)
        self.retriever = self.pc_v.as_retriever()

        # Initialize prompts and chains
        self.contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(llm, self.retriever, self.contextualize_q_prompt)

        self.qa_system_prompt = """
        You are an assistant for question-answering tasks, when you give the answer please dont tell you are an assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise and please just give the answer nothing else{context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

        # Statefully manage chat history
        self.store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_rag_chain(self, index_name: str):
        if not hasattr(self, 'conversational_rag_chain'):
            self.init_rag_chain(index_name)
        return self.conversational_rag_chain



def wrapper(user_id: str, use_case_id: str):
    singleton = Singleton(user_id,use_case_id)
    return singleton

@app.get("/chat")
def use_singleton(question: str,user_id: str, use_case_id: str, singleton: Singleton = Depends(wrapper)):
    # res = patient.get_details()
    res = singleton.get_rag_chain(use_case_id).invoke({"input": f'{question}'}, {'configurable': {f'session_id': f'{user_id}'}})

    return {"message": res}
