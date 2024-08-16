import os
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory




class Ragchain:
    _instances = {}

    def __new__(cls, user_id, use_case_id, embed_model, llm):
        if user_id not in cls._instances:
            cls._instances[user_id] = super().__new__(cls)
            cls._instances[user_id].init(user_id, use_case_id, embed_model, llm)
        return cls._instances[user_id]
    
    def init(self, user_id, use_case_id, embed_model, llm):
        print('X_INIT_TRIGGERED_X')
        self.user_id = user_id
        self.use_case_id = use_case_id
        self.embed_model = embed_model
        self.llm = llm
        self.conditions = []
        self.store = {}


    def init_rag_chain(self, index_name: str):
        # Initialize RAG chain
        self.pc_v = PineconeVectorStore(index_name=index_name, pinecone_api_key=os.environ.get('PINECONE_API_KEY'), embedding=self.embed_model)
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

        self.history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, self.contextualize_q_prompt)

        self.qa_system_prompt = """
        You are an assistant for question-answering tasks, when you give the answer please dont tell you are an assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise and please just give the answer nothing else{context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

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