# from database.pgvector import PGVectorDB
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os
load_dotenv()


class QABotService:
    embeddings = OpenAIEmbeddings()
    CONNECTION_STRING = os.getenv("PGVECTOR_DB_CONNECTION_STRING")
    # pgvector_db = PGVectorDB("PGVECTOR_DB")

    @staticmethod
    def qaEvent(query, metadata):
        filter = {}

        COLLECTION_NAME = os.getenv("QABOT_COLLECTION_NAME")

        store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=QABotService.CONNECTION_STRING,
            embedding_function=QABotService.embeddings,
        )

        docs = store.similarity_search(query, filter=filter, k=10)

        print(docs)

        chain = load_qa_chain(ChatOpenAI(model_name=os.getenv(
            "OPENAI_MODEL_NAME"), temperature=0.7), chain_type="stuff")
        res = chain({"input_documents": docs, "question": query},
                    return_only_outputs=True)

        return res
