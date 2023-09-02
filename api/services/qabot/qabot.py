# from database.pgvector import PGVectorDB
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from datetime import datetime
import os
load_dotenv()


class QABotService:
    embeddings = OpenAIEmbeddings()
    CONNECTION_STRING = os.getenv("PGVECTOR_DB_CONNECTION_STRING")
    # pgvector_db = PGVectorDB("PGVECTOR_DB")

    @staticmethod
    def qaEvent(query):
        k = 15
        amount = 3
        llm = ChatOpenAI(model_name=os.getenv(
            "OPENAI_MODEL_NAME"), temperature=0.3)

        filter = {}
        filter['is_active'] = 'true'

        COLLECTION_NAME = os.getenv("QABOT_COLLECTION_NAME")

        store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=QABotService.CONNECTION_STRING,
            embedding_function=QABotService.embeddings,
        )

        docs = store.similarity_search(query, filter=filter, k=k)

        print(docs)

        query = query + "（根據用戶提供的描述，在我們的數據中推薦{event_amount}個活動，其內容要包含活動名稱、詳情、地點、開始和結束日期以及活動鏈接，輸出的內容只可以用英文或繁體中文！）".format(
            current_date=datetime(2023, 6, 1, 0, 0).strftime("%Y-%m-%d"), event_amount=amount)

        print(query)

        chain = load_qa_chain(llm=llm, chain_type="stuff")
        res = chain({"input_documents": docs, "question": query},
                    return_only_outputs=True)

        return res
