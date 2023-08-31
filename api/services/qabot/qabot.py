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
    def qaEvent(query, metadata):
        filter = {}

        k = 15

        COLLECTION_NAME = os.getenv("QABOT_COLLECTION_NAME")

        store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=QABotService.CONNECTION_STRING,
            embedding_function=QABotService.embeddings,
        )

        docs = store.similarity_search(query, filter=filter, k=k)

        # docs = store.as_retriever(
        #     search_kwargs={'filter': filter, 'k': 10,
        #                    'fetch_k': 20}
        # )

        print(docs)

        query = query + "（根據用戶提供的描述，給用戶推薦兩個有效的活動，有效的活動必須是還未結束的活動（今天是{current_date}），推薦的兩個活動內容要包含活動地點以及活動開始和結束日期，輸出的內容只可以用英文或繁體中文！）".format(
            current_date=datetime(2023, 6, 1, 0, 0).strftime("%Y-%m-%d"), k=k)

        print(query)

        chain = load_qa_chain(ChatOpenAI(model_name=os.getenv(
            "OPENAI_MODEL_NAME"), temperature=0.3), chain_type="stuff")
        res = chain({"input_documents": docs, "question": query},
                    return_only_outputs=True)

        return res
