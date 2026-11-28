from langchain_chroma import Chroma
from config import embeddings

vectorstore = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings,
    collection_name="research_documents"
)

retriver = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)