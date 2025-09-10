from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def vector_store_ali(chunks):
    print("===================开始使用 Embedding Model=========================")
    embeddings = HuggingFaceEmbeddings(model_name="D:\\Models\\bge-m3")

    print("===================开始构建向量库====================")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="Ali_bge",
        persist_directory="./chroma"
    )

    return vector_store

def vector_store_hua(chunks):
    print("===================开始使用 Embedding Model=========================")
    embeddings = HuggingFaceEmbeddings(model_name="D:\\Models\\bge-m3")

    print("===================开始构建向量库====================")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="Huawei_bge",
        persist_directory="./chroma"
    )

    return vector_store