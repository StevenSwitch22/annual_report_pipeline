from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 

def vector_store_hua(chunks):
    print("===================开始使用 Embedding Model=========================")
    embeddings = HuggingFaceEmbeddings(model_name="D:\\Models\\bge-m3")

    print("===================开始构建向量库====================")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="Huawei_bge_char",
        persist_directory="./chroma"
    )

    return vector_store

if __name__=="__main__":
    import chromadb
    client = chromadb.PersistentClient(path="./chroma")
    collections = client.list_collections()
    print("collections = ", collections)

    # delete collection
    client.delete_collection(name="Ali_bge")
    print("==============Collection 已删除==================")

    # show collection
    collections = client.list_collections()
    print("collections = ", collections)