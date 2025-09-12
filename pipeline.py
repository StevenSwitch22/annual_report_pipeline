from pdf_parsing import pdf_parse
from text_splitter import text_splitter
from ingestion import vector_store_hua
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

def vector_store_hua_pipeline():
    documents = pdf_parse()
    chunks = text_splitter(documents)
    vectorStore = vector_store_hua(chunks)

def vector_db_hua():
    embeddings = HuggingFaceEmbeddings(model_name="D:\\Models\\bge-m3")
    vector_db = Chroma(
        collection_name="Huawei_bge_char",
        embedding_function=embeddings,
        persist_directory="./chroma"
    )
    print("=======================开始使用向量库=====================")

    return vector_db 

if __name__ == "__main__": 
    vector_store_hua_pipeline() 
    print("================向量库构建完成==================")