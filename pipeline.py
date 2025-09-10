from pdf_parsing import pdf_parse
from text_splitter import text_splitter
from ingestion import vector_store_ali, vector_store_hua
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def vector_store_ali_pipeline():
    documents = pdf_parse()
    chunks = text_splitter(documents)
    vectorStore = vector_store_ali(chunks)

def vector_store_hua(filePath):
    documents = pdf_parse(filePath)
    chunks = text_splitter(documents)
    vectorStore = vector_store_hua(chunks)

def vector_db_hua():
    embeddings = HuggingFaceEmbeddings(model_name="D:\\Models\\bge-m3")
    vector_db = Chroma(
        collection_name="Huawei_bge",
        embedding_function=embeddings,
        persist_directory="./chroma"
    )
    print("=======================开始使用向量库=====================")

    return vector_db

def vector_db_ali():
    embeddings = HuggingFaceEmbeddings(model_name="D:\\Models\\bge-m3")
    vector_db = Chroma(
        collection_name="Ali_bge",
        embedding_function=embeddings,
        persist_directory="./chroma"
    )
    print("=======================开始使用向量库=====================")

    return vector_db

if __name__ == "__main__":
    file_path = [
        "pdf\阿里巴巴集团控股有限公司2025财务年度报告（繁体中文版）.pdf",
        "pdf\华为年报_2024_cn.pdf"
    ]
    vector_store_ali_pipeline()
    # vector_store_hua(file_path[1])

    # vector_store_pipeline()
    # print("================向量库构建完成==================")