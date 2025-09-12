from langchain_community.document_loaders import PyPDFLoader

def pdf_parse():
    loader = PyPDFLoader(file_path="pdf\华为年报_2024_cn.pdf")
    documents = loader.load()
    print("len documents = ", len(documents))
    
    return documents 