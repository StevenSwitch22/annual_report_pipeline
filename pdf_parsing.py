from langchain_community.document_loaders import PyPDFLoader

def pdf_parse():
    loader = PyPDFLoader(file_path="pdf\阿里巴巴集团控股有限公司2025财务年度报告（繁体中文版）.pdf")
    documents = loader.load()
    print("len documents = ", len(documents))
    
    return documents 