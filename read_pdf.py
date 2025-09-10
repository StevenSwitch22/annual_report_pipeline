from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

def store_knowledge():
    loader = PyPDFLoader(file_path="pdf/阿里巴巴集团控股有限公司2025财务年度报告（繁体中文版）.pdf")
    documents = loader.load()
    print("len documents = ", len(documents))

    text_splitter = CharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
        separator='\n'
    )
    chunks = text_splitter.split_documents(documents=documents)
    print("len chunks = ", len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name="D:\\Models\\all-mpnet-base-v2")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="weixin",
        persist_directory="./chroma"
    )
    return vector_store

vector_store = store_knowledge()

template = """
忠实于一个原则：回答用户问题必须要基于下面的背景资料回答，不能自己胡乱回答！如果背景资料回答不了用户的问题，就说“背景资料无法回答你的问题”
<背景资料>
{context}
</背景资料>
用户问题：
{query}
"""
prompt = PromptTemplate.from_template(template=template)


def search_documents(query):
    results = vector_store.similarity_search(query=query, k=2) 
    search_knowledge = "\n\n".join([res.page_content for res in results]) 
    return search_knowledge

query = "这篇文章在讲什么？"
results = search_documents(query) 
formatted_prompt = prompt.format(context=results, query=query)
print("format = ", formatted_prompt)

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3-0324",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY5ZjM0MWY0LTRmNzUtNDk5OS1iNTgzLTk4MDk3ZDZlMjE3NSIsInNjb3BlIjoiaWVfbW9kZWwiLCJjbGllbnRJZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMCJ9.4vmRCNGKBE8PtLsuLSkp8r3rlZF1cwD7Bg8KB2fLVs4",
    base_url="https://api.gmi-serving.com/v1",
    temperature=0.1,
    top_p=0.1
)
messages = [
    ("human", formatted_prompt)
]
response = llm.invoke(messages)
print("response = ", response.content)
# print(f"len = {len(pages)}\n")  # len = 727  Windsurf Ctrl + D 多光标选择 VS Code Ctrl+Shift+L 