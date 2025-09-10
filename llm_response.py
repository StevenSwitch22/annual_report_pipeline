from retrieval import retrieval
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI


query = "这篇文章在讲什么？"
results = retrieval(query)

template = """
忠实于一个原则：回答用户问题必须要基于下面的背景资料回答，不能自己胡乱回答！如果背景资料回答不了用户的问题，就说“背景资料无法回答你的问题”
<背景资料>
{context}
</背景资料>
用户问题：
{query}
"""
prompt = PromptTemplate.from_template(template=template)

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