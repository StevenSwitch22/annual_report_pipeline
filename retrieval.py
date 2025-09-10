from pipeline import vector_db_hua

def retrieval(query):
    vector_store = vector_db_hua()

    results = vector_store.similarity_search(query=query, k=4) 
    print("documents = ", results)
    print("=====================================")
    print(f"page {results[0].metadata['page_label']}: \n", results[0].page_content)
    print("=====================================")
    print(f"page {results[1].metadata['page_label']}: \n", results[1].page_content)
    print("=====================================")
    print(f"page {results[2].metadata['page_label']}: \n", results[2].page_content)
    print("=====================================")
    print(f"page {results[3].metadata['page_label']}: \n", results[3].page_content)
    search_knowledge = "\n\n".join([res.page_content for res in results]) 
    return search_knowledge

results = retrieval("独立审计师是干什么的？")
# print(f"results = \n{results}\n")