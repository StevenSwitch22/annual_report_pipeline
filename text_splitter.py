from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

def text_splitter(documents):
    text_splitter = CharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separator='\n'
        )
    # text_splitter = RecursiveCharacterTextSplitter(
    #     separators=["\n\n", "\n", "。", "，", " ", ""],
    #     chunk_size=300,
    #     chunk_overlap=50,
    # )
    chunks = text_splitter.split_documents(documents=documents)
    print("len chunks = ", len(chunks))

    return chunks