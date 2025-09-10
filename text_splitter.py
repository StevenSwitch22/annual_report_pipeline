from langchain_text_splitters import CharacterTextSplitter 

def text_splitter(documents):
    text_splitter = CharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=0,
            separator='\n'
        )
    chunks = text_splitter.split_documents(documents=documents)
    print("len chunks = ", len(chunks))

    return chunks