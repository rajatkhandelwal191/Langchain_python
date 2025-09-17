import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain import hub
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain




load_dotenv()

if __name__ == "__main__":
    print("retrieving")

    embeddings = PineconeEmbeddings(
        model="llama-text-embed-v2",
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    query = "what is Pinecone in machine learning?"

    chain = PromptTemplate.from_template(template=query) | llm

    # result = chain.invoke(input={})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)


    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    print(result)







 