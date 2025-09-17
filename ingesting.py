import os
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
        print("Ingesting...")
         # Load the text file from the same directory
        loader = TextLoader("mediumblog.txt", encoding="utf-8") # Assuming your file is named ingesting.txt
        document = loader.load()
        print("Splitting...")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)
        print(f"Created {len(texts)} chunks")

            # Initialize the Pinecone embedding model
        embeddings = PineconeEmbeddings(
            model="llama-text-embed-v2",
            pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        )

        print("Ingesting...")
        # Get the index name from the environment variable
        index_name = os.environ.get("INDEX_NAME")
        # Ingest the documents into your Pinecone index
        PineconeVectorStore.from_documents(
            texts, 
            embeddings, 
            index_name=index_name
        )

        print("Finish")



