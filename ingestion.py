import asyncio
import os
import ssl
from typing import Any, Dict, List


import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma # local vectorstore like faiss
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap


from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = PineconeEmbeddings(
        model="llama-text-embed-v2",
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        # batch_size=50,
        # retry_min_seconds=10
    )


tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestrate the entire process"""
    log_header("Documentation Ingestion Pipeline")
    log_info(
    "🔍 TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/",
    Colors.PURPLE,
    )

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "extract_depth": "advanced",
            "instructions": "Documentatin relevant to ai agents",
            "max_depth": 1,
        }
    )

    # all_docs = res["results"]
    all_docs = [Document(page_content=result['raw_content'], metadata={"source":result['url']}) for result in res['result']]
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site")


if __name__ == "__main__":
    asyncio.run(main())