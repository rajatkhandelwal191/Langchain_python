import asyncio
import os
import ssl
from typing import Any, Dict, List


import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma # local vectorstore like faiss
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
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
        batch_size=50,
        retry_min_seconds=10
    )


tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


