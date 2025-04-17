import fitz  
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
import requests

chroma_client = chromadb.Client()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

print("China")