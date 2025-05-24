import fitz  
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
import requests
import json

prompt = "Can you give me a technical report on DeepSeekv3?"

try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt}
    )
    response.raise_for_status()
    lines = response.text.strip().splitlines()
    responses = [json.loads(line) for line in lines if line.strip()]
    answer = "".join(r.get("response", "") for r in responses)
    print("VICTORY")
except Exception as e:
    print("SAD\n", e)
    answer = "Fail"
    
print("\n[OUTPUT]")
print(f"Q: {prompt}")
print(f"A: {answer}")
