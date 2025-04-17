import fitz 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests
import re
import json
import tiktoken
import os
import timeit

def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'[^\x20-\x7E]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.strip()
    return text

print("\n\n")

chroma_client = chromadb.Client()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_func = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

collection_name = "Collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_func) #For future reference



file_path = "hashes_game.txt"
file_ext = os.path.splitext(file_path)[1].lower()

full_text = ""

if file_ext == ".pdf":
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc, start=1):
        full_text += page.get_text()
elif file_ext == ".txt":
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
else:
    raise ValueError(f"Unsupported type: {file_ext}")


words = full_text.split()
chunks = []
chunk_size = 1000 #1000 too big...nvm
start_time = timeit.default_timer()
for i in range(0, len(words), chunk_size):
    chunk = words[i:i + chunk_size]
    chunks.append(' '.join(chunk))
    
cleaned_chunks = [clean_text(chunk) for chunk in chunks]
elapsed = timeit.default_timer() - start_time
print(f"Time elapsed for chunking and cleaning: {elapsed}")

start_time = timeit.default_timer()
embeddings = embedding_model.encode(chunks)
elapsed = timeit.default_timer() - start_time
print(f"Time elapsed for embedding: {elapsed}")

if collection.count() == 0:
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids) # test performance if no embeds/ 203 with
    print("ChromaDB papildinata")
else:
    print("ChromaDB esosi dati jau ir")
    
print("\n\n")
prompt = "What can you tell me about the given context?"


full_prompt = f"Context:\n{chunks[0]}\n\nQuestion: {prompt}\nAnswer:"
tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode(full_prompt)
print(f"Token count: {len(tokens)}")
print("\n\n")

start_time = timeit.default_timer()
try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": full_prompt}
    )
    response.raise_for_status()
    lines = response.text.strip().splitlines()
    responses = [json.loads(line) for line in lines if line.strip()]
    answer = "".join(r.get("response", "") for r in responses)
    print("VICTORY")
except Exception as e:
    print("SAD\n", e)
    answer = "Fail"
    
elapsed = timeit.default_timer() - start_time
print(f"Time elapsed for response: {elapsed}")

print("\n[OUTPUT]")
print(f"Q: {prompt}")
print(f"A: {answer}")


"""
Q: What can you tell me about the given context?
A: A fascinating question!

After analyzing the provided context, I can tell you that it appears to be a large collection of unique identifiers or hashes, likely representing tokens or IDs in a database or file system. Each identifier is 28 characters long and starts with "28beaec9476c007e" followed by a random sequence of letters and numbers.

The context does not provide any specific information about the purpose or structure of these identifiers, but based on their format, they might be used to uniquely identify records in a database, files, or objects in an object-oriented programming language.

If you'd like me to help with anything else, please feel free to ask!
"""