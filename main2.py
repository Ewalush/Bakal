import fitz 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests
import re
import json
import tiktoken
import os

#remove illegal chars
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



file_path = "2412_19437v2.pdf"
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

#print(full_text)
print(len(full_text))
print("\n\n")

words = full_text.split()
chunks = []
chunk_size = 1000 #1000 too big...nvm

for i in range(0, len(words), chunk_size):
    chunk = words[i:i + chunk_size]
    chunks.append(' '.join(chunk))
    
cleaned_chunks = [clean_text(chunk) for chunk in chunks]

print(chunks[0])
print("\n\n")

embeddings = embedding_model.encode(chunks)
print("Embeddings \n", embeddings[0])

if collection.count() == 0:
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, embeddings=embeddings) # test performance if no embeds
    print("ChromaDB papildinata")
else:
    print("ChromaDB esosi dati jau ir")
    
print("\n\n")

prompt = "Can you give me a technical report on DeepSeekv3?"
results = collection.query(query_texts=[prompt], n_results=3) # Top 3 results to promt
top_results = results['documents'][0]

#print(top_results[0])

######################################################################################## STEP 1 DONE CAN CUT OFF HERE POSSIBLE

full_prompt = f"Context:\n{top_results}\n\nQuestion: {prompt}\nAnswer:"
tokenizer = tiktoken.get_encoding("cl100k_base")  # Approx match for LLaMA
tokens = tokenizer.encode(full_prompt)
print(f"Token count: {len(tokens)}")
print("\n\n")

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

print("\n[OUTPUT]")
print(f"Q: {prompt}")
print(f"A: {answer}")

#https://www.venta.lv/augstskola/parskati-un-zinojumi