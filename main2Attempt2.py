import fitz 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests
import re
import json
import tiktoken
import os

#predefinesana
chunk_size = 1000
prompt = "Can you give me a technical report on DeepSeekv3?"

# Texta satirisana
def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'[^\x20-\x7E]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.strip()
    return text

# Failu mape
rootFolder = "Faili"

# ChromaDB configuresana un palaisana
embeddingFunction = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
client = chromadb.Client()
collection = client.get_or_create_collection(name="Collection", embedding_function=embeddingFunction)

# Katra faila procesesana
for fileName in os.listdir(rootFolder):
    filePath = os.path.join(rootFolder, fileName)
    
    if fileName.lower().endswith(".pdf"):
        try:
            doc = fitz.open(filePath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Failed to read PDF {fileName}: {e}")
            continue
    elif fileName.lower().endswith(".txt"):
        try:
            # Read text file content
            with open(filePath, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Failed to read TXT {fileName}: {e}")
            continue
    else:
        continue

    # Textra satirisana
    cleaned_text = clean_text(text)
    # Vardu sadalisana
    word_tokens = cleaned_text.split()
    total_words = len(word_tokens)
    print(f"Apstradati '{fileName}': {total_words} vardi pec satirisanas.")

    # Dalu sadalisana
    chunks = []
    for i in range(0, total_words, chunk_size):
        chunk_tokens = word_tokens[i:i + chunk_size]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
    print(f"Faili sadaliti {len(chunks)} dalas pa {chunk_size} varadiem.")

    # Datubazes papildinasana ar vardu dalam
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{fileName}_chunk_{idx}"
        collection.add(documents=[chunk], ids=[chunk_id])


results = collection.query(query_texts=[prompt], n_results=3)
top_results = results['documents'][0]

#print(top_results[0])

#

full_prompt = f"Context:\n{top_results[0]}\n\nQuestion: {prompt}\nAnswer:"
tokenizer = tiktoken.get_encoding("cl100k_base")  # Approx match for LLaMA
tokens = tokenizer.encode(full_prompt)
print(f"Token count: {len(tokens)}")
print("\n\n")
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


"""
    https://docs.trychroma.com/docs/embeddings/embedding-functions
    
    https://docs.trychroma.com/docs/overview/getting-started
    
    Model
    architecture        llama
    parameters          8.0B
    context length      8192
    embedding length    4096
    quantization        Q4_0

  Capabilities
    completion

  Parameters
    num_keep    24
    stop        "<|start_header_id|>"
    stop        "<|end_header_id|>"
    stop        "<|eot_id|>"

  License
    META LLAMA 3 COMMUNITY LICENSE AGREEMENT
    Meta Llama 3 Version Release Date: April 18, 2024
    
    
    https://www.venta.lv/augstskola/parskati-un-zinojumi
    
"""