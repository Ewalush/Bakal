import fitz 
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests
import re
import json
import tiktoken
import os

#predefinesana
chunk_size = 1000
prompt = "Kā pieteikties brīvās izvēles kursiem?"

# Texta satirisana
def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Failu mape
rootFolder = "Faili"

# ChromaDB configuresana un palaisana
embeddingFunction = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
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
    print(f"Apstradats fails '{fileName}': {total_words} vardi pec satirisanas.")

    # Dalu sadalisana
    chunks = []
    for i in range(0, total_words, chunk_size):
        chunk_tokens = word_tokens[i:i + chunk_size]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
    print(f"Fails sadalits {len(chunks)} dalas pa {chunk_size} vardiem.")

    # Datubazes papildinasana ar vardu dalam
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{fileName}_chunk_{idx}"
        collection.add(documents=[chunk], ids=[chunk_id])


results = collection.query(query_texts=[prompt], n_results=3)
top_results = results['documents']

fullPrompt = f"Context:\n{top_results[0]}\n\nQuestion: {prompt}\nAnswer:"
tokenizer = tiktoken.get_encoding("cl100k_base")  # Approx match for LLaMA
tokens = tokenizer.encode(fullPrompt)
print(f"Token count: {len(tokens)}")
print("\n\n")

try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen3", "prompt": fullPrompt}
    )
    response.raise_for_status()
    lines = response.text.strip().splitlines()
    responses = [json.loads(line) for line in lines if line.strip()]
    answer = "".join(r.get("response", "") for r in responses)
except Exception as e:
    answer = "Fail"

print("\n[OUTPUT]")
print(f"Q: {prompt}")
print(f"A: {answer}")

