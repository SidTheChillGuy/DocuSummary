allset = True

print("Starting module 'json2embed' with Imports")

try:
    import os, json
except:
    print("Error importing os or json module.")
    allset = False
try:
    from sentence_transformers import SentenceTransformer
except:
    print("Error importing sentence_transformers.")
    allset = False
try:
    import chromadb
except:
    print("Error importing chromadb.")
    allset = False
try:
    import torch
except:
    print("Error importing torch.")
    allset = False
try:
    from transformers import BitsAndBytesConfig
except:
    print("Error accessing transformers module.")
    allset = False

if(allset == False):
    raise ImportError("Fix ImportError.")

print("Imports completed. Loading models.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",model_kwargs={"attn_implementation": "flash_attention_2", "device_map": device, "torch_dtype":torch.float16, "quantization_config":quantization_config},tokenizer_kwargs={"padding_side": "left"})

files = [f for f in os.listdir("tmp") if f.endswith(".json")]

chroma_client = chromadb.PersistentClient(path="db/chroma_store")
collection = chroma_client.get_or_create_collection("my_docs")

for i in files:
    print("Loading Document:", i)
    with open(f"tmp/{i}", 'r', encoding="utf-8") as f:
        text_outputs = json.load(f)
    chunks = []
    with torch.no_grad():    
                for doc_id, page_texts in text_outputs.items():
                    for i, page in enumerate(page_texts):
                        split_chunks = page.split("\n\n")  # You can do smarter chunking here
                        for chunk in split_chunks:
                            chunks.append(chunk)
                        torch.cuda.empty_cache()
                embeddings = embedder.encode(chunks)
                collection.add(
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=[{"source": f"document-{list(text_outputs.keys())[0]}-page-{k}"} for k in range(len(chunks))],
                    ids=[f"doc_{list(text_outputs.keys())[0]}_page_{k}" for k in range(len(chunks))]
                )

if len(os.listdir("tmp"))>0:
    print("Cleaning tempdir...")
    for i in os.listdir("tmp"):    
        os.remove(f"tmp/{i}")

print("Module 'json2embed' completed successfully.")
print("All the values have been updated in the storage base. Function completed you may leave now.")