import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

print("Carregando dataset...")
df = pd.read_csv('data/perfumes_masculinos.csv')

print("Carregando modelo de embeddings...")
modelo = SentenceTransformer('all-MiniLM-L6-v2')

print("Gerando embeddings... (pode demorar alguns minutos)")
embeddings = modelo.encode(df['descricao'].tolist(), show_progress_bar=True)

print("Criando índice FAISS...")
dimensao = embeddings.shape[1]
index = faiss.IndexFlatL2(dimensao)
index.add(np.array(embeddings))

print("Salvando arquivos...")
faiss.write_index(index, 'data/perfumes.index')

with open('data/embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print(f"Pronto! {index.ntotal} perfumes indexados.")