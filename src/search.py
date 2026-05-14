import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Carrega os arquivos
df = pd.read_csv('data/perfumes_masculinos.csv')
index = faiss.read_index('data/perfumes.index')

with open('data/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

modelo = SentenceTransformer('all-MiniLM-L6-v2')

def buscar_por_descricao(texto, k=5):
    vetor = modelo.encode([texto])
    distancias, indices = index.search(np.array(vetor), k)
    resultados = df.iloc[indices[0]][['Perfume', 'Brand', 'mainaccord1', 'mainaccord2', 'Rating Value']]
    return resultados

def buscar_por_nome(nome, k=6):
    matches = df[df['Perfume'].str.contains(nome, case=False, na=False)]
    if len(matches) == 0:
        return None
    idx = matches.index[0]
    vetor = embeddings[idx].reshape(1, -1)
    distancias, indices = index.search(np.array(vetor), k)
    resultados = df.iloc[indices[0][1:]][['Perfume', 'Brand', 'mainaccord1', 'mainaccord2', 'Rating Value']]
    return resultados

# Teste rápido
if __name__ == '__main__':
    print("Teste por descrição:")
    print(buscar_por_descricao("amadeirado fresco para uso noturno"))
    print("\nTeste por nome:")
    print(buscar_por_nome("sauvage"))