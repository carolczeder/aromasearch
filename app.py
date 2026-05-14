import gradio as gr
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

def buscar_por_descricao(texto):
    vetor = modelo.encode([texto])
    distancias, indices = index.search(np.array(vetor), 5)
    resultados = df.iloc[indices[0]]
    resposta = ""
    for _, row in resultados.iterrows():
        resposta += f"**{row['Perfume']}** — {row['Brand']}\n"
        resposta += f"Acordes: {row['mainaccord1']} | {row['mainaccord2']} | {row['mainaccord3']}\n\n"
    return resposta

def buscar_por_nome(nome):
    matches = df[df['Perfume'].str.contains(nome, case=False, na=False)]
    if len(matches) == 0:
        return "Perfume não encontrado. Tente outro nome."
    idx = matches.index[0]
    vetor = embeddings[idx].reshape(1, -1)
    distancias, indices = index.search(np.array(vetor), 6)
    resultados = df.iloc[indices[0][1:]]
    resposta = ""
    for _, row in resultados.iterrows():
        resposta += f"**{row['Perfume']}** — {row['Brand']}\n"
        resposta += f"Acordes: {row['mainaccord1']} | {row['mainaccord2']} | {row['mainaccord3']}\n\n"
    return resposta

css = """
body { background-color: #0a0f1e !important; }
.gradio-container { background-color: #0a0f1e !important; max-width: 750px !important; margin: auto !important; }
button.primary { width: 200px !important; margin: 10px auto !important; display: block !important; background: #1a6b8a !important; border: none !important; border-radius: 8px !important; color: white !important; }
.image-container { border: none !important; background: transparent !important; box-shadow: none !important; }
.image-container button { display: none !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="AromaSearch", css=css) as app:
    gr.Image('logo.png', show_label=False, height=180, container=False)
    gr.Markdown("<center>Encontre seu perfume. Descubra sua essência.</center>")
    gr.Markdown("---")

    with gr.Tab("🔍 Buscar por descrição"):
        gr.Markdown("Descreva o tipo de perfume que você procura")
        texto = gr.Textbox(
            label="Descrição",
            placeholder="Ex: fresco, amadeirado, para uso diário no trabalho"
        )
        btn1 = gr.Button("Buscar perfumes", variant="primary")
        resultado1 = gr.Markdown()
        btn1.click(buscar_por_descricao, inputs=texto, outputs=resultado1)

    with gr.Tab("🎯 Buscar por nome"):
        gr.Markdown("Digite um perfume que você já gosta e encontre similares")
        nome = gr.Textbox(
            label="Nome do perfume",
            placeholder="Ex: Sauvage"
        )
        btn2 = gr.Button("Buscar similares", variant="primary")
        resultado2 = gr.Markdown()
        btn2.click(buscar_por_nome, inputs=nome, outputs=resultado2)

    gr.Markdown("<center><small>AromaSearch — recomendações sem viés comercial 💙</small></center>")

app.launch()