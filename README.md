# 🔍 AromaSearch

> Encontre seu perfume. Descubra sua essência.

## Sobre o projeto

AromaSearch é um sistema de recomendação de perfumes masculinos baseado em busca vetorial semântica. A ideia surgiu de um problema real — sites de perfumaria tendem a recomendar o que é mais lucrativo para eles, não necessariamente o que mais combina com o usuário.

## Como funciona

O sistema usa técnicas de NLP e busca vetorial para encontrar perfumes similares:

- **sentence-transformers** gera embeddings semânticos das descrições
- **FAISS** indexa e busca os vetores por similaridade
- O usuário busca por descrição ou por nome de um perfume que já conhece

## Stack

- Python
- sentence-transformers
- FAISS
- Gradio
- Pandas
- Hugging Face Spaces

## Modos de busca

**Por descrição:** o usuário descreve o que procura em linguagem natural
> "fresco, amadeirado, para uso diário no trabalho"

**Por nome:** o usuário digita um perfume que já gosta e recebe similares
> "Sauvage" → retorna perfumes com perfil olfativo parecido

## Dataset

Dataset do Fragrantica com ~5000 perfumes masculinos, contendo notas de topo, coração e fundo, acordes principais e avaliações.

## Deploy

Aplicação disponível em: [AromaSearch no Hugging Face](https://huggingface.co/spaces/Carolineczeder/aromasearch)

## Autora

Carol Czeder — [GitHub](https://github.com/carolczeder) | [LinkedIn](https://linkedin.com/in/carolczeder)