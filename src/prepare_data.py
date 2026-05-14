import pandas as pd

df = pd.read_csv('data/fra_cleaned.csv', encoding='latin-1', sep=';', on_bad_lines='skip')

# Filtra só masculinos
df = df[df['Gender'] == 'men'].copy()

# Preenche valores nulos com string vazia
df = df.fillna('')
df = df.astype(str)

# Cria coluna de descrição
df['descricao'] = (
    'Perfume: ' + df['Perfume'] + '. ' +
    'Marca: ' + df['Brand'] + '. ' +
    'Notas de topo: ' + df['Top'] + '. ' +
    'Notas de coração: ' + df['Middle'] + '. ' +
    'Notas de fundo: ' + df['Base'] + '. ' +
    'Acordes: ' + df['mainaccord1'] + ' ' + df['mainaccord2'] + ' ' + df['mainaccord3']
)

print(df['descricao'].head(3))

# Salva dataset limpo
df.to_csv('data/perfumes_masculinos.csv', index=False)
print("\nDataset salvo com sucesso!")
print("Total:", len(df), "perfumes")