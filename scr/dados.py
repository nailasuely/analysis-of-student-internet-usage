import pandas as pd
import numpy as np

np.random.seed(42)
N = 100

ids = np.arange(1, N + 1)
idades = np.clip(np.random.normal(22, 3, N).astype(int), 17, 35)
sexos = np.random.choice([1, 2], size=N, p=[0.45, 0.55])
semestres = np.random.choice(range(1, 11), size=N)
rendas = np.round(np.clip(np.random.normal(4000, 1500, N), 1000, 12000), 0).astype(int)
periodos = np.random.choice([1, 2], size=N, p=[0.6, 0.4])
trabalha = np.random.choice([1, 2], size=N, p=[0.4, 0.6])
tempo_estudo_diario = np.round(np.clip(np.random.normal(4.0, 1.5, N), 1, 8), 0).astype(int)
mora_com = np.random.choice([1, 2, 3], size=N, p=[0.2, 0.3, 0.5])
tempo_uso_pc = np.clip(np.random.normal(10, 2.5, N), 4, 16).astype(int)
acessa_internet = np.random.choice([1, 2], size=N, p=[0.98, 0.02])
tempo_estudo_net = np.clip(tempo_estudo_diario * np.random.uniform(0.3, 1.1, N), 1, 8).astype(int)
dispositivo = np.random.choice([1, 2, 3], size=N, p=[0.6, 0.05, 0.35])
tempo_conectado = np.clip(np.random.normal(7, 1.5, N), 3, 12).astype(int)


def binario(p):
    return np.random.choice([1, 2], size=N, p=[p, 1 - p])

usa_trabalho = binario(0.5)
usa_amigos = binario(0.85)
usa_desconhecido = binario(0.4)
usa_email = binario(0.7)
usa_pesquisa = binario(0.95)
usa_noticias = binario(0.6)
usa_compras = binario(0.5)
usa_videos = binario(0.9)
usa_jogos = binario(0.4)
atrapalha_formacao = binario(0.4)
toxidade_redes = binario(0.3)
usa_download = binario(0.5)


representa = np.random.choice([1, 2, 3], size=N, p=[0.3, 0.3, 0.4])
sentimento_info = np.random.choice([1, 2, 3], size=N, p=[0.5, 0.3, 0.2])


df = pd.DataFrame({
    "ID": ids,
    "Idade": idades,
    "Sexo": sexos,
    "Semestre": semestres,
    "Renda_familiar": rendas,
    "Periodo": periodos,
    "Trabalha": trabalha,
    "Tempo_estudo_diario": tempo_estudo_diario,
    "Mora_com": mora_com,
    "Tempo_uso_computador": tempo_uso_pc,
    "Costuma_acessar_internet": acessa_internet,
    "Tempo_estudo_internet": tempo_estudo_net,
    "Dispositivo_mais_acessado": dispositivo,
    "Tempo_conectado_diario": tempo_conectado,
    "Usa_internet_trabalho": usa_trabalho,
    "Usa_internet_amigos": usa_amigos,
    "Usa_internet_desconhecido": usa_desconhecido,
    "Usa_internet_email": usa_email,
    "Usa_internet_pesquisa": usa_pesquisa,
    "Usa_internet_noticias": usa_noticias,
    "Usa_internet_compras": usa_compras,
    "Usa_internet_videos": usa_videos,
    "Usa_internet_jogos": usa_jogos,
    "Internet_atrapalha_formacao": atrapalha_formacao,
    "Redes_sociais_ambiente_toxico": toxidade_redes,
    "Usa_internet_download": usa_download,
    "O_que_computador_representa": representa,
    "Sentimento_informatica": sentimento_info
})

df.to_csv("dados_alunos.csv", index=False)
print(df.head())
