import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

graphics_folder = 'graficos_individuais'
if not os.path.exists(graphics_folder):
    os.makedirs(graphics_folder)
    print(f"Pasta '{graphics_folder}' criada com sucesso.")

try:
    try:
        df = pd.read_csv('dados_alunos.csv', sep=',', encoding='latin-1')
    except UnicodeDecodeError:
        df = pd.read_csv('dados_alunos.csv', sep=',', encoding='utf-8')
    print("Arquivo CSV lido com sucesso.")
except Exception as e:
    print(f"Erro crítico ao carregar o arquivo: {e}")
    exit()

df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('?', '', regex=False)
print("Nomes das colunas padronizados.")

sexo_map = {1: 'Masculino', 2: 'Feminino'}
sim_nao_map = {1: 'Sim', 2: 'Não'}
periodo_map = {1: 'Diurno', 2: 'Noturno'}
sentimento_map = {1: 'Entusiasmado', 2: 'Obrigado a aprender', 3: 'Acho difícil'}
mora_com_map = {1: 'Sozinho(a)', 2: 'Com Amigos', 3: 'Com a Família'}
dispositivo_map = {1: 'Celular', 2: 'Tablet', 3: 'Computador/Notebook'}
representa_map = {
    1: 'Avanço que melhora a vida',
    2: 'Comunicação mais rápida',
    3: 'Atrapalha e complica'
}

map_config = {
    'sexo': sexo_map,
    'periodo': periodo_map,
    'trabalha': sim_nao_map,
    'mora_com': mora_com_map,
    'dispositivo_mais_acessado': dispositivo_map,
    'redes_sociais_ambiente_toxico': sim_nao_map,
    'sentimento_informatica': sentimento_map,
    'internet_atrapalha_formacao': sim_nao_map,
    'o_que_o_computador_representa_para_voce': representa_map,
    'voce_costuma_acessar_a_internet': sim_nao_map
}

colunas_uso_internet = [
    'usa_internet_trabalho', 'usa_internet_amigos', 'usa_internet_desconhecido',
    'usa_internet_email', 'usa_internet_pesquisa', 'usa_internet_noticias',
    'usa_internet_compras', 'usa_internet_videos', 'usa_internet_jogos',
    'usa_internet_download'
]
for col in colunas_uso_internet:
    map_config[col] = sim_nao_map

for col, mapping in map_config.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

if 'renda_familiar' in df.columns:
    df['renda_familiar'] = df['renda_familiar'].astype(str).str.replace(r'R\$\s*', '', regex=True)
    df['renda_familiar'] = df['renda_familiar'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df['renda_familiar'] = pd.to_numeric(df['renda_familiar'], errors='coerce')

numeric_cols = ['idade', 'tempo_estudo_diario', 'tempo_conectado_diario', 'tempo_estudo_internet', 'semestre_que_est_cursando']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nDados mapeados e preparados com sucesso.")

print("\nIniciando a geração dos gráficos individuais...")
sns.set_theme(style="whitegrid", font_scale=1.1, palette='viridis')

def save_and_close(filename):
    plt.savefig(os.path.join(graphics_folder, filename), dpi=300, bbox_inches='tight')
    plt.close()

if 'idade' in df.columns:
    print("\n--- Análise Numérica: 1. Idade ---")
    print(df['idade'].describe().round(2))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['idade'].dropna(), kde=True, bins=15)
    plt.title('1. Distribuição de Idade dos Estudantes')
    plt.xlabel('Idade (Anos)')
    plt.ylabel('Quantidade')
    save_and_close('01_idade_histograma.png')
    print("Gráfico 1 (Idade) gerado.")

if 'sexo' in df.columns:
    print("\n--- Análise Numérica: 2. Sexo ---")
    print(df['sexo'].value_counts())
    print(df['sexo'].value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')
    plt.figure(figsize=(8, 8))
    df['sexo'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    plt.title('2. Distribuição por Sexo')
    plt.ylabel('')
    save_and_close('02_sexo_pizza.png')
    print("Gráfico 2 (Sexo) gerado.")

if 'semestre_que_est_cursando' in df.columns:
    print("\n--- Análise Numérica: 3. Semestre ---")
    print(df['semestre_que_est_cursando'].value_counts().sort_index())
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='semestre_que_est_cursando', palette='magma')
    plt.title('3. Distribuição de Estudantes por Semestre')
    plt.xlabel('Semestre')
    plt.ylabel('Quantidade')
    save_and_close('03_semestre_barras.png')
    print("Gráfico 3 (Semestre) gerado.")

if 'renda_familiar' in df.columns:
    print("\n--- Análise Numérica: 4. Renda Familiar ---")
    print(df['renda_familiar'].describe().round(2))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['renda_familiar'].dropna(), kde=True)
    plt.title('4. Distribuição de Renda Familiar')
    plt.xlabel('Renda (R$)')
    plt.ylabel('Quantidade')
    save_and_close('04_renda_histograma.png')
    print("Gráfico 4 (Renda) gerado.")

if 'periodo' in df.columns:
    print("\n--- Análise Numérica: 5. Período ---")
    print(df['periodo'].value_counts())
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='periodo', hue='periodo', palette='coolwarm', legend=False)
    plt.title('5. Distribuição por Período de Estudo')
    plt.xlabel('Período')
    plt.ylabel('Quantidade')
    save_and_close('05_periodo_barras.png')
    print("Gráfico 5 (Período) gerado.")

if 'trabalha' in df.columns:
    print("\n--- Análise Numérica: 6. Você Trabalha? ---")
    print(df['trabalha'].value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')
    plt.figure(figsize=(8, 8))
    df['trabalha'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('6. Você Trabalha?')
    plt.ylabel('')
    save_and_close('06_trabalha_pizza.png')
    print("Gráfico 6 (Trabalha) gerado.")

if 'tempo_estudo_diario' in df.columns:
    print("\n--- Análise Numérica: 7. Tempo de Estudo Diário ---")
    print(df['tempo_estudo_diario'].describe().round(2))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['tempo_estudo_diario'].dropna(), kde=True, bins=10)
    plt.title('7. Distribuição do Tempo de Estudo Diário')
    plt.xlabel('Horas de Estudo por Dia')
    plt.ylabel('Quantidade')
    save_and_close('07_tempo_estudo_diario_hist.png')
    print("Gráfico 7 (Tempo Estudo Diário) gerado.")

if 'mora_com' in df.columns:
    print("\n--- Análise Numérica: 8. Situação de Moradia ---")
    print(df['mora_com'].value_counts())
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='mora_com', hue='mora_com', order=df['mora_com'].value_counts().index, legend=False)
    plt.title('8. Situação de Moradia')
    plt.xlabel('Quantidade')
    plt.ylabel('')
    save_and_close('08_mora_com_barras.png')
    print("Gráfico 8 (Mora Com) gerado.")

if 'h_quanto_tempo_utiliza_computador' in df.columns:
    print("\n--- Análise Numérica: 9. Há Quanto Tempo Utiliza Computador? ---")
    print(df['h_quanto_tempo_utiliza_computador'].value_counts())
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='h_quanto_tempo_utiliza_computador', hue='h_quanto_tempo_utiliza_computador', order=df['h_quanto_tempo_utiliza_computador'].value_counts().index, legend=False)
    plt.title('9. Há Quanto Tempo Utiliza Computador?')
    plt.xlabel('Quantidade')
    plt.ylabel('')
    save_and_close('09_tempo_uso_pc_barras.png')
    print("Gráfico 9 (Tempo Uso PC) gerado.")

if 'voce_costuma_acessar_a_internet' in df.columns:
    print("\n--- Análise Numérica: 10. Você Costuma Acessar a Internet? ---")
    print(df['voce_costuma_acessar_a_internet'].value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')
    plt.figure(figsize=(8, 8))
    df['voce_costuma_acessar_a_internet'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('10. Você Costuma Acessar a Internet?')
    plt.ylabel('')
    save_and_close('10_acessa_internet_pizza.png')
    print("Gráfico 10 (Acessa Internet) gerado.")

if 'tempo_estudo_internet' in df.columns:
    print("\n--- Análise Numérica: 11. Tempo de Estudo na Internet ---")
    print(df['tempo_estudo_internet'].describe().round(2))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['tempo_estudo_internet'].dropna(), kde=True, bins=10)
    plt.title('11. Distribuição do Tempo de Estudo na Internet')
    plt.xlabel('Horas de Estudo na Internet por Dia')
    plt.ylabel('Quantidade')
    save_and_close('11_tempo_estudo_internet_hist.png')
    print("Gráfico 11 (Tempo Estudo Internet) gerado.")


if 'dispositivo_mais_acessado' in df.columns:
    print("\n--- Análise Numérica: 12. Dispositivo de Acesso ---")
    print(df['dispositivo_mais_acessado'].value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')
    plt.figure(figsize=(8, 8))
    df['dispositivo_mais_acessado'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('12. Dispositivo de Acesso Mais Utilizado')
    plt.ylabel('')
    save_and_close('12_dispositivo_pizza.png')
    print("Gráfico 12 (Dispositivo) gerado.")


if 'tempo_conectado_diario' in df.columns:
    print("\n--- Análise Numérica: 13. Tempo Conectado Diário ---")
    print(df['tempo_conectado_diario'].describe().round(2))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['tempo_conectado_diario'].dropna(), kde=True, bins=12)
    plt.title('13. Tempo Diário Conectado à Internet')
    plt.xlabel('Horas Conectado por Dia')
    plt.ylabel('Quantidade')
    save_and_close('13_tempo_conectado_hist.png')
    print("Gráfico 13 (Tempo Conectado) gerado.")


for i, col in enumerate(colunas_uso_internet, 14):
    if i == 23: i = 25 
    if col in df.columns:
        titulo_amigavel = col.replace('usa_internet_', '').replace('_', ' ').capitalize()
        print(f"\n--- Análise Numérica: {i}. Usa a Internet para {titulo_amigavel}? ---")
        print(df[col].value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')
        plt.figure(figsize=(8, 8))
        df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#99ff99','#ff9999'])
        plt.title(f'{i}. Usa a Internet para {titulo_amigavel}?')
        plt.ylabel('')
        save_and_close(f'{i}_{col}_pizza.png')
        print(f"Gráfico {i} ({titulo_amigavel}) gerado.")

if 'internet_atrapalha_formacao' in df.columns:
    print("\n--- Análise Numérica: 23. Acredita que a Internet Atrapalha a Formação? ---")
    print(df['internet_atrapalha_formacao'].value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')
    plt.figure(figsize=(8, 8))
    df['internet_atrapalha_formacao'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('23. Acredita que a Internet Atrapalha a Formação?')
    plt.ylabel('')
    save_and_close('23_internet_atrapalha_pizza.png')
    print("Gráfico 23 (Internet Atrapalha) gerado.")

if 'redes_sociais_ambiente_toxico' in df.columns:
    print("\n--- Análise Numérica: 24. Considera as Redes Sociais um Ambiente Tóxico? ---")
    print(df['redes_sociais_ambiente_toxico'].value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')
    plt.figure(figsize=(8, 8))
    df['redes_sociais_ambiente_toxico'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('24. Considera as Redes Sociais um Ambiente Tóxico?')
    plt.ylabel('')
    save_and_close('24_redes_toxicas_pizza.png')
    print("Gráfico 24 (Redes Tóxicas) gerado.")

if 'o_que_o_computador_representa_para_voce' in df.columns:
    print("\n--- Análise Numérica: 26. O que o Computador Representa? ---")
    print(df['o_que_o_computador_representa_para_voce'].value_counts())
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='o_que_o_computador_representa_para_voce', hue='o_que_o_computador_representa_para_voce', order=df['o_que_o_computador_representa_para_voce'].value_counts().index, legend=False)
    plt.title('26. O que o Computador Representa?')
    plt.xlabel('Quantidade')
    plt.ylabel('')
    save_and_close('26_representacao_pc_barras.png')
    print("Gráfico 26 (Representação PC) gerado.")

if 'sentimento_informatica' in df.columns:
    print("\n--- Análise Numérica: 27. Sentimento em Relação à Informática ---")
    print(df['sentimento_informatica'].value_counts())
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='sentimento_informatica', hue='sentimento_informatica', order=df['sentimento_informatica'].value_counts().index, legend=False)
    plt.title('27. Sentimento em Relação à Informática')
    plt.xlabel('Quantidade')
    plt.ylabel('')
    save_and_close('27_sentimento_informatica_barras.png')
    print("Gráfico 27 (Sentimento Informática) gerado.")

print("\n\nAnálise finalizada. Todos os gráficos individuais foram salvos na pasta 'graficos_individuais'.")
