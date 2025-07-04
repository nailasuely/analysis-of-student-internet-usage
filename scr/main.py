import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os

graphics_folder = 'graphics3'
if not os.path.exists(graphics_folder):
    os.makedirs(graphics_folder)
    print(f"Pasta '{graphics_folder}' criada com sucesso.")

try:
    df = pd.read_csv('dados_alunos.csv', sep=',', encoding='latin-1')
    print("Arquivo CSV lido com sucesso.")
except Exception as e:
    print(f"Erro crítico ao carregar o arquivo: {e}")
    exit()


df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('?', '', regex=False)

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

df['sexo'] = df['sexo'].map(sexo_map)
df['periodo'] = df['periodo'].map(periodo_map)
df['trabalha'] = df['trabalha'].map(sim_nao_map)
df['mora_com'] = df['mora_com'].map(mora_com_map)
df['dispositivo_mais_acessado'] = df['dispositivo_mais_acessado'].map(dispositivo_map)
df['redes_sociais_ambiente_toxico'] = df['redes_sociais_ambiente_toxico'].map(sim_nao_map)
df['sentimento_informatica'] = df['sentimento_informatica'].map(sentimento_map)
df['internet_atrapalha_formacao'] = df['internet_atrapalha_formacao'].map(sim_nao_map)


if 'o_que_o_computador_representa_para_voce' in df.columns:
    df['o_que_o_computador_representa_para_voce'] = df['o_que_o_computador_representa_para_voce'].map(representa_map)

if 'voce_costuma_acessar_a_internet' in df.columns:
    df['voce_costuma_acessar_a_internet'] = df['voce_costuma_acessar_a_internet'].map(sim_nao_map)

colunas_uso_internet = [
    'usa_internet_trabalho', 'usa_internet_amigos', 'usa_internet_desconhecido',
    'usa_internet_email', 'usa_internet_pesquisa', 'usa_internet_noticias',
    'usa_internet_compras', 'usa_internet_videos', 'usa_internet_jogos',
    'usa_internet_download'
]
for col in colunas_uso_internet:
    if col in df.columns:
        df[col] = df[col].map(sim_nao_map)


df['renda_familiar'] = df['renda_familiar'].astype(str).str.replace(r'R\$\s*', '', regex=True)
df['renda_familiar'] = df['renda_familiar'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df['renda_familiar'] = pd.to_numeric(df['renda_familiar'], errors='coerce')

numeric_cols = ['idade', 'tempo_estudo_diario', 'tempo_conectado_diario', 'tempo_estudo_internet']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'semestre_que_est_cursando' in df.columns:
    df['semestre_que_est_cursando'] = pd.to_numeric(df['semestre_que_est_cursando'], errors='ignore')

print("\nDados mapeados e preparados com sucesso.")


sns.set_theme(style="whitegrid", font_scale=1.1)

plt.figure(figsize=(10, 10))
if 'dispositivo_mais_acessado' in df.columns and df['dispositivo_mais_acessado'].notna().any():
    df['dispositivo_mais_acessado'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14}, wedgeprops=dict(width=0.5))
    plt.title('Dispositivo de Acesso Mais Utilizado', fontsize=16)
    plt.ylabel('')
    plt.savefig(os.path.join(graphics_folder, '01_pizza_dispositivo.png'), dpi=300, bbox_inches='tight')
    plt.show()

plt.figure(figsize=(10, 6))
if 'mora_com' in df.columns:
    sns.countplot(y='mora_com', data=df, hue='mora_com', palette='magma', order=df['mora_com'].value_counts().index, legend=False)
    plt.title('Situação de Moradia dos Estudantes')
    plt.xlabel('Quantidade de Estudantes')
    plt.ylabel('Mora Com')
    plt.savefig(os.path.join(graphics_folder, '02_barras_moradia.png'), dpi=300, bbox_inches='tight')
    plt.show()

plt.figure(figsize=(12, 7))
if 'idade' in df.columns:
    sns.histplot(df['idade'].dropna(), kde=True, color='indigo').set_title('Distribuição de Idade dos Estudantes')
    plt.xlabel('Idade (Anos)')
    plt.ylabel('Contagem de Estudantes')
    plt.savefig(os.path.join(graphics_folder, '03_histograma_idade.png'), dpi=300, bbox_inches='tight')
    plt.show()

plt.figure(figsize=(12, 7))
if 'renda_familiar' in df.columns:
    sns.histplot(df['renda_familiar'].dropna(), kde=True, color='teal').set_title('Distribuição de Renda Familiar')
    plt.xlabel('Renda Familiar (R$)')
    plt.ylabel('Contagem de Famílias')
    plt.savefig(os.path.join(graphics_folder, '04_histograma_renda.png'), dpi=300, bbox_inches='tight')
    plt.show()

atividades_counts = {}
for col in colunas_uso_internet:
    if col in df.columns:
        atividades_counts[col.replace('usa_internet_', '').replace('_', ' ').capitalize()] = df[col].value_counts().get('Sim', 0)
atividades_df = pd.DataFrame(list(atividades_counts.items()), columns=['Atividade', 'Contagem de "Sim"']).sort_values(by='Contagem de "Sim"', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='Contagem de "Sim"', y='Atividade', data=atividades_df, hue='Atividade', palette='flare', dodge=False, legend=False)
plt.title('Principais Finalidades de Uso da Internet', fontsize=16)
plt.xlabel('Número de Estudantes (Respostas "Sim")')
plt.ylabel('Atividade Online')
plt.savefig(os.path.join(graphics_folder, '05_ranking_atividades.png'), dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 7))
if 'sentimento_informatica' in df.columns:
    sns.countplot(y='sentimento_informatica', data=df, hue='sentimento_informatica', palette='viridis', order=df['sentimento_informatica'].value_counts().index, legend=False)
    plt.title('Sentimento dos Estudantes em Relação à Informática', fontsize=16)
    plt.xlabel('Quantidade de Estudantes')
    plt.ylabel('Sentimento Declarado')
    plt.savefig(os.path.join(graphics_folder, '06_barras_sentimento.png'), dpi=300, bbox_inches='tight')
    plt.show()

plt.figure(figsize=(12, 8))
sns.violinplot(x='sexo', y='tempo_conectado_diario', data=df, hue='sexo', palette='pastel', legend=False, inner='quartile')
plt.title('Comparação e Distribuição do Tempo Conectado por Sexo', fontsize=16)
plt.xlabel('Sexo')
plt.ylabel('Horas Conectado por Dia')
plt.savefig(os.path.join(graphics_folder, '07_violino_tempo_sexo.png'), dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8))
sns.regplot(x='tempo_conectado_diario', y='tempo_estudo_internet', data=df, line_kws={'color': 'blue'}, scatter_kws={'alpha':0.5})
plt.title('Tempo Total Conectado vs. Tempo de Estudo na Internet', fontsize=16)
plt.xlabel('Horas Totais Conectado por Dia')
plt.ylabel('Horas de Estudo na Internet por Dia')
plt.savefig(os.path.join(graphics_folder, '08_dispersao_tempo_total_vs_estudo.png'), dpi=300, bbox_inches='tight')
plt.show()

matriz_corr_vars = ['idade', 'renda_familiar', 'tempo_estudo_diario', 'tempo_conectado_diario']
matriz_corr = df[[col for col in matriz_corr_vars if col in df.columns]].corr(method='pearson')
nomes_amigaveis = ['Idade', 'Renda Familiar', 'Tempo de Estudo', 'Tempo Conectado']
matriz_corr_amigavel = matriz_corr.copy()
matriz_corr_amigavel.columns = nomes_amigaveis
matriz_corr_amigavel.index = nomes_amigaveis
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr_amigavel, annot=True, cmap='plasma', fmt=".2f", annot_kws={"size": 12})
plt.title('Mapa de Calor de Correlação entre Variáveis', fontsize=16)
plt.savefig(os.path.join(graphics_folder, '09_heatmap_correlacao.png'), dpi=300, bbox_inches='tight')
plt.show()



plt.figure(figsize=(10, 7))
sns.boxplot(x='periodo', y='renda_familiar', data=df, palette='coolwarm')
plt.title('Distribuição da Renda Familiar por Período')
plt.xlabel('Período')
plt.ylabel('Renda Familiar (R$)')
plt.savefig(os.path.join(graphics_folder, '10_boxplot_renda_periodo.png'), dpi=300, bbox_inches='tight')
plt.show()

if 'trabalha' in df.columns and 'periodo' in df.columns:
    pd.crosstab(df['periodo'], df['trabalha']).plot(kind='bar', stacked=True, colormap='Accent', figsize=(10,7))
    plt.title('Trabalho por Período de Estudo')
    plt.xlabel('Período')
    plt.ylabel('Quantidade de Estudantes')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(graphics_folder, '11_empilhado_trabalho_periodo.png'), dpi=300, bbox_inches='tight')
    plt.show()

plt.figure(figsize=(10, 7))
sns.boxplot(x='sexo', y='tempo_estudo_diario', data=df, palette='Set2')
plt.title('Tempo de Estudo Diário por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Horas de Estudo por Dia')
plt.savefig(os.path.join(graphics_folder, '12_boxplot_estudo_sexo.png'), dpi=300, bbox_inches='tight')
plt.show()
corr_vars_amp = ['idade', 'renda_familiar', 'tempo_estudo_diario', 'tempo_conectado_diario', 'tempo_estudo_internet']
corr_matrix_amp = df[[col for col in corr_vars_amp if col in df.columns]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_amp, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor - Correlação Ampliada')
plt.savefig(os.path.join(graphics_folder, '13_heatmap_ampliado.png'), dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='tempo_conectado_diario', y='tempo_estudo_internet', hue='sexo', data=df, palette='Dark2')
plt.title('Tempo Estudo vs Tempo Conectado (por Sexo)')
plt.xlabel('Tempo Conectado (h/dia)')
plt.ylabel('Tempo Estudo na Internet (h/dia)')
plt.savefig(os.path.join(graphics_folder, '14_dispersao_sexo_estudo_conexao.png'), dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='dispositivo_mais_acessado', hue='sexo', palette='pastel')
plt.title('Dispositivo mais Usado por Sexo')
plt.xlabel('Dispositivo')
plt.ylabel('Quantidade de Estudantes')
plt.savefig(os.path.join(graphics_folder, '15_dispositivo_sexo.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n--- GERANDO GRÁFICOS PARA PERGUNTAS ADICIONAIS ---")

if 'o_que_o_computador_representa_para_voce' in df.columns:
    plt.figure(figsize=(12, 8))
    sns.countplot(y='o_que_o_computador_representa_para_voce', data=df, hue='o_que_o_computador_representa_para_voce',
                  palette='crest', order=df['o_que_o_computador_representa_para_voce'].value_counts().index, legend=False)
    plt.title('O Que o Computador Representa para os Estudantes', fontsize=16)
    plt.xlabel('Quantidade de Estudantes')
    plt.ylabel('Representação')
    plt.savefig(os.path.join(graphics_folder, '16_barras_representacao_pc.png'), dpi=300, bbox_inches='tight')
    plt.show()


if 'semestre_que_est_cursando' in df.columns:
    plt.figure(figsize=(12, 7))
    order_semestre = df['semestre_que_est_cursando'].dropna().astype(str).unique()
    try:
        order_semestre = sorted(order_semestre, key=int)
    except ValueError:
        pass
    sns.countplot(x='semestre_que_est_cursando', data=df, palette='rocket', order=order_semestre)
    plt.title('Distribuição de Estudantes por Semestre', fontsize=16)
    plt.xlabel('Semestre')
    plt.ylabel('Quantidade de Estudantes')
    plt.savefig(os.path.join(graphics_folder, '17_barras_semestre.png'), dpi=300, bbox_inches='tight')
    plt.show()

if 'h_quanto_tempo_utiliza_computador' in df.columns:
    plt.figure(figsize=(12, 7))
    sns.countplot(y='h_quanto_tempo_utiliza_computador', data=df, hue='h_quanto_tempo_utiliza_computador',
                  palette='mako', order=df['h_quanto_tempo_utiliza_computador'].value_counts().index, legend=False)
    plt.title('Tempo de Uso de Computador', fontsize=16)
    plt.xlabel('Quantidade de Estudantes')
    plt.ylabel('Tempo de Uso')
    plt.savefig(os.path.join(graphics_folder, '18_barras_tempo_uso_pc.png'), dpi=300, bbox_inches='tight')
    plt.show()

if 'voce_costuma_acessar_a_internet' in df.columns and df['voce_costuma_acessar_a_internet'].notna().any():
    plt.figure(figsize=(10, 8))
    df['voce_costuma_acessar_a_internet'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
                                                                 colors=['#66c2a5','#fc8d62'], textprops={'fontsize': 14})
    plt.title('Proporção de Estudantes que Acessam a Internet Regularmente', fontsize=16)
    plt.ylabel('')
    plt.savefig(os.path.join(graphics_folder, '19_pizza_acessa_internet.png'), dpi=300, bbox_inches='tight')
    plt.show()


print("\n--- REALIZANDO CÁLCULOS ESTATÍSTICOS ---")

desc_quant_cols = [col for col in ['idade', 'renda_familiar', 'tempo_conectado_diario'] if col in df.columns]
if desc_quant_cols:
    desc_quant = df[desc_quant_cols].agg(['mean', 'median', 'std']).T
    print("\n--- Tabela Descritiva (Variáveis Quantitativas) ---")
    print(desc_quant)

if 'trabalha' in df.columns and 'redes_sociais_ambiente_toxico' in df.columns:
    tabela_contingencia = pd.crosstab(df['trabalha'], df['redes_sociais_ambiente_toxico'])
    chi2, p_valor_chi2, dof, expected = stats.chi2_contingency(tabela_contingencia)
    print("\n--- Teste de Associação (Qui-Quadrado) ---")
    print(f"Teste (Trabalho vs. Percepção Redes Sociais): X²({dof}) = {chi2:.2f}, p-valor = {p_valor_chi2:.4f}")

if 'sexo' in df.columns and 'tempo_conectado_diario' in df.columns:
    tempo_masc = df.loc[df['sexo'] == 'Masculino', 'tempo_conectado_diario'].dropna()
    tempo_fem = df.loc[df['sexo'] == 'Feminino', 'tempo_conectado_diario'].dropna()
    if not tempo_masc.empty and not tempo_fem.empty:
        mannwhitney_test = stats.mannwhitneyu(tempo_masc, tempo_fem, alternative='two-sided')
        p_valor_mw = mannwhitney_test.pvalue
        print("\n--- Teste de Comparação de Grupos (Mann-Whitney U) ---")
        print(f"P-valor da comparação do tempo conectado entre sexos: {p_valor_mw:.4f}")
    else:
        print("\n--- Teste de Comparação de Grupos (Mann-Whitney U) ---")
        print("Teste de comparação do tempo conectado entre sexos não pôde ser realizado por falta de dados em um dos grupos.")


device_order = ['Celular', 'Computador/Notebook', 'Tablet']


plt.figure(figsize=(12, 7))
if 'dispositivo_mais_acessado' in df.columns and 'renda_familiar' in df.columns:
    sns.boxplot(
        x='dispositivo_mais_acessado',
        y='renda_familiar',
        data=df,
        palette='viridis',
        order=device_order
    )
    plt.title('Distribuição da Renda Familiar por Dispositivo Principal', fontsize=16)
    plt.xlabel('Dispositivo Principal', fontsize=12)
    plt.ylabel('Renda Familiar (R$)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_folder, '16_boxplot_renda_dispositivo.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if all(col in df.columns for col in ['dispositivo_mais_acessado', 'renda_familiar', 'periodo', 'trabalha']):
    g = sns.catplot(
        data=df,
        x='dispositivo_mais_acessado',
        y='renda_familiar',
        col='periodo',
        row='trabalha',
        kind='box',
        palette='plasma',
        height=5,
        aspect=1.2,
        order=device_order,
        sharey=True
    )
    g.fig.suptitle('Renda vs. Dispositivo por Período de Estudo e Situação de Trabalho', y=1.03, fontsize=16)
    g.set_axis_labels('Dispositivo Principal', 'Renda Familiar (R$)')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.savefig(os.path.join(graphics_folder, '17_catplot_renda_dispositivo_trabalho_periodo.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

plt.figure(figsize=(12, 7))
if 'semestre_que_est_cursando' in df.columns and 'tempo_estudo_internet' in df.columns:
    sns.boxplot(x='semestre_que_est_cursando', y='tempo_estudo_internet', data=df, palette='magma')
    plt.title('Distribuição do Tempo de Estudo na Internet por Semestre', fontsize=16)
    plt.xlabel('Semestre', fontsize=12)
    plt.ylabel('Horas de Estudo na Internet', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_folder, '18_boxplot_estudo_semestre.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if 'semestre_que_est_cursando' in df.columns:
    bins_semestre = [0, 3, 7, 10]
    labels_semestre = ['Iniciante (1-3)', 'Intermediário (4-7)', 'Finalista (8-10)']
    df['faixa_semestre'] = pd.cut(df['semestre_que_est_cursando'], bins=bins_semestre, labels=labels_semestre, right=True)

if 'renda_familiar' in df.columns:
    bins_renda = [0, 3000, 6000, np.inf]
    labels_renda = ['Baixa (até R$3k)', 'Média (R$3k-R$6k)', 'Alta (> R$6k)']
    df['faixa_renda'] = pd.cut(df['renda_familiar'], bins=bins_renda, labels=labels_renda, right=False)

if 'faixa_semestre' in df.columns:
    finalidades_semestre = [col for col in ['usa_internet_pesquisa', 'usa_internet_videos', 'usa_internet_jogos', 'usa_internet_trabalho'] if col in df.columns]
    if finalidades_semestre:
        df_finalidades_semestre = df.groupby('faixa_semestre')[finalidades_semestre].apply(lambda x: x.eq('Sim').mean()).unstack().reset_index()
        df_finalidades_semestre.columns = ['Faixa de Semestre', 'Finalidade', 'Proporção']
        df_finalidades_semestre['Finalidade'] = df_finalidades_semestre['Finalidade'].str.replace('usa_internet_', '').str.capitalize()

        plt.figure(figsize=(14, 8))
        sns.barplot(x='Faixa de Semestre', y='Proporção', hue='Finalidade', data=df_finalidades_semestre, palette='crest')
        plt.title('Proporção de Finalidades de Uso da Internet por Faixa de Semestre', fontsize=16)
        plt.xlabel('Faixa de Semestre', fontsize=12)
        plt.ylabel('Proporção de Estudantes (%)', fontsize=12)
        plt.legend(title='Finalidade')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_folder, '19_barplot_finalidade_semestre.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

if 'faixa_renda' in df.columns:
    finalidades_renda = [col for col in ['usa_internet_pesquisa', 'usa_internet_trabalho', 'usa_internet_compras', 'usa_internet_jogos'] if col in df.columns]
    if finalidades_renda:
        df_finalidades_renda = df.groupby('faixa_renda')[finalidades_renda].apply(lambda x: x.eq('Sim').mean()).unstack().reset_index()
        df_finalidades_renda.columns = ['Faixa de Renda', 'Finalidade', 'Proporção']
        df_finalidades_renda['Finalidade'] = df_finalidades_renda['Finalidade'].str.replace('usa_internet_', '').str.capitalize()

        plt.figure(figsize=(14, 8))
        sns.barplot(x='Faixa de Renda', y='Proporção', hue='Finalidade', data=df_finalidades_renda, palette='flare')
        plt.title('Proporção de Finalidades de Uso da Internet por Faixa de Renda', fontsize=16)
        plt.xlabel('Faixa de Renda Familiar', fontsize=12)
        plt.ylabel('Proporção de Estudantes (%)', fontsize=12)
        plt.legend(title='Finalidade')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_folder, '20_barplot_finalidade_renda.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

plt.figure(figsize=(10, 7))
if 'redes_sociais_ambiente_toxico' in df.columns and 'tempo_conectado_diario' in df.columns:
    sns.boxplot(x='redes_sociais_ambiente_toxico', y='tempo_conectado_diario', data=df, palette='coolwarm')
    plt.title('Tempo Conectado vs. Percepção de Toxicidade nas Redes Sociais', fontsize=16)
    plt.xlabel('Considera as Redes Sociais um Ambiente Tóxico?', fontsize=12)
    plt.ylabel('Horas Conectado Diariamente', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_folder, '21_boxplot_tempo_toxicidade.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if 'o_que_o_computador_representa_para_voce' in df.columns:
    plt.figure(figsize=(12, 8))
    sns.countplot(y='o_que_o_computador_representa_para_voce', data=df, hue='o_que_o_computador_representa_para_voce',
                  palette='crest', order=df['o_que_o_computador_representa_para_voce'].value_counts().index, legend=False)
    plt.title('O Que o Computador Representa para os Estudantes', fontsize=16)
    plt.xlabel('Quantidade de Estudantes')
    plt.ylabel('Representação')
    plt.savefig(os.path.join(graphics_folder, '22_barras_representacao_pc.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if 'semestre_que_est_cursando' in df.columns:
    plt.figure(figsize=(12, 7))
    order_semestre = sorted(df['semestre_que_est_cursando'].dropna().unique())
    sns.countplot(x='semestre_que_est_cursando', data=df, palette='rocket', order=order_semestre)
    plt.title('Distribuição de Estudantes por Semestre', fontsize=16)
    plt.xlabel('Semestre')
    plt.ylabel('Quantidade de Estudantes')
    plt.savefig(os.path.join(graphics_folder, '23_barras_semestre.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if 'h_quanto_tempo_utiliza_computador' in df.columns:
    plt.figure(figsize=(12, 7))
    sns.countplot(y='h_quanto_tempo_utiliza_computador', data=df, hue='h_quanto_tempo_utiliza_computador',
                  palette='mako', order=df['h_quanto_tempo_utiliza_computador'].value_counts().index, legend=False)
    plt.title('Tempo de Uso de Computador', fontsize=16)
    plt.xlabel('Quantidade de Estudantes')
    plt.ylabel('Tempo de Uso')
    plt.savefig(os.path.join(graphics_folder, '24_barras_tempo_uso_pc.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if 'voce_costuma_acessar_a_internet' in df.columns and df['voce_costuma_acessar_a_internet'].notna().any():
    plt.figure(figsize=(10, 8))
    df['voce_costuma_acessar_a_internet'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
                                                                 colors=['#66c2a5','#fc8d62'], textprops={'fontsize': 14})
    plt.title('Proporção de Estudantes que Acessam a Internet Regularmente', fontsize=16)
    plt.ylabel('')
    plt.savefig(os.path.join(graphics_folder, '25_pizza_acessa_internet.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


print("\n--- Gerando gráficos simples para perguntas básicas ---")

if 'sexo' in df.columns:
    plt.figure(figsize=(8, 8))
    df['sexo'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff','#ff9999'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    plt.title('Distribuição por Sexo', fontsize=16)
    plt.ylabel('') 
    plt.savefig(os.path.join(graphics_folder, '26_pizza_sexo.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Gráfico 26 (Pizza Sexo) gerado.")

if 'periodo' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='periodo', hue='periodo', palette='coolwarm', legend=False)
    plt.title('Distribuição de Estudantes por Período', fontsize=16)
    plt.xlabel('Período de Estudo', fontsize=12)
    plt.ylabel('Quantidade de Estudantes', fontsize=12)
    plt.savefig(os.path.join(graphics_folder, '27_barras_periodo.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Gráfico 27 (Barras Período) gerado.")

if 'trabalha' in df.columns:
    plt.figure(figsize=(8, 8))
    df['trabalha'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ffcc99','#99ff99'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    plt.title('Proporção de Estudantes que Trabalham', fontsize=16)
    plt.ylabel('')
    plt.savefig(os.path.join(graphics_folder, '28_pizza_trabalha.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Gráfico 28 (Pizza Trabalha) gerado.")

if 'internet_atrapalha_formacao' in df.columns:
    plt.figure(figsize=(8, 8))
    df['internet_atrapalha_formacao'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=['#c2c2f0','#ffb3e6'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    plt.title('Percepção: A Internet Atrapalha a Formação?', fontsize=16)
    plt.ylabel('')
    plt.savefig(os.path.join(graphics_folder, '29_pizza_internet_atrapalha.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Gráfico 29 (Pizza Internet Atrapalha) gerado.")


if 'redes_sociais_ambiente_toxico' in df.columns:
    plt.figure(figsize=(8, 8))
    df['redes_sociais_ambiente_toxico'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ff6666','#ffb366'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    plt.title('Percepção: Redes Sociais são um Ambiente Tóxico?', fontsize=16)
    plt.ylabel('')
    plt.savefig(os.path.join(graphics_folder, '30_pizza_redes_toxicas.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Gráfico 30 (Pizza Redes Tóxicas) gerado.")

print("\n\nAnálise finalizada. Gráficos salvos na pasta 'graphics'.")