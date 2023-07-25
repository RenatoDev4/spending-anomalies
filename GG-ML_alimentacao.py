import locale

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv('cota-parlamentar.csv')

frases_unicas = df['txtdescricao'].unique()

for frase in frases_unicas:
    print(frase)

filtro_divulgacao = df['txtdescricao'] == 'DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR.'
filtro_alimentacao = df['txtdescricao'] == 'FORNECIMENTO DE ALIMENTAÇÃO DO PARLAMENTAR'
filtro_manutencao_escritorio = df['txtdescricao'] == 'MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE PARLAMENTAR'
filtro_passagens_aereas = df['txtdescricao'] == 'PASSAGENS AÉREAS'
# filtro_telefonia = df['txtdescricao'] == 'TELEFONIA'
# filtro_locacao_veiculos = df['txtdescricao'] == 'LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES'
# filtro_servicos_postais = df['txtdescricao'] == 'SERVIÇOS POSTAIS'
# filtro_consultoria = df['txtdescricao'] == 'CONSULTORIAS, PESQUISAS E TRABALHOS TÉCNICOS.'
# filtro_servico_seguranca = df['txtdescricao'] == 'SERVIÇO DE SEGURANÇA PRESTADO POR EMPRESA ESPECIALIZADA.'
# filtro_emissao_bilhete_aereo = df['txtdescricao'] == 'Emissão Bilhete Aéreo'
# filtro_hospedagem = df['txtdescricao'] == 'HOSPEDAGEM ,EXCETO DO PARLAMENTAR NO DISTRITO FEDERAL.'
# filtro_taxi = df['txtdescricao'] == 'SERVIÇO DE TÁXI, PEDÁGIO E ESTACIONAMENTO'
# filtro_curso_palestra = df['txtdescricao'] == 'PARTICIPAÇÃO EM CURSO, PALESTRA OU EVENTO SIMILAR'
# filtro_curso_locacao_aeronave = df['txtdescricao'] == 'LOCAÇÃO OU FRETAMENTO DE AERONAVES'
# filtro_passagens = df['txtdescricao'] == 'PASSAGENS TERRESTRES, MARÍTIMAS OU FLUVIAIS'
# filtro_assinatura_publicacoes = df['txtdescricao'] == 'ASSINATURA DE PUBLICAÇÕES'
# filtro_locacao_veiculos_ou_embarcacoes = df['txtdescricao'] == 'LOCAÇÃO DE VEÍCULOS AUTOMOTORES OU FRETAMENTO DE EMBARCAÇÕES'
# filtro_locomocao = df['txtdescricao'] == 'LOCOMOÇÃO, ALIMENTAÇÃO E  HOSPEDAGEM'
# filtro_escritorio = df['txtdescricao'] == 'AQUISIÇÃO DE MATERIAL DE ESCRITÓRIO.'
# filtro_aquisicao_ou_loc_software = df['txtdescricao'] == 'AQUISIÇÃO OU LOC. DE SOFTWARE; SERV. POSTAIS; ASS.'


df_alimentacao = df[filtro_alimentacao]
df_divulgacao = df[filtro_divulgacao]
df_manutencao_escritorio = df[filtro_manutencao_escritorio]
df_passagens_aereas = df[filtro_passagens_aereas]

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

df_alimentacao.isna()['vlrdocumento'].sum()
df_divulgacao.isna()['vlrdocumento'].sum()
df_manutencao_escritorio.isna()['vlrdocumento'].sum()
df_passagens_aereas.isna()['vlrdocumento'].sum()

df_alimentacao.dropna(subset=['vlrdocumento'], inplace=True)
df_divulgacao.dropna(subset=['vlrdocumento'], inplace=True)
df_manutencao_escritorio.dropna(subset=['vlrdocumento'], inplace=True)
df_passagens_aereas.dropna(subset=['vlrdocumento'], inplace=True)

# Função para filtrar partidos indesejados


def filtrar_partidos(df, partidos_indesejados):
    return df[~df['txnomeparlamentar'].isin(partidos_indesejados)]

# Função para calcular os gastos por parlamentar e fornecedor


def calcular_gastos_por_parlamentar(df, coluna_valor):
    return df.groupby('txnomeparlamentar')[coluna_valor].sum()


def calcular_gastos_por_fornecedor(df, coluna_valor):
    return df.groupby('txtfornecedor')[coluna_valor].sum()


# Lista de partidos indesejados
excluir_partidos = ['LIDERANÇA DO PSDB', 'LIDERANÇA DO PT',
                    'LIDMIN', 'NOVO', 'PDT', 'PODE', 'PP', 'PROS', 'PSD', 'PTB', 'SDD']


# Filtrar os dataframes de acordo com os partidos indesejados
df_alimentacao_filtrado = filtrar_partidos(df_alimentacao, excluir_partidos)
df_divulgacao_filtrado = filtrar_partidos(df_divulgacao, excluir_partidos)
df_manutencao_escritorio_filtrado = filtrar_partidos(
    df_manutencao_escritorio, excluir_partidos)
df_passagens_aereas_filtrado = filtrar_partidos(
    df_passagens_aereas, excluir_partidos)

# Calcular os gastos por parlamentar
gastos_por_parlamentar = calcular_gastos_por_parlamentar(
    df_alimentacao_filtrado, 'vlrdocumento')
gastos_por_parlamentar_divulgacao = calcular_gastos_por_parlamentar(
    df_divulgacao_filtrado, 'vlrdocumento')
gastos_por_parlamentar_manutencao_escritorio = calcular_gastos_por_parlamentar(
    df_manutencao_escritorio_filtrado, 'vlrdocumento')
gastos_por_parlamentar_passagens_aereas = calcular_gastos_por_parlamentar(
    df_passagens_aereas_filtrado, 'vlrdocumento')

# Calcular os gastos por fornecedor
pagamentos_por_fornecedor = calcular_gastos_por_fornecedor(
    df_alimentacao_filtrado, 'vlrdocumento')
pagamentos_por_fornecedor_divulgacao = calcular_gastos_por_fornecedor(
    df_divulgacao_filtrado, 'vlrdocumento')
pagamentos_por_fornecedor_manutencao_escritorio = calcular_gastos_por_fornecedor(
    df_manutencao_escritorio_filtrado, 'vlrdocumento')
pagamentos_por_fornecedor_passagens_aereas = calcular_gastos_por_fornecedor(
    df_passagens_aereas_filtrado, 'vlrdocumento')

top_10_gastos = gastos_por_parlamentar.nlargest(10)
top_10_gastos_divulgacao = gastos_por_parlamentar_divulgacao.nlargest(10)
top_10_gastos_manutencao_escritorio = gastos_por_parlamentar_manutencao_escritorio.nlargest(
    10)
top_10_gastos_passagens_aereas = gastos_por_parlamentar_passagens_aereas.nlargest(
    10)


def plot_top_10_gastos(data, title, category):
    plt.figure(figsize=(16, 8))
    colors = plt.cm.tab20.colors[:10]
    ax = plt.bar(data.index, data.values, color=colors)
    plt.ylabel('Total Gasto (R$)')
    plt.title(title)

    plt.gca().yaxis.set_major_formatter(locale.currency)

    plt.legend(ax, data.index, title='Parlamentares',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    for bar in ax:
        height = bar.get_height()
        plt.annotate(locale.currency(height), xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', rotation=40)

    plt.xticks([])

    plt.tight_layout()
    plt.show()


# Dados para os gráficos
plot_top_10_gastos(top_10_gastos_divulgacao,
                   'Top 10 Parlamentares com maior gasto em divulgação', 'divulgação')
plot_top_10_gastos(
    top_10_gastos, 'Top 10 "DEPUTADOS" com maior gasto em alimentação', 'alimentação')
plot_top_10_gastos(top_10_gastos_manutencao_escritorio,
                   'Top 10 "DEPUTADOS" com maior gasto em gastos com manutenção de escritório', 'manutenção de escritório')
plot_top_10_gastos(top_10_gastos_passagens_aereas,
                   'Top 10 "DEPUTADOS" com maior gasto em passagens aéreas', 'passagens aéreas')

top_10_pagamentos = pagamentos_por_fornecedor.nlargest(10)
top_10_pagamentos_fornecedor_divulgacao = pagamentos_por_fornecedor_divulgacao.nlargest(
    10)
top_10_pagamentos_por_fornecedor_manutencao_escritorio = pagamentos_por_fornecedor_manutencao_escritorio.nlargest(
    10)
top_10_pagamentos_por_fornecedor_passagens_aereas = pagamentos_por_fornecedor_passagens_aereas.nlargest(
    10)


def plot_top_10_pagamentos(data, title, row, col):
    colors = plt.cm.tab20.colors[:10]
    font_size = 9

    ax = axs[row, col].barh(data.index, data.values, color=colors)
    axs[row, col].set_xlabel('Total Recebido (R$)', fontsize=font_size)
    axs[row, col].set_title(title)
    axs[row, col].tick_params(axis='both', which='major', labelsize=font_size)
    axs[row, col].set_xticks([])  # Remover eixo X

    for index, value in enumerate(ax.patches):
        axs[row, col].text(value.get_width(), value.get_y() + value.get_height() / 2, locale.currency(value.get_width()),
                           ha='left', va='center', fontsize=font_size)


# Definir a figura geral e criar subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 16))
fig.subplots_adjust(wspace=10, hspace=0.5)

# Plotar os gráficos usando a função criada
plot_top_10_pagamentos(
    top_10_pagamentos, 'Top 10 Fornecedores que mais receberam pagamentos por alimentação', 0, 0)
plot_top_10_pagamentos(top_10_pagamentos_fornecedor_divulgacao,
                       'Top 10 Fornecedores que mais receberam pagamentos por divulgação', 0, 1)
plot_top_10_pagamentos(top_10_pagamentos_por_fornecedor_manutencao_escritorio,
                       'Top 10 Fornecedores que mais receberam pagamentos por manutenção em escritório', 1, 0)
plot_top_10_pagamentos(top_10_pagamentos_por_fornecedor_passagens_aereas,
                       'Top 10 Fornecedores que mais receberam pagamentos por passagens aéreas', 1, 1)

# Ajustar o layout para evitar cortes de rótulos
plt.tight_layout()

# Mostrar os gráficos lado a lado
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

colunas_relevantes = ['vlrdocumento']
df_alimentacao = df_alimentacao[colunas_relevantes]

scaler_minMax = MinMaxScaler()
dados_normalizados = scaler_minMax.fit_transform(df_alimentacao)

train_data, test_data = train_test_split(
    dados_normalizados, test_size=0.2, random_state=42)

tensor_train_data = torch.tensor(
    train_data, dtype=torch.float32).to(device, non_blocking=True)
tensor_test_data = torch.tensor(
    test_data, dtype=torch.float32).to(device, non_blocking=True)

batch_size = 256
train_data_loader = DataLoader(TensorDataset(
    tensor_train_data), batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(TensorDataset(
    tensor_test_data), batch_size=batch_size)


class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


input_size = len(colunas_relevantes)
latent_size = 2
model = Autoencoder(input_size, latent_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_data_loader:
        input_data = batch[0].to(device)
        optimizer.zero_grad()
        reconstructions = model(input_data)
        loss = criterion(reconstructions, input_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_data_loader)}")

model.eval()
with torch.no_grad():
    all_reconstructions = []
    all_losses = []
    for batch in test_data_loader:
        input_data = batch[0].to(device)
        reconstructions = model(input_data)
        loss = criterion(reconstructions, input_data)
        all_reconstructions.append(reconstructions.cpu().numpy())
        all_losses.append(loss.item())


all_reconstructions = np.concatenate(all_reconstructions)
all_losses = np.array(all_losses)
mean_loss = np.mean(all_losses)
std_loss = np.std(all_losses)

threshold = mean_loss + 2 * std_loss


anomalias = all_losses > threshold

plt.figure(figsize=(16, 8))
plt.scatter(range(len(all_losses)), all_losses, c=anomalias,
            cmap='coolwarm', marker='o', s=50, alpha=0.8)
plt.axhline(threshold, color='red', linestyle='dashed',
            label='Limiar de Anomalia')
plt.xlabel('Gastos')
plt.ylabel('Erro de Reconstrução')
plt.title('Detecção de Anomalias com gastos com alimentação dos Deputados')
plt.legend()
plt.colorbar()
plt.show()
