
import locale

import matplotlib.pyplot as plt
from config import ATIVIDADE_PARLAMENTAR, TOP
from load_data import (df_filtrado, top_10_gastos_fornecedor,
                       top_10_gastos_parlamentar)

# Configurar o locale para formatar os valores em reais
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

valor_atividade_parlamentar = df_filtrado.iloc[0][ATIVIDADE_PARLAMENTAR]


def plot_grafico_parlamentar(data):
    """
    Plots a bar graph of the parliamentary expenses.

    Parameters:
        - data (pandas.Series): A pandas Series containing the total expenses for each parliament member.

    Returns:
        None
    """
    plt.figure(figsize=(16, 8))
    colors = plt.cm.tab20.colors[:10]
    ax = plt.bar(data.index, data.values, color=colors)
    plt.ylabel('Total Gasto (R$)')
    plt.title(
        f'Top {TOP} Deputados com mais gastos com ({valor_atividade_parlamentar})')

    plt.gca().yaxis.set_major_formatter(locale.currency)

    plt.legend(ax, data.index, title='Parlamentares',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    for bar in ax:
        height = bar.get_height()
        plt.annotate(locale.currency(height), xy=(bar.get_x()
                     + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', rotation=40)

    plt.xticks([])
    plt.xlabel('')
    plt.ylabel('Total Gasto (R$)')

    plt.tight_layout()
    plt.show()


def plot_grafico_fornecedor(data):
    """
    Generates a bar chart to plot the total amount spent by each supplier.

    Args:
        data (pandas.Series): A pandas Series object containing the total amount spent by each supplier.

    Returns:
        None
    """
    plt.figure(figsize=(16, 8))
    colors = plt.cm.tab20.colors[:10]
    ax = plt.bar(data.index, data.values, color=colors)
    plt.ylabel('Total Gasto (R$)')
    plt.title(
        f'Top {TOP} Fornecedores que mais receberam ({valor_atividade_parlamentar})')

    plt.gca().yaxis.set_major_formatter(locale.currency)

    plt.legend(ax, data.index, title='Fornecedores',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    for bar in ax:
        height = bar.get_height()
        plt.annotate(locale.currency(height), xy=(bar.get_x()
                     + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', rotation=40)

    plt.xticks([])
    plt.xlabel('')
    plt.ylabel('Total Gasto (R$)')

    plt.tight_layout()
    plt.show()
