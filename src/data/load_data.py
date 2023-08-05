from typing import List

import pandas as pd
from pandas import DataFrame, Series

from src.data import config


def load_data() -> pd.DataFrame:  # Declare function arguments and return type
    """
    Load data from a CSV file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    df = pd.read_csv(config.DATA_PATH).dropna(
        subset=['vlrdocumento'])  # Read data from CSV file
    return df  # Return the loaded data


def filtrar_atividade(df: pd.DataFrame, texto_filtro: int, partidos_indesejados: List[str]) -> pd.DataFrame:
    """
    Filter the given DataFrame based on the provided filter text and unwanted parties.

    Arguments:
    - df: The DataFrame to be filtered.
    - texto_filtro: Index of the filter text in the `config.TEXTO_BUSCAR` dictionary.
    - partidos_indesejados: List of unwanted parties.

    Returns:
    - df_filtrado: The filtered DataFrame.
    """
    filtro = df[config.ATIVIDADE_PARLAMENTAR] == list(
        config.TEXTO_BUSCAR.keys())[texto_filtro]
    df_filtrado = df[filtro]

    return df_filtrado


def calcular_gastos_por_parlamentar(df: DataFrame, coluna_valor: str) -> Series:
    """
    Calculates the total expenses per parliament member.

    Args:
        df: The DataFrame containing the expenses data.
        coluna_valor: The column name representing the expenses values.

    Returns:
        The total expenses per parliament member.
    """
    return df.groupby(config.COLUNA_NOME_PARLAMENTAR)[coluna_valor].sum()


# Carregar os dados
df_carregado = load_data()

# Filtrar atividade parlamentar e partidos indesejados
df_filtrado = filtrar_atividade(df_carregado, 15, config.EXCLUIR_PARTIDOS)

# Calcular gastos por parlamentar e obter os gastos
gasto_parlamentar = calcular_gastos_por_parlamentar(
    df_filtrado, config.COLUNA_VALOR)

top_10_gastos = gasto_parlamentar.nlargest(config.TOP)


def calcular_gastos_com_fornecedor(df, coluna_valor):
    """
    Calculate the total expenses with suppliers.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        coluna_valor (str): The column name containing the expense values.

    Returns:
        pandas.Series: The total expenses with suppliers, grouped by supplier.

    """
    # Group the DataFrame by the supplier column and sum the expense values
    return df.groupby(config.COLUNA_FORNECEDOR)[coluna_valor].sum()


# Calcular gastos por parlamentar e fornecedor e obter os gastos
gasto_com_fornecedor = calcular_gastos_com_fornecedor(
    df_filtrado, config.COLUNA_VALOR)

top_10_gastos_parlamentar = gasto_parlamentar.nlargest(config.TOP)
top_10_gastos_fornecedor = gasto_com_fornecedor.nlargest(config.TOP)
