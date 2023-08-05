from load_data import top_10_gastos_fornecedor, top_10_gastos_parlamentar

from src.data.build_features import (plot_grafico_fornecedor,
                                     plot_grafico_parlamentar)


def main():
    """
    Executes the main function.

    This function loads data using the `load_data` function. It then trains a model using the `_train` function, which returns the trained model, criterion, and test data loader. Finally, it calls the `grafico_anomalia` function with the trained model, criterion, and test data loader.

    Parameters:
    None

    Returns:
    None
    """
    plot_grafico_fornecedor(top_10_gastos_fornecedor)
    plot_grafico_parlamentar(top_10_gastos_parlamentar)


if __name__ == '__main__':
    main()
