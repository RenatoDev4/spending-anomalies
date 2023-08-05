
from src.data.load_data import load_data
from src.models.train_model import _train, grafico_anomalia


def main():
    """
    Executes the main function.

    This function loads data using the `load_data` function. It then trains a model using the `_train` function, which returns the trained model, criterion, and test data loader. Finally, it calls the `grafico_anomalia` function with the trained model, criterion, and test data loader.

    Parameters:
    None

    Returns:
    None
    """
    df = load_data()
    model, criterion, test_data_loader = _train(df)
    grafico_anomalia(model, criterion, test_data_loader)


if __name__ == '__main__':
    main()
