from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data import config


def split(df: pd.DataFrame, test_size: float = config.TEST_SIZE, random_state: int = config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the input DataFrame into train and test datasets.
    Args:
        df (pd.DataFrame): The input DataFrame.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to config.TEST_SIZE.
        random_state (int, optional): The seed used by the random number generator. Defaults to config.RANDOM_STATE.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test datasets.
    """
    colunas_relevantes = [config.COLUNA_VALOR]
    df = df[colunas_relevantes]
    train_data, test_data = train_test_split(
        df, test_size=test_size, random_state=random_state)
    colunas_relevantes = [config.COLUNA_VALOR]
    return train_data, test_data


def normalizacao(df: np.ndarray) -> np.ndarray:
    """
    Normalize the given data using MinMaxScaler.

    Args:
        df (np.ndarray): The input data to be normalized.

    Returns:
        np.ndarray: The normalized data.
    """
    scraler_minMax = MinMaxScaler()
    dados_normalizados = scraler_minMax.fit_transform(df)
    return dados_normalizados


def convert(train_data: List[List[float]], test_data: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert the train_data and test_data to tensor format.
    Args:
        train_data (List[List[float]]): The training data.
        test_data (List[List[float]]): The testing data.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the train_data and test_data tensors.
    """
    dados_normalizados_treino = normalizacao(train_data)
    dados_normalizados_teste = normalizacao(test_data)
    tensor_train_data = torch.tensor(
        dados_normalizados_treino, dtype=torch.float32).to(config.DEVICE, non_blocking=True)
    tensor_test_data = torch.tensor(
        dados_normalizados_teste, dtype=torch.float32).to(config.DEVICE, non_blocking=True)
    return tensor_train_data, tensor_test_data


def create_data_loaders(tensor_train_data: torch.Tensor, tensor_test_data: torch.Tensor, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and testing data.

    Args:
        tensor_train_data: The training data as a tensor.
        tensor_test_data: The testing data as a tensor.
        batch_size: The batch size for the data loaders.

    Returns:
        A tuple containing the training data loader and testing data loader.
    """
    train_data_loader = DataLoader(TensorDataset(
        tensor_train_data), batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(TensorDataset(
        tensor_test_data), batch_size=batch_size)
    return train_data_loader, test_data_loader


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


def _train(df: pd.DataFrame) -> Tuple[Autoencoder, nn.MSELoss, DataLoader]:
    """
    Trains an autoencoder model on the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[Autoencoder, nn.MSELoss, DataLoader]: A tuple containing the trained autoencoder model, the mean squared error loss function, and the data loader for the test data.
    """
    train_data, test_data = split(df)
    tensor_train_data, tensor_test_data = convert(train_data, test_data)
    train_data_loader, test_data_loader = create_data_loaders(
        tensor_train_data, tensor_test_data, batch_size=config.BATCH_SIZE)

    input_size = len([config.COLUNA_VALOR])
    latent_size = 2
    model = Autoencoder(input_size, latent_size).to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 5

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_data_loader:
            input_data = batch[0].to(config.DEVICE)
            optimizer.zero_grad()
            reconstructions = model(input_data)
            loss = criterion(reconstructions, input_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_data_loader)}")

    return model, criterion, test_data_loader


def grafico_anomalia(model: torch.nn.Module, criterion: torch.nn.Module, test_data_loader: torch.utils.data.DataLoader) -> None:
    """
    Generates a scatter plot to detect anomalies in the reconstructions of the given model.

    Args:
        model (torch.nn.Module): The trained model to use for generating the reconstructions.
        criterion (torch.nn.Module): The loss criterion used to calculate the reconstruction error.
        test_data_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
        None
    """
    model.eval()
    with torch.no_grad():
        all_reconstructions = []
        all_losses = []
        for batch in test_data_loader:
            input_data = batch[0].to(config.DEVICE)
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
