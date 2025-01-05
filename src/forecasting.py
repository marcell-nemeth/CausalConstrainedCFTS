import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import matplotlib.pyplot as plt

class TimeSeriesDataLoader:
    def __init__(self, file_path: str, target_column: str):
        """
        Initialize the data loader
        
        Args:
            file_path (str): Path to the CSV file
            target_column (str): Name of the target variable column
        """
        self.file_path = file_path
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess the data"""
        # Read the CSV file
        df = pd.read_csv(self.file_path)
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Scale the features and target
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))
        
        return X_scaled, y_scaled.ravel()

def create_sequences(
    X: pd.DataFrame,
    y: np.ndarray,
    back_horizon: int,
    forecast_horizon: int,
    train_size: float = 0.8
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Create sequences for time series forecasting
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (np.ndarray): Target variable
        back_horizon (int): Number of past time steps to consider
        forecast_horizon (int): Number of future time steps to predict
        train_size (float): Proportion of data to use for training
        
    Returns:
        Tuple containing X_train, y_train, X_test, y_test as sequences
    """
    X_sequences = []
    y_sequences = []
    
    # Create sequences
    for i in range(len(X) - back_horizon - forecast_horizon + 1):
        X_sequences.append(X.iloc[i:(i + back_horizon)].values)
        y_sequences.append(y[i + back_horizon:i + back_horizon + forecast_horizon])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Split into train and test
    train_size = int(len(X_sequences) * train_size)
    
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    X_test = X_sequences[train_size:]
    y_test = y_sequences[train_size:]
    
    return X_train, y_train, X_test, y_test 

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use only the last output for prediction
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str
) -> List[float]:
    """Train the LSTM model"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def plot_losses(train_losses: List[float], val_losses: List[float]):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()