import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from .config import Config
from .genome import Genome

def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

def evaluate(id: str, genome: Genome, config: Config):
    net = genome.build_net()

    criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1)

    y_column = ""

    train_df = pd.read_csv(config.train_data_path)
    train_y = torch.tensor(train_df[y_column].values, dtype=torch.float32)
    train_x = torch.tensor(train_df.drop(columns=[y_column]).values, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if config.val_data_path:
        val_df = pd.read_csv(config.val_data_path)
        val_y = torch.tensor(val_df[y_column].values, dtype=torch.float32)
        val_x = torch.tensor(val_df.drop(columns=[y_column]).values, dtype=torch.float32)

        val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    best_loss = float("inf")
    for epoch in range(config.epochs):
        train_loss = 0.0
        train_rmse = 0.0
        val_loss = None
        val_rmse = None

        net.train()
        for batch in train_dataloader:
            optimiser.zero_grad()
            output = net(batch[0])
            loss = criterion(output, batch[1])
            loss.backward()
            optimiser.step()

            train_loss += loss.item()
            train_rmse += rmse(output, batch[1]).item()

        scheduler.step()

        if config.val_data_path:
            net.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_rmse = 0.0
                for batch in val_dataloader:
                    output = net(batch[0])
                    loss = criterion(output, batch[1])
                    val_loss += loss.item()
                    val_rmse += rmse(output, batch[1]).item()

        if val_loss:
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    {
                        'state_dict': net.state_dict(),
                        'genome': genome
                    }, 
                    config.model_path + f"{id}.pt"
                )
        else:
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(
                    {
                        'state_dict': net.state_dict(),
                        'genome': genome
                    }, 
                    config.model_path + f"{id}.pt"
                )
    
    net.load_state_dict(torch.load(config.model_path + f"{id}.pt")['state_dict'])

    if config.val_data_path:
        return {
            "loss": val_loss / len(val_dataloader),
            "rmse": val_rmse / len(val_dataloader)
        }
    else:
        return {
            "loss": train_loss / len(train_dataloader),
            "rmse": train_rmse / len(train_dataloader)
        }