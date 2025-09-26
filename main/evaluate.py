import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import os
import pandas as pd
import copy

from .config import Config
from .genome import Genome

def evaluate(id: str, genome: Genome, config: Config):
    net = genome.build_net()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1)

    train_indices = list(range(0, 50000))
    val_indices = list(range(50000, 60000))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_full = datasets.MNIST(root=config.data_path, train=True, transform=transform, download=False)

    train_set = Subset(train_full, train_indices)
    val_set = Subset(train_full, val_indices)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    best_loss = float("inf")
    for epoch in range(config.epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        net.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).long()
            optimiser.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * inputs.size(0)

            _, predicted = output.max(1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_rmse = 0.0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).long()
                output = net(inputs)
                loss = criterion(output, targets)

                val_loss += loss.item() * inputs.size(0)

                _, predicted = output.max(1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    'state_dict': net.state_dict(),
                    'genome': genome
                }, 
                config.model_path + f"{id}.pt"
            )
    
    net.load_state_dict(torch.load(config.model_path + f"{id}.pt", weights_only=False)['state_dict'])
    genotype = net.export_genotype()
    child_genome = copy.deepcopy(genome.update_genome(genotype))
    return (
        child_genome,
        {
            "loss": val_loss / len(val_loader),
            "accuracy": val_correct / val_total
        }
    )