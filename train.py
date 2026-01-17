import os
import numpy as np
import torch
import torch.nn as nn
from dataloader import get_dataloaders
from model import AadNet
from config import Config
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-10):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon 
    
    def forward(self, pred, target):
        abs_percent_error = torch.abs((pred - target) / (target + self.epsilon))
        return torch.mean(abs_percent_error)


def train_and_validate(train_loader, val_loader, device, cfg):
    model = AadNet(input_param=cfg.input_param,
                   dropout=cfg.dropout,
                   device=device,
                   config=cfg)
    model.to(device)

    albedo_criterion = nn.MSELoss()
    diameter_criterion = MAPELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    log_file = os.path.join(cfg.model_save_path, f'training_log.txt')
    with open(log_file, "w") as log:
        log.write("Epoch,Train Loss,Val Loss\n")

        train_losses, val_losses = [], []
        best_val_loss = float('inf')

        for epoch in range(cfg.epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                albedo_pred, diameter_pred = model(inputs)

                albedo_loss = albedo_criterion(albedo_pred, labels[:, 0].unsqueeze(1))
                diameter_loss = diameter_criterion(diameter_pred, labels[:, 1].unsqueeze(1))
                total_loss = 0.6 * albedo_loss + 0.4 * diameter_loss

                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    albedo_pred, diameter_pred = model(inputs)
                    albedo_loss = albedo_criterion(albedo_pred, labels[:, 0].unsqueeze(1))
                    diameter_loss = diameter_criterion(diameter_pred, labels[:, 1].unsqueeze(1))
                    total_val_loss = 0.6 * albedo_loss + 0.4 * diameter_loss
                    val_loss += total_val_loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            log.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f}\n")
            print(f'Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                model_save_file = os.path.join(cfg.model_save_path, f'AadNet.pth')
                torch.save(best_model_state, model_save_file)

        plt.figure()
        epochs = range(1, cfg.epochs + 1)
        plt.plot(epochs, train_losses, label='Training loss')
        plt.plot(epochs, val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(cfg.model_save_path, f'loss_curve.png'), dpi=600)
        plt.close()

    model.load_state_dict(best_model_state)
    return model, best_val_loss


def model_training(cfg):
    device = cfg.device
    all_labels = []
    all_predictions = []

    print(f'Start training')
    train_loader, val_loader = get_dataloaders(train_data_file=cfg.train_data_file, val_data_file=cfg.val_data_file,
                                               batch_size=cfg.batch_size, input_param=cfg.input_param)

    best_model, best_val_loss = train_and_validate(train_loader=train_loader, val_loader=val_loader, device=device, cfg=cfg)

    best_model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            albedo_pred, diameter_pred = best_model(inputs)
            albedo_denorm = albedo_pred * cfg.albedo_std + cfg.albedo_mean
            diameter_denorm = diameter_pred * cfg.diameter_std + cfg.diameter_mean
            labels_denorm = labels * torch.tensor([cfg.albedo_std, cfg.diameter_std], device=device) + \
                            torch.tensor([cfg.albedo_mean, cfg.diameter_mean], device=device)
            all_labels.append(labels_denorm.cpu())
            all_predictions.append(torch.stack([albedo_denorm, diameter_denorm], dim=1).cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_predictions = torch.cat(all_predictions).numpy()

    albedo_rmse = np.sqrt(mean_squared_error(all_labels[:, 0], all_predictions[:, 0]))
    albedo_mae = mean_absolute_error(all_labels[:, 0], all_predictions[:, 0])
    albedo_mape = np.mean(np.abs((all_labels[:, 0] - all_predictions[:, 0]) / (all_labels[:, 0] + 1e-10))) * 100
    albedo_r2 = r2_score(all_labels[:, 0], all_predictions[:, 0])

    diameter_rmse = np.sqrt(mean_squared_error(all_labels[:, 1], all_predictions[:, 1]))
    diameter_mae = mean_absolute_error(all_labels[:, 1], all_predictions[:, 1])
    diameter_mape = np.mean(np.abs((all_labels[:, 1] - all_predictions[:, 1]) / (all_labels[:, 1] + 1e-10))) * 100
    diameter_r2 = r2_score(all_labels[:, 1], all_predictions[:, 1])

    with open(os.path.join(cfg.model_save_path, 'metrics.txt'), "w") as log:
        log.write(f'Baet validation loss (MSE): {best_val_loss:.4f}\n')
        log.write(f'Albedo - RMSE: {albedo_rmse:.4f}, MAE: {albedo_mae:.4f}, MAPE: {albedo_mape:.2f}%, R2: {albedo_r2:.4f}\n')
        log.write(f'Diameter - RMSE: {diameter_rmse:.4f}, MAE: {diameter_mae:.4f}, MAPE: {diameter_mape:.2f}%, R2: {diameter_r2:.4f}\n')

    print(f'Training completed. Best validation loss: {best_val_loss:.4f}')
    print(f'Albedo metrics: RMSE: {albedo_rmse:.4f}, MAE: {albedo_mae:.4f}, MAPE: {albedo_mape:.2f}%, R2: {albedo_r2:.4f}')
    print(f'Diameter metrics: RMSE: {diameter_rmse:.4f}, MAE: {diameter_mae:.4f}, MAPE: {diameter_mape:.2f}%, R2: {diameter_r2:.4f}')


if __name__ == "__main__":
    config = Config()
    parameters_dict = {}
    os.makedirs(config.model_save_path, exist_ok=True)
    model_training(config)
