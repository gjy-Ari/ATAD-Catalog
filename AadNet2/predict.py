import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model import AadNet
from config import Config


class AsteroidDataset(Dataset):
    normalize_params = {
        'H': {'mean': 14.7620476063665, 'std': 1.51174405432132},
        'a': {'mean': 2.7721563313262, 'std': 0.720819608720993},
        'e': {'mean': 0.149539104351581, 'std': 0.0822899908797655},
        'i': {'mean': 9.86309327109933, 'std': 6.54312838768382},
        'Orbital_period': {'mean': 4.69657547197742, 'std': 4.05508125584656},
        'Perihelion_dist': {'mean': 2.36562636375793, 'std': 0.639333288239313},
        'Aphelion_dist': {'mean': 3.17868629889447, 'std': 0.898001437616106},
        'Semilatus_rectum': {'mean': 1.34717066355784, 'std': 0.341941114503914},
        'Type': {'mean': 3.27315781139788, 'std': 1.41895182181395},
        'Type_quality': {'mean': 1.5385279765075, 'std': 0.595323151225717},
        'Initial_albedo': {'mean': 0.178413826311016, 'std': 0.084205099781631},
        'albedo': {'mean': 0.123025454759187, 'std': 0.0929380876992498},
        'diameter': {'mean': 7.03224317854704, 'std': 16.1240995317874}
    }

    def __init__(self, csv_file, input_param=None):
        self.data = pd.read_csv(csv_file, dtype={'Number': str})
        self.input_param = input_param

        self._add_initial_albedo()
        self._map_type_to_number()
        self._normalize_data()

    def _map_type_to_number(self):
        type_mapping = {'A': 1, 'C': 2, 'D': 3, 'S': 4, 'V': 5, 'X': 6}
        self.data['Type'] = self.data['Type'].astype(str).str.strip().map(type_mapping).astype(int)

    def _add_initial_albedo(self):
        type_albedo_map = {
            'A': 0.2848, 'C': 0.0949, 'D': 0.1523,
            'S': 0.2674, 'V': 0.3540, 'X': 0.1615
        }
        self.data['Initial_albedo'] = self.data['Type'].astype(str).str.strip().map(type_albedo_map)

    def _normalize_data(self):
        for param in self.input_param:
            if param in self.normalize_params:
                mean = self.normalize_params[param]['mean']
                std = self.normalize_params[param]['std']
                self.data[param] = (self.data[param] - mean) / std

    def get_processed_data(self):
        return self.data.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        inputs = np.array([self.data.iloc[idx][param].astype(np.float32) for param in self.input_param])
        return torch.tensor(inputs, dtype=torch.float32)


def get_dataloaders(dataset, batch_size=64):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


def prediction(input_path, model_path, input_param, output_path, config):

    test_dataset = AsteroidDataset(csv_file=input_path, input_param=input_param)
    processed_data = test_dataset.get_processed_data()
    test_loader = get_dataloaders(dataset=test_dataset, batch_size=1)

    aad_model = AadNet(input_param=input_param, dropout=config.dropout, device=config.device, config=config)
    aad_model.load_state_dict(torch.load(model_path, map_location=config.device))
    aad_model.to(config.device)
    aad_model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(config.device)
            albedo_pred, diameter_pred = aad_model(inputs)
            predictions.append([albedo_pred.item(), diameter_pred.item()])
    predictions = np.vstack(predictions)

    def denorm(value, param):
        mean = AsteroidDataset.normalize_params[param]['mean']
        std = AsteroidDataset.normalize_params[param]['std']
        return value * std + mean

    albedo_pred = denorm(predictions[:, 0], 'albedo')
    diameter_pred = denorm(predictions[:, 1], 'diameter')

    h_mean = AsteroidDataset.normalize_params['H']['mean']
    h_std = AsteroidDataset.normalize_params['H']['std']
    processed_data['H_original'] = processed_data['H'] * h_std + h_mean
    H_original = processed_data['H_original'].values

    def bowell_diameter(H, albedo):
        return 1329 * (10** (-H / 5)) / np.sqrt(albedo + 1e-10)

    def bowell_albedo(H, diameter):
        return (1329 * (10 **(-H / 5)) / (diameter + 1e-10))** 2

    final_albedo, final_diameter = [], []
    for i in range(len(albedo_pred)):
        alb, diam = albedo_pred[i], diameter_pred[i]
        H = H_original[i]
        num = processed_data['Number'].iloc[i]

        alb_valid = 0.02 <= alb <= 0.7
        diam_valid = 0.01 <= diam <= 1000

        if alb_valid and diam_valid:
            final_albedo.append(round(alb, 3))
            final_diameter.append(round(diam, 3))
        elif not alb_valid and diam_valid:
            print(f"{num}: Albedo prediction exceeds limits, switching to Bowell relation.")
            final_albedo.append(round(bowell_albedo(H, diam), 3))
            final_diameter.append(round(diam, 3))
        elif alb_valid and not diam_valid:
            print(f"{num}: Diameter prediction exceeds limits, switching to Bowell relation.")
            final_albedo.append(round(alb, 3))
            final_diameter.append(round(bowell_diameter(H, alb), 3))
        else:
            print(f"{num}: Both predictions exceed limits, switching to Bowell relation.")
            traditional_alb = processed_data['Initial_albedo'].iloc[i]
            final_albedo.append(round(traditional_alb, 3))
            final_diameter.append(round(bowell_diameter(H, traditional_alb), 3))

    result_df = pd.DataFrame({
        'Number': processed_data['Number'],
        'Albedo': final_albedo,
        'Diameter': final_diameter
    })
    result_df.to_csv(output_path, index=False)
    print(f"Prediction completed.")
    return result_df


def main():
    config = Config()
    input_path = r'./predict/inputs.csv'
    model_path = r'./model/AadNet2.pth'
    output_path = r'./predict/prediction_results.csv'
    input_param = 'H-a-e-i-Orbital_period-Perihelion_dist-Aphelion_dist-Semilatus_rectum-Type-Type_quality-Initial_albedo'.split('-')

    prediction(
        input_path=input_path,
        model_path=model_path,
        input_param=input_param,
        output_path=output_path,
        config=config
    )


if __name__ == "__main__":
    main()
