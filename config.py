import torch


class Config:
    def __init__(self):
        self.epochs = 1000
        self.learning_rate = 0.00001
        self.batch_size = 64
        self.dropout = 0.2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data_file = r'./data/train_data.csv'
        self.val_data_file = r'./data/val_data.csv'
        self.model_save_path = r'./model'
        self.input_param = ['H', 'a', 'e', 'i', 'Orbital_period', 'Perihelion_dist', 'Aphelion_dist',
                            'Semilatus_rectum', 'Type', 'Type_quality', 'Initial_albedo']
        self.h_mean, self.h_std = 14.7620476063665, 1.51174405432132
        self.albedo_mean, self.albedo_std = 0.123025454759187, 0.0929380876992498
        self.diameter_mean, self.diameter_std = 7.03224317854704, 16.1240995317874
