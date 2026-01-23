===========
Usage
===========
Training: Execute train.py to train the model. Input parameters require Z-score normalization.

Prediction: Execute predict.py to estimate the albedo and diameter of asteroids. Unnormalized data can be used directly for ease of application.


===========
Folders
===========
data: Contains the datasets used for training and validation, including both unnormalized and normalized versions.

model: Contains AadNet2.pth, the trained model for application.

predict: Contains input data and the resulting outputs for constructing the ATAD catalog by applying the AadNet2.pth model.


===========
Requirements
===========
torch==2.0.1
scikit_learn==1.4.1.post1
pandas==2.3.3
numpy==1.25.2
matplotlib==3.8.0




