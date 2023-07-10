import numpy as np

# Load the preprocessed features and labels
features = np.load("Audio/processed_features.npy", allow_pickle=True)
labels = np.load("Audio/labels.npy", allow_pickle=True)

# Load the label encoder
label_encoder = np.load("Audio/label_encoder.npy", allow_pickle=True)

# Access the loaded data
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)
print("Label encoder classes:", label_encoder)


