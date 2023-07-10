import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder

# Define the path to your audio folders
sleep_folder = "Audio\Sleep_audio\sleep"
engaging_folder = "Audio\good oration\engaging"
test_audio="Audio\sample_60_sec_window"

# Define the audio feature extraction parameters
sampling_rate = 44100
n_mfcc = 13
frame_length = 0.025
frame_hop = 0.01

features = []
labels = []

# Function to extract features from audio files in a folder
def extract_features_from_folder(folder, class_label):
    for filename in os.listdir(folder):
        if filename.endswith(".mp3"):
            filepath = os.path.join(folder, filename)

            # Load the audio file
            audio, _ = librosa.load(filepath, sr=sampling_rate)

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=n_mfcc,
                                        hop_length=int(frame_hop * sampling_rate),
                                        n_fft=int(frame_length * sampling_rate))

            # Flatten the MFCC features
            mfcc_flat = mfcc.flatten()

            # Add the features and labels to the lists
            features.append(mfcc_flat)
            labels.append(class_label)

# Extract features from the "sleep" folder
# extract_features_from_folder(sleep_folder, "sleep")

# # Extract features from the "engaging" folder
# extract_features_from_folder(engaging_folder, "engaging")

#extract features from test audio folder
extract_features_from_folder(test_audio, "none")

# Convert the lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Preprocess the features if necessary (e.g., normalization)

# Encode the labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Shuffle the data
random_indices = np.random.permutation(len(features))
features = features[random_indices]
encoded_labels = encoded_labels[random_indices]

# Save the preprocessed features and labels
np.save("Audio\sample_60_features/processed_features.npy", features)
np.save("Audio\sample_60_features/labels.npy", encoded_labels)

# Save the label encoder for later use
np.save("Audio\sample_60_features/label_encoder.npy", label_encoder.classes_)







