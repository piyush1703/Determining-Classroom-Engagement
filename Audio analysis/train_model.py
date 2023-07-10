import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = np.load("Audio/processed_features.npy",allow_pickle=True)
Y = np.load("Audio/labels.npy",allow_pickle=True)
rows, columns = X.shape[0],X[0].shape[0]
array_2d = np.empty((rows, columns), dtype=object)

for i in range(rows):
    for j in range(X[i].shape[0]):
        array_2d[i][j] = X[i][j]
array_2d = array_2d.astype(np.float32)
X=array_2d

split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Reshape the data for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert labels to one-hot encoded vectors
Y_train = keras.utils.to_categorical(Y_train, num_classes=2)
Y_test = keras.utils.to_categorical(Y_test, num_classes=2)

# CNN model
model = keras.Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))  # Assuming 2 output classes (binary classification)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 8
epochs = 10

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test) )

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

#predictions on test audio
new_samples = np.load("Audio\\test audios\processed_features.npy",allow_pickle=True)

rows, columns = new_samples.shape[0],new_samples[0].shape[0]
array_2 = np.empty((rows, columns), dtype=object)

for i in range(rows):
    for j in range(columns):
        array_2[i][j] = new_samples[i][j]
array_2 = array_2.astype(np.float32)
new_samples=array_2
new_samples = new_samples.reshape(new_samples.shape[0], new_samples.shape[1], 1)
predictions = model.predict(new_samples)

# Convert probabilities to class labels
class_labels = ['engaging', 'sleep-inducing']
predicted_labels = [class_labels[int((pred.argmax(axis=0)))] for pred in predictions]
print(predicted_labels)

