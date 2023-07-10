import os
import soundfile as sf
import librosa

# Create a new directory to store the segments
output_folder = "Audio\window_test_samples"
os.makedirs(output_folder, exist_ok=True)

audio, sampling_rate = librosa.load("james_audio.mp3")

# Sliding window parameters
window_size_sec = 5 * 60  # 5 minutes window size in seconds
stride_size_sec = 30  # 30 seconds stride size in seconds

# Convert window and stride size to samples
window_size = int(window_size_sec * sampling_rate)
stride_size = int(stride_size_sec * sampling_rate)

# Iterate over the audio data using sliding window
segments = []
for i in range(0, len(audio), stride_size):
    start = i
    end = start + window_size

    # Make sure the window does not exceed the audio length
    if end > len(audio):
        break

    # Extract the segment using the window indices
    segment = audio[start:end]

    # Append the segment to the list
    segments.append(segment)

    # Generate a filename for the segment
    filename = f"segment_{i // stride_size}.mp3"
    filepath = os.path.join(output_folder, filename)

    # Save the segment as an audio file
    sf.write(filepath, segment, sampling_rate)

    print(f"Segment {i // stride_size + 1} saved as {filepath}")
