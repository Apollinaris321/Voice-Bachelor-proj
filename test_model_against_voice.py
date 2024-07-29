import os

import numpy
import torch

from torchvision.transforms import transforms
import numpy as np
import librosa
import librosa.display

from Alex import LocalAlexNet


def generate_mel(waveform, sample_rate):
    sr = sample_rate
    y = np.array(waveform)
    # fmax highest frequency (in Hz)
    # output shape of tensor is [n_mels = 128, samplerate/hop_length = ~44]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mel_db, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0)  # put low frequencies at the bottom in image
    # img = 255 - img  # invert. make black==more energy

    return img


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
# Load the trained model
model_path = "../goodModel.pth"
model = LocalAlexNet(11)
model.load_state_dict(torch.load(model_path))
model.eval()

counter = 0
correct = 0

for file in os.listdir("../test_aufnahmen"):

    y, sr = librosa.load("../test_aufnahmen/" + file)
    generate_mel(waveform=y, sample_rate=sr)
    img = generate_mel(y,sr)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((227, 227), antialias=True),
        transforms.RandomVerticalFlip(p=1)
    ])

    with torch.no_grad():
        t = trans(img.copy())
        t = t.unsqueeze(0)
        outputs = model(t)
        _, predicted = torch.max(outputs.data, 1)

        classes = [
            "ukraine",
            "germany",
            "china",
            "russia",
            "sun",
            "airplane",
            "beer",
            "europe",
            "peace",
            "tank",
            "war"
        ]

        # Print labels with their corresponding probabilities
        print(f"predicted: {classes[predicted]}, word: {file.split('_')[0]}")
        counter += 1
        if classes[predicted] == file.split('_')[0]:
            correct += 1

print("counter: " , counter , ", correct: " , correct)