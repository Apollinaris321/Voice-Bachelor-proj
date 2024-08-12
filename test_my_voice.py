import os

import numpy
import skimage
import torch
import tkinter as tk
import pyaudio
import wave

from PIL import Image
from pydub import AudioSegment
from Alex import LocalAlexNet
from torchvision.transforms import transforms
import numpy as np
import librosa
import librosa.display
from models.SimpleModel import SimpleNet

def generate_mel():
    y, sr = librosa.load("my_word.wav")
    y = np.array(y)
    # fmax highest frequency (in Hz)
    # output shape of tensor is [n_mels = 128, samplerate/hop_length = ~44]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mel_db, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0)  # put low frequencies at the bottom in image
    # img = 255 - img  # invert. make black==more energy
    return img

    try:
        # delete old pictures if exists
        os.remove(path="myWord.png")
    except OSError:
        pass
    skimage.io.imsave("myWord.png", img)


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def padd_to_len_audio_file(pad_ms: int = 2000, path: str = ""):
    audio = AudioSegment.from_wav(f"{path}")
    silence = AudioSegment.silent(duration=pad_ms - len(audio))
    padded = audio + silence
    padded = padded.set_channels(1)
    padded.export(f"{path}", format='wav')



def record_audio():

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    SAMPLE_RATE = 44100

    p = pyaudio.PyAudio()
    # Open a new stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=1
    )

    print("start recording in test_my_voice...")

    frames = []
    seconds = 2
    for i in range(0, int(SAMPLE_RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("recording stopped")

    stream.stop_stream()
    stream.close()

    if not os.path.exists("my_word.wav"):
        with wave.open("my_word.wav", "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # wf.setnchannels(1)  # mono
            # wf.setsampwidth(2)  # number of bytes per sample
            # wf.setframerate(44100)  # samples per second
            # wf.writeframes(b'')

    wf = wave.open("my_word.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    #padd_to_len_audio_file(2000, "my_word.wav")


# def generate_mel():
#     y, sr = librosa.load("my_word.wav")
#     return generate_mel(waveform=y, sample_rate=sr)
#     img = Image.open("myWord.png")
#
#     return img


def inference(img):
    global model
    img = img.copy()


    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize((128, 87), antialias=True),
        transforms.RandomVerticalFlip(p=1)
    ])
    with torch.no_grad():
        t = trans(img)
        # 1, 128, 87
        t = t.unsqueeze(0)
        outputs = model(t)
        _, predicted = torch.max(outputs.data, 1)


        # Print labels with their corresponding probabilities
        print(f"predicted: {classes[predicted]}, prob: {outputs}")


def check_recording():
    record_audio()
    img = generate_mel()
    inference(img)


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
    "war",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
# Load the trained model
#model_path = "../goodModel.pth"
model_path = "../generated_models/model__v0.0.pth"
#model = LocalAlexNet(11)
from models.Net import Net3
model = Net3(len(classes))
model.load_state_dict(torch.load(model_path))
model.eval()

# Set up the GUI
root = tk.Tk()
root.title("Voice Classification Demo")
root.geometry("500x200")

record_button = tk.Button(root, text="Record", command=check_recording)
record_button.pack()
root.mainloop()
