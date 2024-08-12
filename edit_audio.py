import os
import random
import numpy
import skimage
import torch
import torchaudio
import librosa
import numpy as np
import wave
import glob
import tqdm

import matplotlib.pyplot as plt


def create_path(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")


def load_base_vocab(audio_path):
    audio = []
    for dir in os.listdir(audio_path):
        for file in os.listdir(os.path.join(audio_path, dir)):
            voice, word, _ = file.split("_")
            waveform, samplerate = torchaudio.load(audio_path + "\\" + dir + "\\" + file)
            audio.append([waveform, samplerate, voice, word])
    return audio


def load_samples(audio_path, words):
    audio = []
    for dir in os.listdir(audio_path):
        if dir in words or len(words) == 0:
            for file in os.listdir(os.path.join(audio_path, dir)):
                voice, word, _ = file.split("_")
                waveform, samplerate = torchaudio.load(audio_path + "\\" + dir + "\\" + file)
                audio.append([waveform, samplerate, voice, word])
    return audio


# do this in pydub as well
def add_random_padding(audio, amount):
    new_audio = []
    for waveform, sample_rate, voice, word in tqdm.tqdm(audio, desc="Adding Random Padding"):
        if len(waveform[0]) > sample_rate * 2:
            print("error! file too long!")
        else:
            for i in range(0, amount):
                padding = sample_rate * 2 - len(waveform[0])
                index = random.randint(0, padding)
                before = torch.zeros(index)
                after = torch.zeros(padding - index)

                transformed_waveform = torch.cat((before, waveform[0], after), 0)
                transformed_waveform = transformed_waveform.unsqueeze(0)

                arr = torchaudio.functional.resample(transformed_waveform, orig_freq=sample_rate, new_freq=22050)
                new_audio.append([arr, sample_rate, voice, word])
    return new_audio


def add_noise(audio):
    noise_path = "C:/Users/blura/Desktop/Voice Bachelor proj/noise"
    new_audio = []
    for waveform, sample_rate, voice, word in tqdm.tqdm(audio, desc="Adding noise"):
        for i, filename in enumerate(os.listdir(noise_path)):
            noise_wav, noise_sample = torchaudio.load(f"{noise_path}/{filename}")
            noise_wav = torch.mean(noise_wav, dim=0, keepdim=True)
            new_noise = torchaudio.functional.resample(waveform=noise_wav, orig_freq=noise_sample,
                                                       new_freq=sample_rate, )

            quiet_noises = ["faucet", "people2", "space", "airplane", "fan", "wind"]
            if filename.split(".")[0] in quiet_noises:
                gain = 0.2
            else:
                gain = 0.4
            vol_transform = torchaudio.transforms.Vol(gain_type="amplitude", gain=gain)
            new_noise = vol_transform(new_noise)

            overlay = waveform + new_noise[:, :len(waveform[0])]
            new_audio.append([overlay, sample_rate, voice, word])
    return new_audio


def trim(audio):
    for i, (waveform, sample_rate, voice, word) in enumerate(audio):
        one_second_length = sample_rate
        truncated_waveform = waveform[:, :one_second_length * 2]
        audio[i] = [truncated_waveform, sample_rate, voice, word]


def generate_mel(audio, results_path):
    for i, (waveform, sample_rate, voice, word) in enumerate(tqdm.tqdm(audio, desc="generating Mels")):
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

        image_path = f'{results_path}/{voice}_{word}_{i}.png'

        try:
            # delete old pictures if exists
            os.remove(path=image_path)
        except OSError:
            pass

        skimage.io.imsave(image_path, img)


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled



def edit_audio_main(word):
    #TODO path und results_path müssen übergeben werden
    # der erste path kann gleich bleiben weil die wörter einfach nur hinzugefügt werden
    path = r'C:\Users\blura\Desktop\Voice Bachelor proj\results'
    path_base = r'C:\Users\blura\Desktop\Voice Bachelor proj\base_vocabulary'
    #der path kann wegen des geringen aufwands temp sein und immer wieder gelöscht oder überschrieben werden oder sogar
    # gar nicht erst gespeichert werden sondern sofort ans training übergeben werden

    results_path = "C:/Users/blura/Desktop/Voice Bachelor proj/image_dataset"

    # Specify the path to the folder containing the images
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the relative path to the folder inside your project
    folder_name = 'image_dataset'
    folder_path = os.path.join(ROOT_DIR, folder_name)

    # List all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    create_path(results_path)
    words = []
    if word is not "":
        words.append(word)

    # give the results path where all audio is stored and only pick the words that are mentioned in the array
    audio = load_samples(path, words)
    audio_base_vocab = load_base_vocab(path_base)

    audio.extend(audio_base_vocab)
    print("fin loading")

    audio = add_random_padding(audio, 30)
    print("fin padding")

    #audio = add_noise(audio)
    print("fin add noise")

    trim(audio)
    print("fin trim")

    generate_mel(audio, results_path)
    print("fin mel")