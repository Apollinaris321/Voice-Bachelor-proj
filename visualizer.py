import os
import shutil
import librosa
import numpy as np
import librosa.display
from matplotlib import pyplot as plt


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
audio_files_dir = os.path.join(project_root, "image_dataset")

audio_files_dir2 = os.path.join(project_root, "results/airplane/0_airplane_0.wav")
audio_data, sample_rate = librosa.load(audio_files_dir2, sr=None)  # sr=None keeps the original sample rate

plt.figure(figsize=(14, 5))
librosa.display.waveshow(y=audio_data, sr=sample_rate,  color="blue")
plt.title('Waveform of Audio')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()

#wavefile
#gen spectrogram
# mel spectrogram

exit()

word = "airplane"


added_voices = []
for root, dirs, files in os.walk(audio_files_dir):
    for file in files:
        voice = file.split("_")[0:2]
        voice = voice[0] + voice[1]
        if voice not in added_voices:
            source_path = os.path.join(audio_files_dir, file)
            destination_path = os.path.join(project_root, "filtered")
            shutil.copy2(source_path, destination_path)
            added_voices.append(voice)


def generate_mel(audio, results_path):
    for i, (waveform, sample_rate, voice, word) in enumerate((audio)):
        sr = sample_rate
        y = np.array(waveform)
        # fmax highest frequency (in Hz)
        # output shape of tensor is [n_mels = 128, samplerate/hop_length = ~44]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # min-max scale to fit inside 8-bit range
        img = scale_minmax(mel_db, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        # img = 255 - img  # invert. make black==more energy

        image_path = f'{results_path}/{voice}_{word}_{i}.png'

        try:
            # delete old pictures if exists
            os.remove(path=image_path)
        except OSError:
            pass



def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

# take an audio file path
# generate:
# waveform
# spectrogram
# mel spectrogram
# check the sample rate