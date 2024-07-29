import librosa
import numpy as np
import wave

import matplotlib.pyplot as plt

# plane
#path = r'C:\Users\blura\Desktop\Voice Bachelor proj\results\airplane\aiExplained_airplane_0.wav'
#y, sr = librosa.load(path)
#
#D = librosa.stft(y)  # STFT of y
#S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#
#fig, ax = plt.subplots()
#img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
#ax.set(title='Stft logarithmic frequency axis')
#fig.colorbar(img, ax=ax, format="%+2.f dB")
#
#fig, ax = plt.subplots()
#M = librosa.feature.melspectrogram(y=y, sr=sr)
#M_db = librosa.power_to_db(M, ref=np.max)
#img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax)
#ax.set(title='Mel spectrogram')
#fig.colorbar(img, ax=ax, format="%+2.f dB")
#
#plt.show()


# plt.figure()
# librosa.display.specshow(stft_db, x_axis='time', y_axis='log', sr=sr)
# plt.colorbar()
# plt.title('stft spectrogram')
# plt.show()

# mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)
# mel_db = librosa.power_to_db(mel, ref=np.max)

## min-max scale to fit inside 8-bit range
# img = scale_minmax(mel_db, 0, 255).astype(numpy.uint8)
# img = numpy.flip(img, axis=0)  # put low frequencies at the bottom in image
##img = 255 - img  # invert. make black==more energy

# plt.figure()
# librosa.display.specshow(img, x_axis='time', y_axis='mel', sr=sr, fmax=sr/2)
# plt.colorbar()
# plt.title('mel spectrogram')
# plt.show()

#plt.figure()
#plt.plot(waveform)
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.show()
#
#spf = wave.open('aiExplained_airplane_0.wav', "r")
#
## Extract Raw Audio from Wav File
#signal = spf.readframes(-1)
#signal = np.fromstring(signal, np.int16)
#fs = spf.getframerate()
#
## If Stereo
#if spf.getnchannels() == 2:
#    print("Just mono files")
#
#Time = np.linspace(0, len(signal) / fs, num=len(signal))
#
#plt.figure(1)
#plt.title("Waveform")
#plt.plot(Time, signal)
#plt.show()
# image_path =  "C:\Users\blura\Desktop\Voice Bachelor proj"
# skimage.io.imsave(image_path, img)



########################################## training ###############################################################

# '# Plotting the data
# 'plt.figure(1, figsize=(10, 6))
# 'plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label='Loss')
# '
# '# Adding titles and labels
# 'plt.xlabel('Epoch')
# 'plt.ylabel('Loss')
# 'plt.title('Loss Over Epochs')
# 'plt.legend()
# '
# '# Ensure the x-axis has integer steps
# 'plt.xticks(epochs)
# '# Set the x-axis limits to start from the first epoch and end at the last epoch
# 'plt.xlim(1, len(loss_values))
# '
# 'last_epoch = epochs[-1]
# 'last_loss = loss_arr[-1]
# 'plt.annotate(f'{last_loss:.3f}', xy=(last_epoch, last_loss), xytext=(last_epoch, last_loss + 0.05))
# '
# '# Save the graph to the current directory
# 'plt.savefig('loss_over_epochs.png')
# '
# 'plt.figure(2, figsize=(10, 6))
# 'plt.plot(epochs, accuracy_values, marker='o', linestyle='-', color='b', label='Loss')
# '
# '# Adding titles and labels
# 'plt.xlabel('Epoch')
# 'plt.ylabel('Accuracy')
# 'plt.title('Accuracy Over Epochs')
# 'plt.legend()
# '
# '# Ensure the x-axis has integer steps
# 'plt.xticks(epochs)
# '# Set the x-axis limits to start from the first epoch and end at the last epoch
# 'plt.xlim(1, len(loss_values))
# '
# 'last_accuracy = accuracy_values[-1]
# 'plt.annotate(f'{last_accuracy:.2f}%', xy=(last_epoch, last_accuracy), xytext=(last_epoch, last_accuracy - 5))
# '
# '# Save the graph to the current directory
# 'plt.savefig('accuracy_over_epochs.png')