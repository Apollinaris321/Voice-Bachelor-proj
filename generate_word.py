import argparse
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
import os


def create_path(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")


def word_exists(word):
    project_root = os.path.dirname(os.path.abspath(__file__))  # This gets the current file's directory
    parent_dir = os.path.dirname(project_root)
    folder_path = os.path.join(parent_dir, 'results')
    word_path = os.path.join(folder_path, word)

    how_many_words_exist = 0
    if os.path.exists(word_path):
        how_many_words_exist = len(os.listdir(word_path))

    path_exists = os.path.exists(word_path)

    return [path_exists, how_many_words_exist]


def remove_audio(word):
    project_root = os.path.dirname(os.path.abspath(__file__))  # This gets the current file's directory
    parent_dir = os.path.dirname(project_root)
    folder_path = os.path.join(parent_dir, 'results')

    for item in os.listdir(folder_path):
        if item in word:
            # List all files in the folder
            for filename in os.listdir(os.path.join(folder_path,item)):
                folder = os.path.join(folder_path,item)
                file = os.path.join(folder, filename)
                os.remove(file)


def generate_tts(word, amount_of_voices=-1, voices_start_index=0):
    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_root)

    voice_path = os.path.join(parent_dir, 'voices')
    results_path = os.path.join(parent_dir, 'results')
    results_audio_path = f"{results_path}/{word}/"

    create_path(results_audio_path)

    items = os.listdir(voice_path)
    # Filter the items to keep only the directories (folders)
    voice_names = [item for item in items if os.path.isdir(os.path.join(voice_path, item))]

    if amount_of_voices == -1:
        amount_of_voices = len(voice_names)
    if voices_start_index >= amount_of_voices:
        voices_start_index = amount_of_voices
    elif voices_start_index < 0:
        voices_start_index = 0

    tts = TextToSpeech(kv_cache=True)
    for i, voice in enumerate(voice_names[voices_start_index:amount_of_voices]):
        clips_paths = [
            f"{voice_path}/{voice}/snippet0.wav",
            f"{voice_path}/{voice}/snippet1.wav",
            f"{voice_path}/{voice}/snippet2.wav",
            f"{voice_path}/{voice}/snippet3.wav",
        ]
        reference_clips = [load_audio(p, 22050) for p in clips_paths]
        pcm_audio = tts.tts_with_preset(
            "[loud and clear]" + word + "...",
            voice_samples=reference_clips,
            preset='ultra_fast',
        )

        pcm_audio = pcm_audio.squeeze(0)
        audio_save_path = f'{results_audio_path}/{voice}_{word}_0.wav'
        # resample to 441 khz
        new_noise = torchaudio.functional.resample(waveform=pcm_audio, orig_freq=22050,
                                                   new_freq=44100, )
        torchaudio.save(audio_save_path, new_noise, 44100)

    return results_audio_path


def gen_words_main(words=[], replace=False, amount_of_voices=None):

    for word in words:
        exists, amount = word_exists(word)
        if exists and amount < amount_of_voices:
            generate_tts(word, amount_of_voices=amount_of_voices, voices_start_index=amount)
        elif exists and amount > amount_of_voices:
            pass
        elif not exists:
            generate_tts(word, amount_of_voices)