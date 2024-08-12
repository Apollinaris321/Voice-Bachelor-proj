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


def generate_word_main(word):
    tts = TextToSpeech(kv_cache=True)
    voice_path = "C:/Users/blura/Desktop/Voice Bachelor proj/voices"
    results_path = "C:/Users/blura/Desktop/Voice Bachelor proj/results"
    results_audio_path = f"{results_path}/{word}/"

    create_path(results_audio_path)

    items = os.listdir(voice_path)
    # Filter the items to keep only the directories (folders)
    voice_names = [item for item in items if os.path.isdir(os.path.join(voice_path, item))]

    for i, voice in enumerate(voice_names):
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
        audio_save_path = f'{results_audio_path}/{i}_{word}_0.wav'
        # resample to 441 khz
        new_noise = torchaudio.functional.resample(waveform=pcm_audio, orig_freq=22050,
                                                   new_freq=44100, )
        torchaudio.save(audio_save_path, new_noise, 44100)

    return results_audio_path
