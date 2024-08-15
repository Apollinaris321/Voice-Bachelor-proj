from generate_word import gen_words_main

from train_cnn import train_model_main
from edit_audio import edit_audio_main
import argparse
import os
import time
import tqdm
from test_model_against_voice import validate_model
import torch

from Alex import LocalAlexNet
from models.SimpleModel import SimpleNet
from models.Net import Net2

from torchinfo import summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple program to print command-line parameters.")
    parser.add_argument("-w", "--word", help="word to be spoken")
    parser.add_argument("-r", "--random", help="use random voices")

    # Parse the command-line arguments
    args = parser.parse_args()
    word = args.word
    word = ""


    labels = [
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

    model = LocalAlexNet(len(labels))

    batch_size = 16
    epochs = 3
    #summary(model, input_size=(batch_size, 1, 128, 87))

    #gen_words_main(words=word, amount_of_voices=5)
    #edit_audio_main(labels)

    model = train_model_main(epochs, 1, labels, model)
    #validate_model(model)


