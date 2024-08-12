from generate_word import generate_word_main

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple program to print command-line parameters.")
    parser.add_argument("-w", "--word", help="word to be spoken")
    parser.add_argument("-r", "--random", help="use random voices")

    # Parse the command-line arguments
    args = parser.parse_args()
    word = args.word
    word = ""

    epochs = 20

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
        "war"
    ]

    model = LocalAlexNet(len(labels))

    #torch.save(labels, "labels.pth")

    #label_load = torch.load("labels.pth")
    #print(label_load)

    #generate_word_main(word)
    #edit_audio_main(word)
    model = train_model_main(epochs, 1, labels, model)
    #validate_model(model)


