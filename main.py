from generate_word import generate_word_main
from train_cnn import train_model_main
from edit_audio import edit_audio_main
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple program to print command-line parameters.")
    parser.add_argument("-w", "--word", help="word to be spoken")
    parser.add_argument("-r", "--random", help="use random voices")

    # Parse the command-line arguments
    args = parser.parse_args()
    word = args.word

    generate_word_main(word)
    edit_audio_main(word)
    train_model_main(word)

    #comment
