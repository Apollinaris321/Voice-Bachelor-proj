from generate_word import gen_words_main

from train_cnn import train_model_main
from edit_audio import edit_audio_main
import os
import shutil

from code.models.Alex import LocalAlexNet

from analyze_data import draw_graphs
import tkinter as tk
from tkinter import filedialog


# Function to prompt the user for a file save location
def ask_save_location(default_filename="output.txt"):
    # Hide the root Tkinter window
    root = tk.Tk()
    root.withdraw()

    dir_name = filedialog.askdirectory()
    return dir_name


def is_number_in_range(s):
    # First, check if the string is a digit (positive integer)
    if s.isdigit():
        # Convert the string to an integer
        num = int(s)
        # Check if the number is within the desired range
        if 0 <= num <= 92:
            return True
    # If the string is not a digit or out of range, return False
    return False


def check_epoch_input(user_input):
    if user_input.isdigit():
        num = int(user_input)
        if 1 <= num:
            return True
    return False


if __name__ == '__main__':



    print("Automatic Speech Recognition Builder")
    print("Please enter the words you want to recognize seperated by spaces")

    word_list_unfiltered = input()
    word_list = word_list_unfiltered.split()

    print("this is what you wrote: ", word_list)
    #TODO accept -> yes/no ; no -> redo (while loop)

    res = False
    while not res:
        print("how many voices should be used ? 0-92 available")

        voices_amount = input()

        res = is_number_in_range(voices_amount)

        print("you chose ", voices_amount)
        print("accepted answer: ", res)
    voices_amount = int(voices_amount)

    word_list = [
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

    run = 0

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_path = os.path.join(project_root, "generated_models")
    run_path = os.path.join(models_path, str(run))

    if not os.path.exists(run_path):
        os.makedirs(run_path)
    else:
        while os.path.exists(run_path):
            run += 1
            run_path = os.path.join(models_path, str(run))
        os.makedirs(run_path)

    model = LocalAlexNet(len(word_list))
    batch_size = 64
    epochs = 8
    #summary(model, input_size=(batch_size, 1, 128, 87))

    epoch_valid = False
    while not epoch_valid:
        print("How many epochs: ")
        epochs = input()
        epoch_valid = check_epoch_input(epochs)
        if not epoch_valid:
            print("Please enter a valid number of epochs")
        else:
            epochs = int(epochs)

    gen_words_main(words=word_list, amount_of_voices=voices_amount)
    edit_audio_main(word_list)

    train_model_main(epochs, 1, word_list, model, batch_size, run_path)
    draw_graphs(run_path, 1, 0)


    run_path = os.path.join(models_path, str(1))
    save_path = ask_save_location()
    if save_path:
        for file in os.listdir(run_path):
            shutil.copy(os.path.join(run_path, file), save_path)
        print(f"File saved at: {save_path}")
    else:
        print("Save operation canceled.")

    exit()