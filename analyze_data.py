import os
import torch
import matplotlib.pyplot as plt
import numpy as np


# This gets the current file's directory
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
graph_path = os.path.join(parent_dir, 'graphs')


def draw_graph(array, title, x_label, y_label, tick_steps, save_name):
    xpoints = np.arange(len(array))
    ypoints = np.array(array)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(xpoints, ypoints)

    plt.xlim(0, len(array) - 1)
    plt.xticks(np.arange(0, len(xpoints) + 5, step=tick_steps))
    plt.grid()

    plt.savefig(graph_path + "/" + save_name + '.png', bbox_inches='tight')
    plt.show()


def draw_per_batch(array, title, y_label, tick_steps , file_name):
    for i, arr in enumerate(array):
        draw_graph(arr, title +  " Epoch: " + i, "Batch", y_label, tick_steps, file_name + "_epoch_" + i)


def draw_per_epoch(array, title, y_label, tick_steps, file_name):
    avg_arr = []
    for arr in array:
        n_arr = np.array(arr)
        avg_arr.append(np.average(n_arr))
    draw_graph(avg_arr, title, "Epoch", y_label, tick_steps, file_name)


def draw_graphs(training_run, draw_epoch, draw_batch):
    loaded_run = torch.load(training_run + "/training_data.pth")

    test_run = loaded_run['test']
    train_run = loaded_run['train']

    loss = train_run['loss']
    accuracy = train_run['accuracy']

    test_loss = test_run['loss']
    test_accuracy = test_run['accuracy']

    if draw_epoch == 1:
        # train graphs
        draw_per_epoch(loss, "Training Loss per Epoch", "Loss", 1, "train_loss_epoch")
        draw_per_epoch(accuracy, "Training Accuracy per Epoch", "Accuracy", 1, "train_accuracy_epoch")

        draw_per_epoch(test_loss, "Test Loss per Epoch", "Loss", 1, "test_loss_epoch")
        draw_per_epoch(test_accuracy, "Test Accuracy per Epoch", "Accuracy", 1, "test_accuracy_epoch")

    if draw_batch == 1:
        # draw_per_batch()

        # test graphs

        # draw_per_epoch()
        # draw_per_batch()
        pass

