from datetime import datetime

from torch.optim import Adam
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from PIL import Image

import os

from Alex import LocalAlexNet
from models.SimpleModel import SimpleNet
from models.Net import Net2

import numpy
import numpy as np
import librosa
import tqdm


class MelSpectrograms(Dataset):
    def __init__(self, image_dir, transform=None, dataset_labels=[]):
        """
        Arguments:
            root_dir (string): Directory with all the images.
        """
        self.labels = dataset_labels
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.data = []
        transformed_data = []
        for file in os.listdir(self.image_dir):
            img = Image.open(image_dir + "/" + file)
            self.images.append(img.load())
            label = file.split("_")[1]
            label_index = self.labels.index(label)
            self.data.append([img, label_index])

        for img, label in self.data:
            transformed_img = self.transform(img)
            transformed_data.append([transformed_img, label])

        self.data = transformed_data
        transformed_data = []
        print("data loaded")

    def __len__(self):
        return len([name for name in os.listdir(self.image_dir) if os.path.isfile(self.image_dir + "/" + name)])
        # return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range.")

        return self.data[idx]

        files = os.listdir(self.image_dir)
        img_path = os.path.join(self.image_dir, files[idx])
        this_label = img_path.split("img2")[1].split("_")[1]

        img = Image.open(img_path)
        t = self.transform(img)

        index = 0
        try:
            index = self.labels.index(this_label)
        except ValueError:
            print(f"'{this_label}' not found in the list.")

        return t, index


# Function to save the model
def saveModel(model, labels):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the relative path to the folder inside your project
    folder_name = 'generated_models'
    folder_path = os.path.join(ROOT_DIR, folder_name)

    version = 0

    model_path = "./model_" + model.__class__.__name__ + "_v" + str(version) + ".0.pth"
    path = os.path.join(folder_path, model_path)

    while os.path.exists(path):
        version += 1
        model_path = "./model_" + model.__class__.__name__ + "_v" + str(version) + ".0.pth"
        labels_path = "./model_" + model.__class__.__name__ + "_v" + str(version) + "_labels.pth"
        path = os.path.join(folder_path, model_path)
        labels_path_os = os.path.join(folder_path, labels_path)

    torch.save(labels, labels_path_os)
    torch.save(model.state_dict(), path)


def load_validation_set():
    images = []
    for file in os.listdir("../test_aufnahmen"):
        y, sr = librosa.load("../test_aufnahmen/" + file)
        #generate_mel(waveform=y, sample_rate=sr)
        img = generate_mel(y,sr)
        images.append([img, file.split('_')[0]])
    return images


def generate_mel(waveform, sample_rate):
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

    return img


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def validateModel(model, labels, device, images):
    counter = 0
    correct = 0
    model.eval()
    for img,label in images:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=1)
        ])

        with torch.no_grad():
            t = trans(img.copy())
            t = t.unsqueeze(0)
            t = t.to(device)
            outputs = model(t)
            _, predicted = torch.max(outputs.data, 1)

            counter += 1
            if labels[predicted] == label:
                correct += 1
    return [correct, counter]



def testAccuracy(model, device, test_loader):
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs, model, device, train_loader, test_loader, loss_fn, optimizer, validationImages, labels_val):
    best_accuracy = 0.0

    for epoch in tqdm.trange(num_epochs, desc="training"):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        train_acc = 0.0
        train_acc_total = 0.0

        running_acc = 0.0

        # TODO
        # train_loader muss dann meins rein
        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = images.to(device)
            labels = labels.to(device)

            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value

            ### calc accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_acc_total += labels.size(0)
            train_acc += (predicted == labels).sum().item()

        #TODO save all this in an object to use later on for generating images
        print("running loss: ", running_loss)
        print('[%d, %5d] epoch loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
        print('train acc: ', (train_acc * 100) / train_acc_total)

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy(model, device, test_loader)
        print('Test accuracy %d %%' (accuracy))
        v_accuracy = validateModel(model, labels_val, device, validationImages)
        print("Validation Accuracy - ", v_accuracy[0], " correct out of ", v_accuracy[1])

    return model


def train_model_main(epochs, dataset_usage, labels, model):

    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    batch_size = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.00001)

    data_path = "C:/Users/blura/Desktop/Voice Bachelor proj/image_dataset"
    data_set = MelSpectrograms(data_path, transform=trans, dataset_labels=labels)
    use, discard = torch.utils.data.random_split(data_set, [dataset_usage, 1 - dataset_usage])
    train_set, test_set = torch.utils.data.random_split(use, [0.8, 0.2])

    # Create a loader for the training set which will read the data within batch size and put into memory.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    validationSet = load_validation_set()

    model = train(epochs, model, device, train_loader, test_loader, loss_fn, optimizer, validationSet, labels)
    saveModel(model, labels)
    return model