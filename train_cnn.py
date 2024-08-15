from torch.optim import Adam
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy
import numpy as np
import librosa
import tqdm

from torchinfo import summary

from Alex import LocalAlexNet
from models.SimpleModel import SimpleNet
from models.Net import Net2


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


def testAccuracy(model, device, test_loader, loss_fn):
    model.eval()

    loss_array = []
    correct_predictions_array = []
    train_acc_total = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            # extract the loss value
            loss_array.append(loss.item())

            ### calc accuracy
            _, predicted = torch.max(outputs.data, 1)
            # the amount of predictions can be less than epochs * bach_size so we have to store it
            train_acc_total += labels.size(0)

            # how many predictions are the correct labels -> sum up
            correct_predictions = (predicted == labels).sum().item()
            correct_predictions_array.append(correct_predictions)

    # compute the accuracy over all test images
    #accuracy = (100 * accuracy / total)
    return loss_array, train_acc_total, correct_predictions_array


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs, model, device, train_loader, test_loader, loss_fn, optimizer, validationImages, labels_val):
    best_accuracy = 0.0

    running_loss_arr = []
    trainig_acc_arr = []
    test_acc_arr = []

    for epoch in tqdm.trange(num_epochs, desc="training"):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        train_acc = 0.0
        train_acc_total = 0.0

        running_acc = 0.0

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
        print('Test accuracy %d %%',  (accuracy))
        #v_accuracy = validateModel(model, labels_val, device, validationImages)
        #print("Validation Accuracy - ", v_accuracy[0], " correct out of ", v_accuracy[1])

    return model


def test_one_epoch(model, device, test_loader, loss_fn):
    model.eval()

    loss_array = []
    correct_predictions_array = []
    train_acc_total = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            # extract the loss value
            loss_array.append(loss.item())

            ### calc accuracy
            _, predicted = torch.max(outputs.data, 1)
            # the amount of predictions can be less than epochs * bach_size so we have to store it
            train_acc_total += labels.size(0)

            # how many predictions are the correct labels -> sum up
            correct_predictions = (predicted == labels).sum().item()
            correct_predictions_array.append(correct_predictions)

    # compute the accuracy over all test images
    #accuracy = (100 * accuracy / total)
    return loss_array, train_acc_total, correct_predictions_array


def train_one_epoch(model, device, train_loader, loss_fn, optimizer):
    model.train()

    loss_array = []
    correct_predictions_array = []
    train_acc_total = 0.0

    for i, (images, labels) in enumerate(train_loader, 0):
        # get the inputs
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # predict classes using images from the training set
        outputs = model(images)

        # compute the loss based on model output and real labels
        loss = loss_fn(outputs, labels)
        # backpropagate the loss
        loss.backward()

        # adjust parameters based on the calculated gradients
        optimizer.step()

        # extract the loss value
        loss_array.append(loss.item())

        ### calc accuracy
        _, predicted = torch.max(outputs.data, 1)
        # the amount of predictions can be less than epochs * bach_size so we have to store it
        train_acc_total += labels.size(0)

        #TODO this is retarded just return predicted and labels and do the rest elsewhere if you want to do it like this
        # how many predictions are the correct labels -> sum up
        # save the array so we can see if certain labels are harder to predict
        correct_predictions = (predicted == labels).sum().item()
        correct_predictions_array.append(correct_predictions)
    return loss_array, train_acc_total, correct_predictions_array


# print('train acc: ', (train_acc * 100) / train_acc_total)
def calc_accuracy(total_predictions, correct_predictions):
    sum_correct_predictions = correct_predictions.sum().item()
    return (correct_predictions * 100) / total_predictions


def calc_loss(loss_array):
    epochs = len(loss_array)
    sum = sum(loss_array)
    return sum / epochs


def train_model_main(epochs, dataset_usage, labels, model):
    #TODO check if the new methods produce the same output as the old functions

    #TODO def for calc accuracy
    #TODO def for calc loss

    #TODO def for creating picture from data

    # these are the calcs for the predicitions
    #TODO save all this in an object to use later on for generating images
    #print("epoch: ", (i+1))
    #print("running loss: ", running_loss)
    #print('running loss epoch: %.3f' % ( running_loss / (i + 1)))
    #print('train acc: ', (train_acc * 100) / train_acc_total)
    # TODO move to seperate function calls
    # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
    #accuracy = testAccuracy(model, device, test_loader)
    #print('Test accuracy %d %%',  (accuracy))
    #v_accuracy = validateModel(model, labels_val, device, validationImages)
    #print("Validation Accuracy - ", v_accuracy[0], " correct out of ", v_accuracy[1])



    project_root = os.path.dirname(os.path.abspath(__file__))  # This gets the current file's directory
    parent_dir = os.path.dirname(project_root)
    folder_path = os.path.join(parent_dir, 'image_dataset')

    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    batch_size = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    data_set = MelSpectrograms(folder_path, transform=trans, dataset_labels=labels)
    use, discard = torch.utils.data.random_split(data_set, [dataset_usage, 1 - dataset_usage])
    train_set, test_set = torch.utils.data.random_split(use, [0.8, 0.2])

    # Create a loader for the training set which will read the data within batch size and put into memory.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    validationSet = load_validation_set()

    # results for the complete run
    # loss_array, train_acc_total, correct_predictions_array
    training_data_complete = {
        "train": {},
        "test": {},
    }

    # results for one epoch/run
    training_data = {
        "loss_array": [],
        "train_acc_total": [],
        "correct_predictions_array": []
    }

    # results for one epoch/run
    test_data = {
        "loss_array": [],
        "train_acc_total": [],
        "correct_predictions_array": []
    }

    for e in range(epochs):
        train_loss_array, train_total_acc, train_correct_pred_array = train_one_epoch(model, device, train_loader, loss_fn, optimizer)
        training_data["loss_array"].append(train_loss_array)
        training_data["train_acc_total"].append(train_total_acc)
        training_data["correct_predictions_array"].append(train_correct_pred_array)

        test_loss_array, test_total_acc, test_correct_pred_array = test_one_epoch(model, device, test_loader, loss_fn)
        test_data["loss_array"].append(test_loss_array)
        test_data["train_acc_total"].append(test_total_acc)
        test_data["correct_predictions_array"].append(test_correct_pred_array)

    training_data_complete["train"] = training_data
    training_data_complete["test"] = test_data

    torch.save(training_data_complete, "../generated_models/training_run.pth")
    loaded_run = torch.load("../generated_models/training_run.pth")

    exit()
    #model = train(epochs, model, device, train_loader, test_loader, loss_fn, optimizer, validationSet, labels)
    #saveModel(model, labels)
    #return model