#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
#from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from  PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from Network.Network import CIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *
import seaborn as sns

import torch


# Don't generate pyc codes
sys.dont_write_bytecode = True
class CIFAR10Custom(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels_path, transform=None):
        """
        Args:
            root_dir: Directory with all images (1 to 50000)
            labels_path: Path to LabelsTrain.txt
            transform: Optional transform to be applied
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_images = len(os.listdir(root_dir))

         # Read labels using your existing function
        self.labels,_ = ReadLabels((labels_path), "")
        self.labels = list(map(int, self.labels))
        print(self.labels)


    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        """Get image and label at index idx"""
          # Images are numbered from 1 to 50000
        #img_name = os.path.join(self.root_dir, str(idx + 1).png)
        img_name = os.path.join(self.root_dir, f"{idx + 1}.png")
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

def SetupAll_Test():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]
     # Read and Setup Labels
    LabelsPath = "D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/LabelsTest.txt"
    TestLabels, _ = ReadLabels(LabelsPath, " ")

    BasePath = "D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles"
    # Setup DirNames
    DirNamesTest =  SetupDirNames(BasePath)

    # Image Input Shape
    ImageSize = [32, 32, 3]
    NumTestSamples = len(DirNamesTest)
    return ImageSize, NumTestSamples, TestLabels


def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTest= ReadDirNames(
        'D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/DirNamesTest.txt')

    return DirNamesTest

def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames


def StandardizeInputs(Img):
    ##########################################################################
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
        ##########################################################################
    return transform(Img)
    
# def ReadImages(Img):
#     """
#     Outputs:
#     I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
#     I1 - Original I1 image for visualization purposes only
#     """
#     I1 = Img
#
#     if(I1 is None):
#         # OpenCV returns empty list if image is not read!
#         print('ERROR: Image I1 cannot be read')
#         sys.exit()
#
#     I1S = StandardizeInputs(np.float32(I1))
#
#     I1Combined = np.expand_dims(I1S, axis=0)
#
#     return I1Combined, I1

def GenerateBatch(TestSet, TestLabels, ImageSize, MiniBatchSize):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1Batch = []
    LabelBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TestSet)-1)

        ImageNum += 1

    I1, Label = TestSet[RandIdx]
    I1 = StandardizeInputs(I1)

    # Append All Images and Mask
    I1Batch.append(I1)
    LabelBatch.append(torch.tensor(Label))

    return torch.stack(I1Batch), torch.stack(LabelBatch)

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if (not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in ' + LabelsPathPred)
        return LabelTest, None
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """
    # Print shapes for debugging
    print(f"Shape of y_true: {LabelsTrue.shape}")
    print(f"Shape of y_pred: {LabelsPred.shape}")
    # Get the confusion matrix using sklearn.
    # LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(LabelsTrue,  # True class for dataset-set.
                          LabelsPred)  # Predicted class.
    print("Confusion matrix shape:", cm.shape)

    # Create figure and axes
    plt.figure(figsize=(10, 8))

    # Plot using seaborn
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')

    # Customize the plot
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, MiniBatchSize, LogsPath, NumEpochs, DivTest, TestLabels):
    """
    Inputs: 
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = CIFAR10Model(InputSize=3*32*32,OutputSize=10) 
    
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.eval()
    print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print
    #         name, param.data
    # If CheckPointPath doesn't exist make the path
    OutSaveT = open(LabelsPathPred, 'w')
    StartEpoch = 0
    epoch_losses = []
    epoch_accuracies = []
    inference_times = []  # To store inference times
    NumTestSamples = len(TestSet)
    Testing_Start = tic()
    all_Test_Prediction = []
    all_Test_Label = []
    # Initialize TensorBoard writer
    Writer = SummaryWriter(LogsPath)
    with torch.no_grad():
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTestSamples / MiniBatchSize / DivTest)  # 195 epochs in each iteration
            # Initialize total loss for the epoch
            total_loss = 0.0
            total_accuracy = 0.0
            epoch_inference_time = 0.0
            Test_Label = []
            Test_Prediction = []
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                Batch = GenerateBatch(TestSet, TestLabels, ImageSize, MiniBatchSize)
                I1Batch, LabelBatch = Batch
                #Measure inference time
                start_time = tic()

                #Forward pass

                # For minibatch_size = 128:
                Prediction = model(I1Batch)
                if Epochs == NumEpochs - 1:
                    PredT = torch.argmax(Prediction, dim=1)  # Shape: [128]
                    Test_Prediction.extend(PredT.cpu().numpy())
                    Test_Label.extend(LabelBatch.cpu().numpy())

                inference_time = toc(StartTime=start_time)
                print("inference_time: ", inference_time)
                epoch_inference_time += inference_time

                # # Compute accuracy manually
                # correct_predictions = torch.sum(PredT == LabelBatch).item()
                # total_accuracy += correct_predictions / MiniBatchSize

                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(Prediction, LabelBatch)
                total_loss += loss.item()

                Writer.add_scalar('LossEveryIter', loss, Epochs * NumIterationsPerEpoch + PerEpochCounter)
                Writer.add_scalar('Accuracy', total_loss, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()

            #Calculate average inference time
            avg_inference_time = epoch_inference_time / NumIterationsPerEpoch
            inference_times.append(avg_inference_time)

            # Calculate average loss and accuracy for the epoch
            avg_loss = total_loss / NumIterationsPerEpoch
            avg_accuracy =   total_accuracy / NumIterationsPerEpoch # Average accuracy for the epoch
            print(f"Epoch {Epochs + 1}, Average inference time: {avg_inference_time}Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

            # Store the average accuracy for the epoch
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(avg_accuracy)

            # Extend the complete lists with this epoch's predictions
            all_Test_Prediction.extend(Test_Prediction)
            all_Test_Label.extend(Test_Label)
        OutSaveT.close()
        Testing_End = toc(Testing_Start)
        ConfusionMatrix(
            np.array(all_Test_Label),
            np.array(all_Test_Prediction),

        )
        plt.savefig(os.path.join(LogsPath, 'Testing_confusion_matrix.png'))
        plt.close()

        # Plot the loss over epochs after training
        plt.plot(range(NumEpochs), epoch_losses, label='Testloss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.show()

        # Plot the accuracy over epochs after training
        plt.plot(range(NumEpochs), epoch_accuracies, label='Testaccuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.show()


def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/Checkpointsimprovedbasic/49model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--LogsPath',
                        default='D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/Logstestimpr/',
                        help='Path to save Logs for Tensorboard, Default=Logs/')

    Parser.add_argument('--NumEpochs', type=int, default= 40, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTest', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default= 1, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--NumTestSamples', type=int, default=1, help='Size of the MiniBatch to use, Default:1')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath
    LogsPath = Args.LogsPath
    MiniBatchSize = Args.MiniBatchSize
    NumEpochs = Args.NumEpochs
    DivTest = Args.DivTest
    NumTestSamples = Args.NumTestSamples

    #TestSet = CIFAR10(root='data/', train=False)

    TestSet = CIFAR10Custom(root_dir="D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/CIFAR10/Test",
                            labels_path="D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/LabelsTest.txt",
    transform=transforms.Compose([transforms.ToTensor()]))

    # Setup all needed parameters including file reading
    ImageSize, NumTestSamples, TestLabels = SetupAll_Test()

    # Define PlaceHolder variables for Predicted output
    LogsPath = "D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/Logs_RESNEXT14TEST"
    LabelsPathPred = 'D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/PredOut.txt' # Path to save predicted labels


    TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, MiniBatchSize, LogsPath, NumEpochs, DivTest, TestLabels)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)

    #ConfusionMatrix(LabelsTrue, LabelsPred)
     
if __name__ == '__main__':
    main()
 
