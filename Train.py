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
# termcolor, do (pip install termcolor)


import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
import seaborn as sns
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
from PIL import Image
import os
import glob
import random
#from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
#from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
#import MiscUtils as iu
from Network.Network import CIFAR10Model
from Misc.MiscUtils import *
from Misc.DataUtils import *
from sklearn.metrics import confusion_matrix



# Don't generate pyc codes
sys.dont_write_bytecode = True
class CIFAR10Custom(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels_path, transform=None, dataset_type="train"):
        """
        Args:
            root_dir: Directory with all images (1 to 50000)
            labels_path: Path to LabelsTrain.txt
            transform: Optional transform to be applied
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_images = len(os.listdir(root_dir))
        self.dataset_type = dataset_type

         # Read labels using your existing function
        self.labels = ReadLabels(labels_path)
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


def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize, istest = False):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet)-1)
        
        ImageNum += 1
        
          ##########################################################
          ###########################################################
    	  # Add any standardization or data augmentation here
        #
        # augmentation = transforms.Compose([
        #     transforms.Pad(4),
        #     transforms.RandomHorizontalFlip(), transforms.RandomCrop(32),
        #     # Randomly flip the image horizontally
        #     # transforms.RandomRotation(15, expand=False),  # Randomly rotate the image by 15 degrees
        #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        #     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        #     # Random changes to color
        #     # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Randomly crop the image and resize to 32x32
        #     # transforms.ToTensor()  # Convert image to Tensor
        # ])
        # Apply augmentation only during training, not during testing
        if not istest:
            augmentation = transforms.Compose([
                #transforms.Pad(4),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomCrop(32),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        else:
            # For testing, no augmentation, just normalization
            augmentation = transforms.Compose([
               # transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        ## Add any standardization or data augmentation here!
          ##########################################################

        I1, Label = TrainSet[RandIdx]
        I1 = augmentation(I1)

        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))
        
    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """
    # Print shapes for debugging
    print(f"Shape of y_true: {LabelsTrue.shape}")
    print(f"Shape of y_pred: {LabelsPred.shape}")
    # Get the confusion matrix using sklearn.
    #LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(LabelsTrue,  # True class for dataset-set.
                          LabelsPred)  # Predicted class.
    print("Confusion matrix shape:", cm.shape)

    # Create figure and axes
    plt.figure(figsize=(10, 8))

    # Plot using seaborn
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
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

   # print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')

def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, LogsPath, TestSet, TestLabels):#22testset
    """
    Inputs: 
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Initialize the model
    model = CIFAR10Model(InputSize=3*32*32,OutputSize=10) 
    ###############################################
    # Parameters
    print("\nModel Parameter Summary:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} - {param.data.shape}")
            total_params += param.numel()
    print(f"\nTotal trainable parameters: {total_params:,}\n")

    # Optimizer of
    #Optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    Optimizer = AdamW(model.parameters(), lr= 0.01, weight_decay=1e-4)
    #Optimizer = AdamW(model.parameters(), lr= 0.001, weight_decay=1e-4)
    # Fill your optimizer of choice here!
    #scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=15, gamma=0.1)
    ###############################################
    #Optimizer = ...

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
    epoch_losses = []
    epoch_accuracies = []
    epoch_testing_accuracies = []
    epoch_testing_losses = []
    all_Train_Prediction  = []
    all_Train_Label = []
    all_Test_Prediction = []
    all_Test_Label = []
    Training_time = tic()
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
        total_loss = 0.0
        total_accuracy = 0.0
        Train_Label = []
        Train_Prediction = []
        # Training loop
        model.train()  # Set model to training mode
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)
            # Predict output with forward pass
            I1Batch, LabelBatch = Batch
            #Forward Pass
            LossThisBatch = model.training_step(Batch)

            #Prediction of labels
            if Epochs == NumEpochs - 1:
                Pred = model(I1Batch)
                _, Predicted = torch.max(Pred.data, 1)

                # Storing Predictions and true labels
                Train_Prediction.extend(Predicted.cpu().numpy())
                Train_Label.extend(LabelBatch.cpu().numpy())

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Accumulate the loss for this epoch
            total_loss += LossThisBatch.item()
            print(f"LossThisBatch {LossThisBatch}, NumIterationsPerEpoch {NumIterationsPerEpoch}")

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                
                torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
                print('\n' + SaveName + ' Model Saved...')

            result = model.validation_step(Batch)
            # Accumulate the accuracy for this epoch
            total_accuracy += result["acc"].item()


            model.epoch_end(Epochs*NumIterationsPerEpoch + PerEpochCounter, result)
            # Tensorboard
            Writer.add_scalar('LossEveryIter', result["loss"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            Writer.add_scalar('Accuracy', result["acc"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()




        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / NumIterationsPerEpoch  # You can uncomment this line if you want avg loss per epoch
        avg_accuracy = total_accuracy / NumIterationsPerEpoch  # Average accuracy for the epoch
        print(f"Epoch {Epochs + 1}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
        print(f"Learning rate at epoch {Epochs}: {Optimizer.param_groups[0]['lr']}")

        # Store the average accuracy for the epoch
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(avg_accuracy)

        # Extend the complete lists with this epoch's predictions
        all_Train_Prediction.extend(Train_Prediction)
        all_Train_Label.extend(Train_Label)

        #Checking Testing Model Accuracy
        # Evaluate test accuracy
        test_accuracy = 0.0
        test_loss = 0.0
        Test_Label = []
        Test_Prediction = []
       #test_correct = 0
        #test_total = 0
        # Set the model to evaluation mode for test accuracy
        model.eval()  # Disable layers like dropout during evaluation
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for PerEpochCounter in range(NumIterationsPerEpoch):
                # Generate a batch for testing
                TestBatch = GenerateBatch(TestSet, TestLabels, ImageSize, MiniBatchSize, istest=True)
                TestI1Batch, TestLabelBatch = TestBatch

                # Forward pass

                # Prediction of Test labels
                if Epochs == NumEpochs - 1:
                    Pred = model(TestI1Batch)
                    _, Predicted = torch.max(Pred.data, 1)

                    # Storing Predictions and true labels
                    Test_Prediction.extend(Predicted.cpu().numpy())
                    Test_Label.extend(TestLabelBatch.cpu().numpy())

                result = model.validation_step(TestBatch)
                test_accuracy += result["acc"].item()
                test_loss+= result["loss"].item()
                #TestPredT = torch.argmax(TestPrediction, dim=1)

                # # Update test accuracy counters
                # test_correct += (TestPredT == TestLabelBatch).sum().item()
                # test_total += MiniBatchSize



        # Calculate test accuracy for the epoch
        avg_test_accuracy = test_accuracy / NumIterationsPerEpoch
        avg_test_loss = test_loss / NumIterationsPerEpoch
        print(f"Epoch {Epochs + 1}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
        print(f"Epoch {Epochs + 1}, Testing Average Loss: {avg_test_loss:.4f}, Testing Average Accuracy: {avg_test_accuracy:.4f}")
        print(f"Learning rate at epoch {Epochs}: {Optimizer.param_groups[0]['lr']}")

        epoch_testing_accuracies.append(avg_test_accuracy)
        epoch_testing_losses.append(avg_test_loss)

        # Extend the complete lists with this epoch's predictions
        all_Test_Prediction.extend(Test_Prediction)
        all_Test_Label.extend(Test_Label)


        # Print epoch results
        print(f"Epoch {Epochs + 1}: Training Accuracy = {avg_accuracy:.4f}, Training loss = {avg_test_loss:.4f}")
        # Optionally, switch back to training mode if continuing to train
        #scheduler.step()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
        print('\n' + SaveName + ' Model Saved...')

        # Plot confusion matrix after training

    Training_Stop = toc(Training_time)
    print("Training Complete and training time: ", Training_Stop)

    ConfusionMatrix(
        np.array(all_Train_Label),
        np.array(all_Train_Prediction),

    )
    plt.savefig(os.path.join(LogsPath, 'Training_confusion_matrix.png'))
    plt.close()

    ConfusionMatrix(
        np.array(all_Test_Label),
        np.array(all_Test_Prediction),

    )
    plt.savefig(os.path.join(LogsPath, 'Testing_confusion_matrix.png'))
    plt.close()

    # Plot the loss over epochs after training
    plt.plot(range(NumEpochs), epoch_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.show()

    # Plot the accuracy over epochs after training
    plt.plot(range(NumEpochs), epoch_accuracies, label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.show()

    # Plot training and test accuracies
    plt.plot(range(NumEpochs), epoch_accuracies, label='Training Accuracy')
    plt.plot(range(NumEpochs), epoch_testing_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot both testing and training loss
    plt.plot(range(NumEpochs), epoch_losses, label='Training loss')
    plt.plot(range(NumEpochs), epoch_testing_losses, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Training and Test loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
        

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/Checkpointsbasicwithoutbn/', help='Path to save Checkpoints, Default: '
                                                                            '../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default= 256, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/Logsbasicwithoutbn/', help='Path to save Logs for Tensorboard, Default=Logs/')
    #TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        #download=True, transform=ToTensor())

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath

    
    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses, TestLabels = SetupAll(CheckPointPath)


    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    TrainSet = CIFAR10Custom(root_dir="D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/CIFAR10/Train",
                             labels_path="D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/LabelsTrain.txt",
                             transform=transforms.Compose([transforms.ToTensor()]), dataset_type="train")

    TestSet = CIFAR10Custom(
        root_dir="D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/CIFAR10/Test",
        labels_path="D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/LabelsTest.txt",
        transform=transforms.Compose([transforms.ToTensor()]), dataset_type="test")

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, LogsPath, TestSet, TestLabels) # 22passing testlabels and test set for accuracies


    
if __name__ == '__main__':

    main()
 
