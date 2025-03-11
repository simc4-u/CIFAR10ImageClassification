#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Prof. Nitin J. Sanket (nsanket@wpi.edu)
Assistant Professor,
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(CheckPointPath):
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    BasePath = "D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles"
    # Setup DirNames
    DirNamesTrain =  SetupDirNames(BasePath)

    # Read and Setup Labels
    LabelsPathTrain = 'D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/LabelsTrain.txt'
    TrainLabels = ReadLabels(LabelsPathTrain)

    # Read and Setup Test Labels
    BasePath_Test = "D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles"
    DirNamesTest = SetupDirNames(BasePath_Test)

    # Read and Setup Labels
    LabelsPathTest = 'D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/LabelsTest.txt'
    TestLabels = ReadLabels(LabelsPathTest)

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)
        
    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100 
    # Number of passes of Val data with MiniBatchSize 
    NumTestRunsPerEpoch = 5
    
    # Image Input Shape
    ImageSize = [32, 32, 3]
    NumTrainSamples = len(DirNamesTrain)
    NumTestSamples = len(DirNamesTest)

    # Number of classes
    NumClasses = 10

    return DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses , TestLabels

def ReadLabels(LabelsPathTrain):
    if(not (os.path.isfile(LabelsPathTrain))):
        print('ERROR: Train Labels do not exist in '+LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, 'r')
        TrainLabels = TrainLabels.read()
        TrainLabels = map(float, TrainLabels.split())

    return TrainLabels
    

def SetupDirNames(BasePath): 
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTrain = ReadDirNames('D:/Computer vision/Computer Vision_Assignments/Simran_HW0/schauhan_hw0/Phase2/Code/TxtFiles/DirNamesTrain.txt')
    
    return DirNamesTrain

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
