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

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.onnx  # Note the correct spelling: onnx, not onx
import torch.nn as nn
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss = F.cross_entropy(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))
#While testing network, Please uncomment networks
# I have defined all five network below

# Implementation of Basic Network
### My Basic Network STARTS
class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      super(CIFAR10Model, self).__init__()
      """
      Inputs:
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """


      #Convolutional layer 1
      self.conv1 = nn.Conv2d(3, 32, 3)
      self.pool1 = nn.MaxPool2d(2, 2)
      #
      # # Convolutional layer 2
      self.conv2 = nn.Conv2d(32, 64, 3)
      self.pool2 = nn.MaxPool2d(2, 2)
      #
      # # Fully connected layers
      self.fc1 = nn.Linear(64 * 6 * 6, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)


  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network


      # """
      # #####FORWARD FOR BASIC STARTS HERE
      xb = self.pool1(nn.ReLU()(self.conv1(xb)))
      xb = self.pool2(nn.ReLU()(self.conv2(xb)))
      xb = xb.view(xb.size(0), -1)  # Flatten the output
      xb = nn.ReLU()(self.fc1(xb))
      xb = nn.ReLU()(self.fc2(xb))
      out = self.fc3(xb)
      ############################ ENDS HER

      return out
  # ##### MY BASIC NETWORK ENDS.

# Implementation of Improved Network###IMPROVED BASIC ### STARTS HERE
class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      super(CIFAR10Model, self).__init__()
      """
      Inputs:
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      ############################
      self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
      self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1
      self.pool1 = nn.MaxPool2d(2, 2)

      # Convolutional layer 2 with batch normalization
      self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
      self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2
      self.pool2 = nn.MaxPool2d(2, 2)

      # Fully connected layers
      self.fc1 = nn.Linear(64 * 8 * 8, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, OutputSize)
      ####

  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network"""
      xb = F.relu(self.bn1(self.conv1(xb)))  # Apply batch normalization and ReLU activation
      xb = self.pool1(xb)  # Max pooling

      # Second convolutional block
      xb = F.relu(self.bn2(self.conv2(xb)))  # Apply batch normalization and ReLU activation
      xb = self.pool2(xb)  # Max pooling

      # Flatten the output of the conv layers before feeding into the fully connected layers
      xb = xb.view(xb.size(0), -1)  # Flatten to (batch_size, num_features)

      # Fully connected layers with ReLU activation
      xb = F.relu(self.fc1(xb))
      xb = F.relu(self.fc2(xb))

      # Output layer
      out = self.fc3(xb)

      return out






# Implementation of ResNet
class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      super(CIFAR10Model, self).__init__()
      """
      Inputs:
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      ############################

      ###### RESNET14
     #Implementation of ResNet -14 , when n =2
      #First layer: 3x3 convolution, filters=16, output size = 32x32
      #STARTS HERE

      # Initial convolutional layer
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
      self.bn1 = nn.BatchNorm2d(16)

      # Stage 1: Two residual blocks with 16 filters
      self.conv2_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
      self.bn2_1 = nn.BatchNorm2d(16)
      self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
      self.bn2_2 = nn.BatchNorm2d(16)

      self.conv3_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
      self.bn3_1 = nn.BatchNorm2d(16)
      self.conv3_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
      self.bn3_2 = nn.BatchNorm2d(16)

      # Stage 2: Downsampling + two residual blocks with 32 filters
      self.conv4_down = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
      self.bn4_down = nn.BatchNorm2d(32)
      self.shortcut4 = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0)
      self.bn4_shortcut = nn.BatchNorm2d(32)

      self.conv4_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
      self.bn4_2 = nn.BatchNorm2d(32)

      self.conv5_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
      self.bn5_1 = nn.BatchNorm2d(32)
      self.conv5_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
      self.bn5_2 = nn.BatchNorm2d(32)

      # Stage 3: Downsampling + two residual blocks with 64 filters
      self.conv6_down = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
      self.bn6_down = nn.BatchNorm2d(64)
      self.shortcut6 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
      self.bn6_shortcut = nn.BatchNorm2d(64)

      self.conv6_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
      self.bn6_2 = nn.BatchNorm2d(64)

      self.conv7_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
      self.bn7_1 = nn.BatchNorm2d(64)
      self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
      self.bn7_2 = nn.BatchNorm2d(64)

      # Global average pooling
      self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

      # Fully connected layer
      self.fc = nn.Linear(64, 10)

  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network"""


      ##########################
      #Res Net Structure FORWARD STARTS
      out = F.relu(self.bn1(self.conv1(xb)))

      # Stage 1
      residual = out
      out = F.relu(self.bn2_1(self.conv2_1(out)))
      out = self.bn2_2(self.conv2_2(out))
      out += residual
      out = F.relu(out)

      residual = out
      out = F.relu(self.bn3_1(self.conv3_1(out)))
      out = self.bn3_2(self.conv3_2(out))
      out += residual
      out = F.relu(out)

      # Stage 2
      residual = self.bn4_shortcut(self.shortcut4(out))
      out = F.relu(self.bn4_down(self.conv4_down(out)))
      out = self.bn4_2(self.conv4_2(out))
      out += residual
      out = F.relu(out)

      residual = out
      out = F.relu(self.bn5_1(self.conv5_1(out)))
      out = self.bn5_2(self.conv5_2(out))
      out += residual
      out = F.relu(out)

      # Stage 3
      residual = self.bn6_shortcut(self.shortcut6(out))
      out = F.relu(self.bn6_down(self.conv6_down(out)))
      out = self.bn6_2(self.conv6_2(out))
      out += residual
      out = F.relu(out)

      residual = out
      out = F.relu(self.bn7_1(self.conv7_1(out)))
      out = self.bn7_2(self.conv7_2(out))
      out += residual
      out = F.relu(out)

      # Global average pooling
      out = self.global_avg_pool(out)
      out = out.view(out.size(0), -1)

      # Fully connected layer
      out = self.fc(out)

# RESNET END HERE







# Implementation of  ResNext
class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      super(CIFAR10Model, self).__init__()
      self.cardinality = 8
      """
      Inputs:
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      ############################

  #RESNEXT IMPLEMNTATION # STARTS HERE
      # Initial convolutional layer
      cardinality = self.cardinality
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
      self.bn1 = nn.BatchNorm2d(16)

      # Stage 1: Grouped convolutions without downsampling
      self.conv2_1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, groups=cardinality, padding=0)
      self.bn2_1 = nn.BatchNorm2d(16)
      self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, groups=cardinality, padding=1)
      self.bn2_2 = nn.BatchNorm2d(16)

      self.conv3_1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, groups=cardinality, padding=0)
      self.bn3_1 = nn.BatchNorm2d(16)
      self.conv3_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, groups=cardinality, padding=1)
      self.bn3_2 = nn.BatchNorm2d(16)

      # Stage 2: Downsampling + grouped convolutions
      self.conv4_down = nn.Conv2d(16, 32, kernel_size=1, stride=2, groups=cardinality, padding=0)
      self.bn4_down = nn.BatchNorm2d(32)
      self.shortcut4 = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0)
      self.bn4_shortcut = nn.BatchNorm2d(32)

      self.conv4_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, groups=cardinality, padding=1)
      self.bn4_2 = nn.BatchNorm2d(32)

      self.conv5_1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, groups=cardinality, padding=0)
      self.bn5_1 = nn.BatchNorm2d(32)
      self.conv5_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, groups=cardinality, padding=1)
      self.bn5_2 = nn.BatchNorm2d(32)

      # Stage 3: Downsampling + grouped convolutions
      self.conv6_down = nn.Conv2d(32, 64, kernel_size=1, stride=2, groups=cardinality, padding=0)
      self.bn6_down = nn.BatchNorm2d(64)
      self.shortcut6 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
      self.bn6_shortcut = nn.BatchNorm2d(64)

      self.conv6_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=cardinality, padding=1)
      self.bn6_2 = nn.BatchNorm2d(64)

      self.conv7_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, groups=cardinality, padding=0)
      self.bn7_1 = nn.BatchNorm2d(64)
      self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=cardinality, padding=1)
      self.bn7_2 = nn.BatchNorm2d(64)

      # Global average pooling
      self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

      # Fully connected layer
      self.fc = nn.Linear(64, OutputSize)

  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network


      # """

      # #RESNEXT IMPLEMENTATION
      out = F.relu(self.bn1(self.conv1(xb)))

      # Stage 1
      residual = out
      out = F.relu(self.bn2_1(self.conv2_1(out)))
      out = self.bn2_2(self.conv2_2(out))
      out += residual
      out = F.relu(out)

      residual = out
      out = F.relu(self.bn3_1(self.conv3_1(out)))
      out = self.bn3_2(self.conv3_2(out))
      out += residual
      out = F.relu(out)

      # Stage 2
      residual = self.bn4_shortcut(self.shortcut4(out))
      out = F.relu(self.bn4_down(self.conv4_down(out)))
      out = self.bn4_2(self.conv4_2(out))
      out += residual
      out = F.relu(out)

      residual = out
      out = F.relu(self.bn5_1(self.conv5_1(out)))
      out = self.bn5_2(self.conv5_2(out))
      out += residual
      out = F.relu(out)

      # Stage 3
      residual = self.bn6_shortcut(self.shortcut6(out))
      out = F.relu(self.bn6_down(self.conv6_down(out)))
      out = self.bn6_2(self.conv6_2(out))
      out += residual
      out = F.relu(out)

      residual = out
      out = F.relu(self.bn7_1(self.conv7_1(out)))
      out = self.bn7_2(self.conv7_2(out))
      out += residual
      out = F.relu(out)

      # Global average pooling
      out = self.global_avg_pool(out)
      out = out.view(out.size(0), -1)

      # Fully connected layer
      out = self.fc(out)



      return out
      #RESNEXTEND HERE



## DENSENET
class CIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize, growth_rate=12, num_layers_per_block=4):
        super().__init__()

        self.growth_rate = growth_rate
        self.num_layers_per_block = num_layers_per_block
        self.num_classes = OutputSize

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Dense blocks
        num_features = 16  # Initial number of features

        # First dense block
        self.dense_block1 = self._make_dense_block(num_features, num_layers_per_block)
        num_features += num_layers_per_block * growth_rate

        # First transition layer
        self.trans1 = self._make_transition_layer(num_features)
        num_features = num_features // 2

        # Second dense block
        self.dense_block2 = self._make_dense_block(num_features, num_layers_per_block)
        num_features += num_layers_per_block * growth_rate

        # Second transition layer
        self.trans2 = self._make_transition_layer(num_features)
        num_features = num_features // 2

        # Third dense block
        self.dense_block3 = self._make_dense_block(num_features, num_layers_per_block)
        num_features += num_layers_per_block * growth_rate

        # Final batch norm
        self.final_bn = nn.BatchNorm2d(num_features)

        # Global average pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, OutputSize)

    def _make_dense_layer(self, in_features):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, self.growth_rate, kernel_size=3, stride=1, padding=1)
        )

    def _make_dense_block(self, in_features, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(self._make_dense_layer(in_features + i * self.growth_rate))
        return nn.ModuleList(layers)

    def _make_transition_layer(self, in_features):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features // 2, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, xb):
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(xb)))

        # First dense block
        for layer in self.dense_block1:
            new_features = layer(out)
            out = torch.cat([out, new_features], 1)

        # First transition
        out = self.trans1(out)

        # Second dense block
        for layer in self.dense_block2:
            new_features = layer(out)
            out = torch.cat([out, new_features], 1)

        # Second transition
        out = self.trans2(out)

        # Third dense block
        for layer in self.dense_block3:
            new_features = layer(out)
            out = torch.cat([out, new_features], 1)

        # Final layers
        out = F.relu(self.final_bn(out))
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

#DENSENET NETWORK ENDS



model = CIFAR10Model(InputSize = 3*32*32, OutputSize =10)


if __name__ == "__main__":
    import torch.onnx

    # Create model instance
    model = CIFAR10Model(InputSize=3*32*32, OutputSize=10)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "densnet_architecture.onnx",
        export_params=True,
        opset_version=11
    )

    print("Model exported to densenext_architecture.onnx")

    # View the architecture
    import netron
    netron.start('densenet_architecture.onnx')
