import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#--- hyperparameters ---
N_EPOCHS = 50
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.001


#--- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'

NUM_CHANNELS = 3
WIDTH = HEIGHT = 28

# --- Dataset initialization ---

# We transform image files' contents to tensors
# Plus, we can add random transformations to the training data if we like
# Think on what kind of transformations may be meaningful for this data.
# Eg., horizontal-flip is definitely a bad idea for sign language data.
# You can use another transformation here if you find a better one.
train_transform = transforms.Compose([
                                    transforms.ColorJitter(
                                                            brightness = (0.5, 1.5), 
                                                            contrast = (0.5, 1.5), 
                                                            saturation = (0.5, 1.5), 
                                                            hue = (-0.1, 0.1)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

test_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
dev_set   = datasets.ImageFolder(DATA_DIR % 'dev',   transform=test_transform)
test_set  = datasets.ImageFolder(DATA_DIR % 'test',  transform=test_transform)


# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dataset=dev_set, shuffle=False)

#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        
        self.model = nn.Sequential(
            # ImgIn shape=(?, 28, 28, 1)
            nn.Conv2d(3, 32, 3), # Conv -> (?, 26, 26, 32)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            nn.MaxPool2d(3, 3, 1), # Pool -> (?, 9, 9, 32) 

            nn.Conv2d(32, 64, 3), # Conv -> (?, 7, 7, 64)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2, 1), # Pool -> (?, 4, 4, 64)

            nn.Flatten(),
            
            nn.Linear(4 * 4 * 64, 512, bias=True), # 4x4x64 inputs -> 512 outputs
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, num_classes, bias=True) # 512 inputs -> 10 outputs
        )

    def forward(self, x):
        # self.model.apply(weights_init)
        return self.model(x)


# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
#         torch.nn.init.zeros_(m.bias)

#--- set up ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)

# WRITE CODE HERE
loss_function = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay=0.05)

#--- training ---
dev_loss = np.iinfo(int).max

def validation_loss(model, dev_loader):
    total_correct = 0
    total_loss = 0
    with torch.no_grad():
        for (data, target) in dev_loader:
            data, target = data.to(device), target.to(device)

            output = model.forward(data)
            total_loss += loss_function(output, target)
            total_correct += torch.argmax(output, dim=1) == target

    return total_loss, total_correct

for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0

    total = 0
    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # WRITE CODE HERE
        optimizer.zero_grad()

        output = model.forward(data)
        predicted = torch.argmax(output, dim=1)

        loss = loss_function(output, target)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        total += BATCH_SIZE_TRAIN
        train_correct += torch.sum(predicted == target)

        # print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
        #     (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
        #     100. * train_correct / total, train_correct, total))
    
    print('Training: Epoch %d - Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
        (epoch, train_loss / len(train_loader), 
        100. * train_correct / total, train_correct, total))

    # WRITE CODE HERE
    # Please implement early stopping here.
    # You can try different versions, simplest way is to calculate the dev error and
    # compare this with the previous dev error, stopping if the error has grown.

    dev_loss_current, dev_correct = validation_loss(model, dev_loader)
    print('Validation: Loss: %.4f | Acc: %.3f%% (%d/%d)' % (dev_loss_current / len(dev_loader), 
        100. * dev_correct / len(dev_loader), dev_correct, len(dev_loader)))

    if (dev_loss_current > dev_loss):
        print("Early Stopping: Epoch %d" % epoch)
        break
    else: dev_loss = dev_loss_current
    


#--- test ---
test_loss = 0
test_correct = 0
total = 0

with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # WRITE CODE HERE
        output = model.forward(data)
        predicted = torch.argmax(output, dim=1)

        loss = loss_function(output, target)
        test_loss += loss.item()

        total += BATCH_SIZE_TRAIN
        test_correct += torch.sum(predicted == target)

        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
              (batch_num, len(test_loader), test_loss / (batch_num + 1), 
               100. * test_correct / total, test_correct, total))

# Report

# 1
# Neither regularization, nor optimization methods, LR = 0.1, SGD
# (but used Early Stopping to avoid overfitting and Nonlinearity):
# 2 Convolutional layers (CL) with 2 Linear (LL) ones
# Test Accuracy: 88%, trained (on the average) 3-4 epochs. Train Accuracy: 99%, Validation Accuracy: 87%

# 2 Regularization
# Added Dropout with 0.05 probability for CL and 0.2 for LL
# LR = 0.1, Test Accuracy: 90%, trained (on the average) 3-4 epochs. Train Accuracy: 99%, Validation Accuracy: 89%
# Other LR values decrease Test Accuracy

# 3 Optimization
# Tried Adam Optimizer
# LR = 0.001, Test Accuracy: 91.5%, trained 6-7 epochs. Train Accuracy: 99%, Validation Accuracy: 92% 

# 4 Regularization
# Tried Data Augmentation methods
# ColorJitter + RandomPerspective -> Test accuracy: 90%
# ColorJitter + RandomAffine -> Test accuracy: 91%
# ColorJitter + RandomResizedCrop -> Test accuracy: 90%, but Val Acc: 94%
# RandomResizedCrop -> 86%
# ColorJitter + GaussianBlur -> 85%
# ColorJitter -> 91 - 92% 

# 5 Optimization 
# Batch Normalization with input normalization
# Rapidly decreased training time, 2-3 epochs
# Test Accuracy still 91%, Validation Accuracy: 90% and Training Acuuracy: 99%

# 6 Optimization + Regularization
# Tried different optimizers with/without L2 or weight_decay
# RMSprop: LR = 0.001: Test Accuracy: 92%, Validation Accuracy: 91% and Training Accuracy: 99%
# AdamW: LR = 0.001: Test Accuracy: 91-92%, Validation Accuracy: 91% and Training Accuracy: 99.5%
# Adam: LR = 0.001: Test Accuracy: 90%, Validation Accuracy: 88% and Training Accuracy: 99%
# Adamgrad (8 epochs, too much): LR = 0.001: Test Accuracy: 88%, Validation Accuracy: 87% and Training Accuracy: 99%
# Adamgrad: LR = 0.01: Test Accuracy: 91%, Validation Accuracy: 91% and Training Accuracy: 99%
# SGD: LR = 0.1: Test Accuracy: 90%, Validation Accuracy: 90% and Training Accuracy: 99%
# SGD (weight_decay (L2) = 0.005): LR = 0.5: Test Accuracy: 91%, Validation Accuracy: 91% and Training Accuracy: 97% 
# SGD (weight_decay (L2) = 0.001): LR = 0.1: Test Accuracy: 93%, Validation Accuracy: 92% and Training Accuracy: 99% 
# (Bad) Adam (weight_decay = 0.001): LR = 0.1: Test Accuracy: 60%, Validation Accuracy: 51% and Training Accuracy: 67% 

# 7 Optimization
# Tried Weight Initialization, but average accuracy for training and validation set is nearly 5%, model is not training
# Used different oprimizers, different LR, different methods (uniform, normal, etc.), still accuracy is too low


