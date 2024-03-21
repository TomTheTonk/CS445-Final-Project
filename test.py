import os
from matplotlib.pyplot import imshow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as C

# Hyper parameters
num_epochs = 20
batchsize = 1000
lr = 0.001
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return C.pad(image, padding, 0, 'constant')
EPOCHS = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "facedata/archive/train/"
TEST_DATA_PATH = "facedata/archive/test/"
TRANSFORM_IMG = transforms.Compose([
    SquarePad(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=6)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=6) 

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batchsize,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batchsize,
                                          shuffle=False)

# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conlayer1 = nn.Sequential(
            nn.Conv2d(3,3,3),
            nn.Sigmoid(),
            nn.MaxPool2d(2))
        self.conlayer2 = nn.Sequential(
            nn.Conv2d(3,3,3),
            nn.Sigmoid(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(11532,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10))

    def forward(self, x):
        out = self.conlayer1(x)
        out = self.conlayer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out



cnn = CNN()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        #images = images.permute(1, 0, 2, 3)
        outputs = cnn(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %
                  (epoch+1,num_epochs,i+1,len(train_data)//batchsize,loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var)
correct = 0
total = 0

for images, labels in test_loader:
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on test images: %.6f%%' % (100.0*correct/total))


#Save the Trained Model
torch.save(cnn.state_dict(),'cnn.pkl')
