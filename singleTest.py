from PIL import ImageTk, Image  
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as C
import torch.nn.utils.prune as prune
"""
SquarePad, takes an image and squares it by adding padding so each image inputed is similar

:param image: image given to be given square dimensions 
:type: Image
:return: Image that is squared
:rtype: image
"""
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return C.pad(image, padding, 0, 'constant')
     
#Transformation that is applied to an image given
#First Squared
#Transformed to Tensor
#The tensor's values are normalized
#The image is resized
#The image is then greyscaled
transform = transforms.Compose(
    [
     SquarePad(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize((64, 64)), 
     transforms.Grayscale(3) ])

import torch.optim as optim
import warnings
#Set the default device to the gpu if the machine running has one
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#ignore warnings this is for if  __name__ == __main__
warnings.filterwarnings("ignore")
#val to check if the user wants a defualt 
val = -1
while(val != 1 and val != 0):
    val = int(input("Enter 0 to enter your own image or 1 to run on the default: "))
    if val == 0:
        file = input("Enter the image directory to test on: ") 
        images = Image.open(file)
    elif val == 1:
        #Default Image path to run on
        IMAGE_DATA_PATH = "facedata/archive/test/disgust/Disgust_17.jpg"
        images = Image.open(IMAGE_DATA_PATH)

#All the classes to be predicted
classes = ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
"""
class Net(nn.Module):
    #Define the module functions
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (3,3)) 
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, (3,3))
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, (3,3))
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(128, 256, (3,3))
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, (2,2))
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(9216, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.25)
       
    #calls for the classes functions, not all functions are called 
    def forward(self, x):
        x = self.pool(((F.relu(self.conv1(x)))))
        x = (self.batchnorm1(x))
        x = self.pool(((F.relu(self.conv2(x)))))
        x = (self.batchnorm2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = (self.batchnorm3(x))
        #x = self.pool(F.relu(self.conv4(x)))
        #x = self.batchnorm4(x)
        #x = self.pool(F.relu(self.conv5(x)))
        #x = self.batchnorm5(x)
        x = torch.flatten(x, 1) # flatten all dimensi        ons except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""
class Net(nn.Module):
    def __init__(self):
        #Define the module functions
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, (3,3)) 
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(256, 512, (3,3))
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1006, (3,3))
        self.batchnorm3 = nn.BatchNorm2d(1006)
        self.conv4 = nn.Conv2d(128, 256, (3,3))
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, (2,2))
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(36216, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 10)
        self.dropout = nn.Dropout(0.25)
        
    #calls for the classes functions, not all functions are called 
    def forward(self, x):
        x = self.pool(((F.relu(self.conv1(x)))))
        x = (self.batchnorm1(x))
        x = self.pool(((F.relu(self.conv2(x)))))
        x = (self.batchnorm2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = (self.batchnorm3(x))
        #x = self.pool(F.relu(self.conv4(x)))
        #x = self.batchnorm4(x)
        #x = self.pool(F.relu(self.conv5(x)))
        #x = self.batchnorm5(x)
        x = torch.flatten(x, 1) # flatten all dimensi        ons except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    #Transform the single image
    images = transform(images)       
    #Model used
    TEST = './57.pth'
    #Define the models functions
    net = Net()
    #Load the saved model
    net.load_state_dict(torch.load(TEST, map_location=torch.device('cpu')))
    #With the model not to learn have it run on the image
    with torch.no_grad():
        outputs = net(images.unsqueeze(0))
        #Get the probablities for all labels
        softmaxpredicted = torch.softmax(outputs, 1)
        #Get the label it thinks is most accurate
        mostPredicted = torch.argmax(outputs, 1)
    #Print all the info given
    print('Predicted Probablities:')
    for j in range(len(classes)):
         print(classes[j], softmaxpredicted[0][j])
    print('Argmax Prediction:', mostPredicted)
    #Print the image after it is done to show the transformation
    imshow(torchvision.utils.make_grid(images))







