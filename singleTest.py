from PIL import ImageTk, Image  
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as C
import torch.nn.utils.prune as prune

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return C.pad(image, padding, 0, 'constant')
    
transform = transforms.Compose(
    [
     SquarePad(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize((64, 64)), 
     transforms.Grayscale(3) ])
import torch.optim as optim
import warnings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
IMAGE_DATA_PATH = "facedata/singleTest/disgust/Disgust_15.jpg"


classes = ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

 
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
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
# get some random training images
    images = Image.open(IMAGE_DATA_PATH)
    images = transform(images)
# show images
            
    TEST = './57.pth'
    
    #dataiter = iter(testloader)
    #images, labels = next(dataiter)
    
    #print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(5)))
    net = Net()
    net.load_state_dict(torch.load(TEST, map_location=torch.device('cpu')))
    with torch.no_grad():
        outputs = net(images.unsqueeze(0))
        softmaxpredicted = torch.softmax(outputs, 1)
        mostPredicted = torch.argmax(outputs, 1)

    print('Predicted Probablities:')
    for j in range(len(classes)):
         print(classes[j], softmaxpredicted[0][j])
    print('Argmax Prediction:', mostPredicted)
    imshow(torchvision.utils.make_grid(images))







