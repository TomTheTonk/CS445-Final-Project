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
TEST_DATA_PATH = "facedata/archive2/test/"
batch_size = 5
testset = torchvision.datasets.ImageFolder(TEST_DATA_PATH, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

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
    dataiter = iter(testloader)
    images, labels = next(dataiter)
# show images
            
    TEST = './57.pth'
    #dataiter = iter(testloader)
    #images, labels = next(dataiter)
    
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(5)))
    net = Net()
    net.load_state_dict(torch.load(TEST, map_location=torch.device('cpu')))
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                            for j in range(5)))
    imshow(torchvision.utils.make_grid(images))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    accuracy = 100 * correct // total
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    # print images






