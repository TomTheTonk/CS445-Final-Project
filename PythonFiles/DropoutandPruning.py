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
TRAIN_DATA_PATH = "facedata/archive2/train/"
TEST_DATA_PATH = "facedata/archive2/test/"
batch_size = 55
epochRun = 200
learnRate = 0.005
flag = False
trainset = torchvision.datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(TEST_DATA_PATH, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('anger', 'fear', 'happy', 'sad', 'surprised', 'neutral')

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
        self.conv3 = nn.Conv2d(128, 400, (3,3))
        self.batchnorm3 = nn.BatchNorm2d(400)
        self.conv4 = nn.Conv2d(256, 512, (3,3))
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(256, 512, (2,2))
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(14400, 120)
        self.batchnormlin1 = nn.BatchNorm2d(120)
        self.fc2 = nn.Linear(120, 84)
        self.batchnormlin2 = nn.BatchNorm2d(84)
        self.fc3 = nn.Linear(84, 10)
        self.batchnormlin3 = nn.BatchNorm2d(10)
        self.dropout = nn.Dropout(0.25)
       
    
    def forward(self, x):
        x = self.pool(((F.relu(self.conv1(x)))))
        x = self.dropout(self.batchnorm1(x))
        x = self.pool(((F.relu(self.conv2(x)))))
        x = self.dropout(self.batchnorm2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(self.batchnorm3(x))
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
    #dataiter = iter(trainloader)
    #images, labels = next(dataiter)
    print("Epoch", epochRun)
    print("Batch Size", batch_size)
    print("Learn Rate", learnRate)
# show images
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    #imshow(torchvision.utils.make_grid(images))

    
    highestAccur = 0
    accuracy = 0
    count = 0
    highestEpoch = 0
    net = Net()

    if torch.cuda.is_available():
        net.cuda()
        print("Running on GPU")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learnRate, momentum=0.9)
    for epoch in range(epochRun):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            correct = 0
            total = 0
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                PATH = './test.pth'
                torch.save(net.state_dict(), PATH)
                dataiterTest = iter(testloader) 
                imagesTest, labelsTest = next(dataiterTest)
                with torch.no_grad():
                    for testData in testloader:
                        imagesTest, labelsTest = testData
                        imagesTest, labelsTest = imagesTest.to(device), labelsTest.to(device)
                    # calculate outputs by running images through the network
                        testOutputs = net(imagesTest)
                    # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(testOutputs.data, 1)
                        total += labelsTest.size(0)
                        correct += (predicted == labelsTest).sum().item()

                    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
                    accuracy = 100 * correct // total
                if highestAccur < accuracy:
                    highestEpoch = epoch + 1
                    highestAccur = accuracy
                    
                    BEST = './Best.pth'
                    torch.save(net.state_dict(), BEST)
                    count = 0
                else:
                    count = count + 1
                    if count == 1000:
                        '''
                        net.load_state_dict(torch.load(BEST))
                        print(f"No improvement in {count} runs, loaded best run to retrain")
                        '''
                        #epoch = highestEpoch - 1
                        
                        paramenter_to_prune = (
                            (net.conv1, 'weight'),
                            (net.conv2, 'weight'),
                            (net.conv3, 'weight'),
                            (net.fc1, 'weight'),
                            (net.fc2, 'weight'),
                            (net.fc3, 'weight'),
                        )
                        prune.global_unstructured(
                            paramenter_to_prune,
                            pruning_method=prune.L1Unstructured, 
                            amount=0.2,
                        )
                        print("Pruned 20 Percent of structure")
                        count = 0
                print("Highest Accurracy is", highestAccur,"%","from Epoch", highestEpoch)
                        
                        
                    
            
    correct = 0
    total = 0
    print('Finished Training')
    LAST = './LAST.pth'
    TEST = './57.pth'
    torch.save(net.state_dict(), LAST)
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(10)))
    net = Net()
    net.load_state_dict(torch.load(BEST))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                            for j in range(10)))
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

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
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





