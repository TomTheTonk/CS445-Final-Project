#Last file used for training
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
#Set the path for the training set folder
TRAIN_DATA_PATH = "facedata/archive2/train/"
#Set the path for the test set folder
TEST_DATA_PATH = "facedata/archive2/test/"
#Size of the gradients to keep in memory 
batch_size = 55
#How many times to run throught the training data
epochRun = 200
#How quickly to learn from each batch size
learnRate = 0.005
#Create the training set object
trainset = torchvision.datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
#Create the test set object
testset = torchvision.datasets.ImageFolder(TEST_DATA_PATH, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
#All the labels for the images possiable
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
        self.fc1 = nn.Linear(36216, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)
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
    #Print all the data for the model training 
    print("Epoch", epochRun)
    print("Batch Size", batch_size)
    print("Learn Rate", learnRate)
    #highest accuracy the model got on the test set
    highestAccur = 0
    #the accuracy of the current model on the test set
    accuracy = 0
    #How long it has been since the model improved
    count = 0
    #The epochrun with the highest accuracy on the testset
    highestEpoch = 0
    #define the model
    net = Net()
    #Check if cuda is available if so run on cuda
    if torch.cuda.is_available():
        net.cuda()
        print("Running on GPU")
    #Loss function
    criterion = nn.CrossEntropyLoss()
    #How the model learns
    optimizer = optim.SGD(net.parameters(), lr=learnRate, momentum=0.9)
    for epoch in range(epochRun):  # loop over the dataset multiple times for the amount of epechRun
        running_loss = 0.0
        #Go over the training data
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            #int storing number of correct and total images correct
            correct = 0
            total = 0
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
                #Save the current state of the model
                PATH = './Last.pth'
                torch.save(net.state_dict(), PATH)
                #Run the current model on the test set with no gradient IE it does not learn
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
                    #Print the model's accuracy on the test set
                    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
                    accuracy = 100 * correct // total
                #If the model is more accurate then the last epochrun on the test set
                if highestAccur < accuracy:
                    highestEpoch = epoch + 1
                    highestAccur = accuracy
                    #Save the model
                    BEST = './Best.pth'
                    torch.save(net.state_dict(), BEST)
                    count = 0
                else:
                    #If the model is not an improvement on the last on the test set increase the count
                    count = count + 1
                    #If the count is equal to a hard coded value
                    #Currently set to never fire
                    if count == -1:
                        #Was code to load the best epoch run
                        '''
                        net.load_state_dict(torch.load(BEST))
                        print(f"No improvement in {count} runs, loaded best run to retrain")
                        '''
                        #epoch = highestEpoch - 1

                        #prunes all layers of the model
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
                        #Print that the model has been pruned
                        print("Pruned 20 Percent of structure")
                        #Reset count since it has been pruned
                        count = 0
                #Print the epoch with the highest accuracy with the testset
                print("Highest Accurracy is", highestAccur,"%","from Epoch", highestEpoch)
                        
                        
                    
    #Reset correct and total
    correct = 0
    total = 0
    print('Finished Training')
    #Save the Last run of the model
    LAST = './LAST.pth'
    torch.save(net.state_dict(), LAST)
    #Grab the testset's labels and iamges
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    #Print ground truth for the first 10 images
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(10)))
    net = Net()
    #Load the Best run of the model 
    net.load_state_dict(torch.load(BEST))
    #run over the 10 images with the model
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
    #Print the outputs
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                            for j in range(10)))
    #Run over all the testing data
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

    #Run over all the testing data for individual classes correctness
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





