import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

'''
In this file you will write end-to-end code to train a neural network to categorize fashion-mnist data
'''


'''
PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted.
'''
transform = transforms.Compose([  # Use transforms to convert images to tensors and normalize them
    transforms.ToTensor(),              
    transforms.Normalize((0.5,), (0.5,))])
                               
batch_size = 64

'''
PART 2:
Load the dataset. Make sure to utilize the transform and batch_size you wrote in the last section.
'''

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# used ChatGPT to search up more about FashionMNIST + the classes 
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


'''
PART 3:
Design a neural network in PyTorch. Architecture is up to you, but please ensure that the model achieves at least 80% accuracy.
Do not directly import or copy any existing models from other sources, spend time tweaking things.
'''

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,500)
        self.fc2 = nn.Linear(500,10) 
        
    # ChatGPT to debug
    def forward(self, x):
        x = x.view(-1, 784) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
net = Net()

'''
PART 4:
Choose a good loss function and optimizer
'''

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

'''
PART 5:
Train your model!
'''
num_epochs = 5

for epoch in range(num_epochs):  
    running_loss = 0.0
    total_reward = 0.0
    all_losses = [] 
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        all_losses.append(loss.item())


    print(f"Training loss: {running_loss}")

print('Finished Training')


'''
PART 6:
Evalute your model! Accuracy should be greater or equal to 80%

'''

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # ChatGPT to debug 

print('Accuracy: ', correct/total)

'''
PART 7:
Check the written portion. You need to generate some plots. 
'''

plt.plot(all_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

# Used https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model to determine
# basic outline for showing classified images
# used ChatGPT to get the syntax for generating the figures and debugging 
def imageshow(img, predicted, true):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'Predicted: {classes[predicted]} \nTrue: {classes[true]}')
    plt.show()


fig, axs = plt.subplots(1, 2) # ChatGPT 
# Incorrect 
for images, labels in testloader:
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    for i in range(len(images)):
        if predicted[i] != labels[i]:
            axs[0].axis('off')
            axs[0].imshow(images[i].squeeze(), cmap='gray')
            axs[0].set_title(f'Predicted: {classes[predicted[i]]} \nTrue: {classes[labels[i]]}')
            break

# Correct
axs[1].axis('off')
for images, labels in testloader:
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    for i in range(len(images)):
        if predicted[i] == labels[i]:
            axs[1].imshow(images[i].squeeze(), cmap='gray')
            axs[1].set_title(f'Predicted: {classes[predicted[i]]} \nTrue: {classes[labels[i]]}')
            break

plt.show()

