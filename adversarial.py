import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import os
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

if torch.backends.mps.is_available():
	device=torch.device("mps")
elif torch.cuda.is_available():
	device=torch.device("cuda")
else:
	device=torch.device("cpu")

print(device)

# define CNN for a 3-class problem with input size 160x160 images
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(256 * 5 * 5, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 3)
		self.relu = nn.ReLU()
		self.final_activation = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = self.pool(self.relu(self.conv3(x)))
		x = self.pool(self.relu(self.conv4(x)))
		x = self.pool(self.relu(self.conv5(x)))
		x = x.view(-1, 256 * 5 * 5)
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		x = self.final_activation(x)
		return x

# Load dataset
train_dir = './adv_data/train'
test_dir = './adv_data/test'
image_size = 160
batch_size = 16
workers = 0

class CropToSmallerDimension(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img):
        # Get the original image size
        width, height = img.size
        # Determine the smaller dimension
        smaller_dimension = min(width, height)
        # Crop the image to the smaller dimension
        return transforms.CenterCrop(smaller_dimension)(img)

train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms.Compose([CropToSmallerDimension(256),transforms.ToTensor(),transforms.Resize(image_size)]))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms.Compose([CropToSmallerDimension(256),transforms.ToTensor(),transforms.Resize(image_size)]))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

print('Number of training images: {}'.format(len(train_dataset)))
print('Number of test images: {}'.format(len(test_dataset)))
print('Detected Classes are: ', train_dataset.classes) # classes are detected by folder structure

# Define the attack
# Used ChatGPT to debug errors related to the device 
def FGSM(model, image, alpha, data_grad):
    model.eval()
    # ChatGPT
    image, labels = image.to(device), data_grad.to(device)
    image.requires_grad = True
    outputs = model(image)
    loss = F.nll_loss(outputs, labels)
    model.zero_grad()
    loss.backward()
    sign_data_grad = image.grad.data.sign()
    perturbed_image = image + alpha * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Used ChatGPT to get an outline of the code for attack
# Used ChatGPT to debug 
def PGD(model, image, labels, epsilon, iterations, alpha):
    model.eval()
   
    image, labels= image.to(device), labels.to(device)
    image.requires_grad = True
    perturbed_image = image.clone().detach().requires_grad_(True)
    # ChatGPT   
    for _ in range(iterations):
        outputs = model(perturbed_image)
        loss = F.nll_loss(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data.sign()
        perturbed_image = perturbed_image + alpha*data_grad
        perturbed_image.data = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)
        perturbed_image = torch.clamp(image + perturbed_image.data, min=0, max=1).detach_().requires_grad_()
    return perturbed_image

     
net = Net()
net.to(device)

#Train the network
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
epochs = 100
running_loss = 0
train_losses, test_losses = [], []
i=0

for epoch in tqdm(range(epochs)):
        for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = net(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

# Save the model
torch.save(net.state_dict(), 'model.pth')


# Test the model
net.load_state_dict(torch.load('model.pth', map_location="cpu"))
net.to(device)
torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

correct=[]
net.eval()
accuracy = 0

for inputs, labels in tqdm(test_dataloader):
    inputs, labels = inputs.to(device), labels.to(device)  
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    accuracy += (predicted == labels).sum().item()
    correct.append((predicted == labels).tolist())
print('Accuracy of the network on the test images: %d %%' % (100 * accuracy / len(test_dataset)))


# Test the model with adversarial examples
# Save adversarial examples for each class using FGSM with (alpha = 0.001, 0.01, 0.1)
# Save one adversarial example for each class using PGD with (eps = 0.01, 0.05, and 0.1, alpha = 2/255, respectively, iterations = 50)


# fgsm
epsilon_values_fgsm = [0.001, 0.01, 0.1]
for epsilon in epsilon_values_fgsm:
        adversarial_accuracy = 0
        total = 0
        for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                fgsm_perturbed_inputs = FGSM(net, inputs, epsilon, labels)
                outputs = net(fgsm_perturbed_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total+=labels.size(0)
                adversarial_accuracy += (predicted == labels).sum().item()
        print('Accuracy of the network on FGSM adversarial test images with epsilon = ', epsilon,': %d %%' % (100 * adversarial_accuracy / total))

 
# pgd
epsilon_values_pgd = [0.01, 0.05, 0.1]
alpha_pgd = 2/255
for epsilon in epsilon_values_pgd:
        adversarial_accuracy2 = 0
        total2 = 0
        pgd_examples = []
        for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                pgd_perturbed_inputs = PGD(net, inputs, labels, epsilon, 50, alpha_pgd)
                outputs = net(pgd_perturbed_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total2+=labels.size(0)
                adversarial_accuracy2 += (predicted == labels).sum().item()
                pgd_examples.append((pgd_perturbed_inputs, labels))
        print('Accuracy of the network on PGD adversarial test images:', epsilon,' %d %%' % (100 * adversarial_accuracy2 / total2))

# Used ChatGPT to debug img.cpu() related errors
# Used https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
# as a reference for displaying photos 
def imageshow(img, title):
        img = img.cpu()
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(title)
        plt.show()

def test(fgsm_alphas, pgd_alphas):
    # Used ChatGPT to debug because only oranges were showing at first
    # instead of the all 3 classes         
    images_to_display = []  
    labels_to_display = [] 
    class_counts = [0] * len(test_dataset.classes)  
        
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        class_index = label
        if class_counts[class_index] == 0:  # ChatGPT 
            images_to_display.append(image.unsqueeze(0))
            labels_to_display.append(label)
            class_counts[class_index] += 1
        if all(count > 0 for count in class_counts):  # ChatGPT 
            break
        
    for i in range(len(images_to_display)):
        imageshow(vutils.make_grid(images_to_display[i], normalize=True), f'Original Image (Predicted: {determine_label(labels_to_display[i])})')

    # ChatGPT 
    for e in fgsm_alphas:
        for i in range(len(images_to_display)):
            image, label = images_to_display[i][0], labels_to_display[i]
            fgsm_perturbed_inputs = FGSM(net, image.unsqueeze(0), e, torch.tensor([label]))
            outputs = net(fgsm_perturbed_inputs)
            _, predicted = torch.max(outputs.data, 1)
            imageshow(vutils.make_grid(fgsm_perturbed_inputs, normalize=True), f'FGSM adversarial image (Predicted: {determine_label(labels_to_display[i])}, Alpha: {e})')
    
    for p in pgd_alphas:
        for i in range(len(images_to_display)):
            image, label = images_to_display[i][0], labels_to_display[i]
            pgd_perturbed_inputs = PGD(net, image.unsqueeze(0), torch.tensor([label]), 0.075, 50, p)
            outputs = net(pgd_perturbed_inputs)
            _, predicted = torch.max(outputs.data, 1)
            imageshow(vutils.make_grid(pgd_perturbed_inputs, normalize=True), f'PGD adversarial image (Predicted: {determine_label(labels_to_display[i])}, Alpha: {p})')

# prints either apple, banana, or oranges 
def determine_label(label):
    if (label == 0):
        return 'apple'
    elif (label == 1):
        return 'banana'
    elif (label == 2):
        return 'orange'
                

fgsm_alphas = [0.001, 0.01, 0.1]
pgd_alphas = [0.01, 0.05, 0.1]
test(fgsm_alphas, pgd_alphas)


optimizer = optim.Adam(net.parameters(), lr=0.001)
epsilon = 0.075
alpha = 2/255
iterations = 50
epoch_running_loss = 0

num_epochs = 10
for epoch in range(num_epochs):
    epoch_running_loss = 0.0
    net.train()

    for inputs, labels in tqdm(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        perturbed_inputs = PGD(net, inputs, labels, epsilon, iterations, alpha)
        optimizer.zero_grad()
        outputs = net(perturbed_inputs)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()
print(f"Epoch {epoch + 1}/{num_epochs}", epoch_running_loss/len(test_dataloader))

net.eval() # Used ChatGPT to debug 
correct_clean = 0
total_clean = 0
correct_pgd = 0
total_pgd = 0

# clean 
with torch.no_grad(): # ChatGPT 
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total_clean += labels.size(0)
        correct_clean += (predicted == labels).sum().item()
# pgd 
with torch.no_grad():
    for inputs, labels in pgd_examples:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total_pgd += labels.size(0)
        correct_pgd += (predicted == labels).sum().item()

clean_test_accuracy = correct_clean / total_clean * 100
pgd_test_accuracy = correct_pgd / total_pgd * 100


print(f"Accuracy on clean test images: {clean_test_accuracy:.2f}%")
print(f"Accuracy on PGD adversarial test images: {pgd_test_accuracy:.2f}%")



