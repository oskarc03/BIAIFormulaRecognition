import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#function for printing accuracy and loss for training and validation
def model_acc_and_loss_plt(train_losses, train_acc, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy', color='green')
    plt.plot(val_accuracies, label='Val Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

#displaying confusion matrix for analysis
def conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

#basic parameters
BATCH_SIZE = 32 #training samples processed simultaneously
NUM_CLASSES = 8 #number of class (f1 cars)
LEARNING_RATE = 0.001 #learning tempo
DATA_DIR = "Formula One Cars"

#gpu preffered
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#resizing images and converting them to tensors
all_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010])
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=all_transform)

#division for different sets
#80% for training
#20% for validation
train_size = int(0.8 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
rest_size = len(full_dataset) - train_size - val_size

#model trains and validates on random data
train_dataset, val_dataset, rest_size = random_split(full_dataset, [train_size, val_size, rest_size])

#data to batches
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle = False)
#test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Klasy: ", full_dataset.classes)

#simple convolutional network with 4 Conv2D layers, 2 pooling 
#and 2 dense (Linear) layers. At the end we have NUM_CLASSES neurons – one per class.
#convolutional layers proceed a certain are of an image (kernel 3x3)
#then it creates map of characteristics containg of 32 channels
#maxpooling chooses the most important data
#relu makes the function non-linear which makes the neural network work much better (can test)
class CNN(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 32, 3)
        self.conv_layer2 = nn.Conv2d(32, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2, 2)

        self.conv_layer3 = nn.Conv2d(32, 64, 3)
        self.conv_layer4 = nn.Conv2d(64, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        #returns logits of classes (before softmax)
        return out


#initializing model
#crossentropy for multiple classes
model = CNN(NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

#lists for further data processing
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

#training
for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0
    #training mode
    model.train()

    #processing batches
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) #working on gpu or cpu if gpu not available

        optimizer.zero_grad()       #zeroing the gradient
        outputs = model(images)     #batches through model
        loss = criterion(outputs, labels)   #calculating loss
        loss.backward()     #calculating gradient
        optimizer.step()    #weights update

        running_loss += loss.item()     #loss sum in epoch

        _, predicted = torch.max(outputs, 1)    #choosing the class with the highest logit
        total += labels.size(0)                 #sample size in batch
        correct += (predicted == labels).sum().item()   #calculating the correct predictions

    #loss and accuracy for training epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    y_true = []
    y_pred = []

    #validation mode
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())     #true labels
            y_pred.extend(predicted.cpu().numpy())  #predictions made
            val_total += labels.size(0)             #batch size
            val_correct += (predicted == labels).sum().item()   #calculating correct predictions based on labels


    #loss and accuracy for validation epoch
    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_accuracy = val_correct / val_total 
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_accuracy)

    print(f"Epoch {epoch+1}: "
          f"Train Loss = {epoch_loss:.4f}, Train Acc = {epoch_accuracy:.4f}, "
          f"Val Loss = {val_epoch_loss:.4f}, Val Acc = {val_epoch_accuracy:.4f}")


#printing plots for loss and accuracies + confusion matrix
model_acc_and_loss_plt(train_losses, train_accuracies, val_losses, val_accuracies)
conf_matrix(y_true, y_pred)


