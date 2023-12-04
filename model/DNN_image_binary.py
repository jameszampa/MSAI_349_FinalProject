from utils.read import read_data
import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils
from torch.utils.data import Dataset
import numpy as np
from  torch.utils.data import DataLoader,random_split
import torch.nn.functional as F
from utils.evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class CustomDatasetForPCA(Dataset):
    def __init__(self, images, labels):
        self.images = images_tensor = torch.tensor(images).float()
        self.labels = self.convert_alpha_to_num(labels)

    def convert_alpha_to_num(self, labels):
        flattened_labels = [label[0] for label in labels]
        label_to_int = {label: idx for idx, label in enumerate(sorted(set(flattened_labels)))}
        numeric_labels = [label_to_int[label] for label in flattened_labels]
        return numeric_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label    

# Model architecture
class SimpleImageClassifier(nn.Module):
    def __init__(self):
        super(SimpleImageClassifier, self).__init__()
        self.fc1=nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 120)  # 200*200 pixels, RGB three channel
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 26) #26 classes

    def forward(self, x):
        # x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def fit(model, criterion, optimizer, train_loader,valid_loader, num_epochs=10,patience=5):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print("epochs : ", epoch)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_loss=comp_val_loss(model,valid_loader,criterion)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)},Validation Loss:{val_loss}')
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve == patience:
            print("Early stopping triggered")
            break

        # Revert to the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def comp_val_loss(model,valid_loader,criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(valid_loader)
    return avg_loss

def predict(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    true_labels=[]
    pred_labels=[]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    precision,recall,f1=evaluate('DNN_binary_images',true_labels,pred_labels)
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')



def DNN_image_binary(pca_train , pca_test, train_label, test_label):
    # Load the full training dataset
    # Change the path based on your local folder
  
    full_train_dataset = CustomDatasetForPCA(pca_train, train_label)
    test_dataset = CustomDatasetForPCA(pca_test, test_label)
    # Calculate the sizes for training and validation sets (80-20 split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create data loaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)



    # Initialize the model, loss criterion, and optimizer
    net = SimpleImageClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    print("In training")
    # Train the model using the training and validation loaders
    net=fit(net, criterion, optimizer, train_loader, val_loader, num_epochs=100,patience=15)
    print("In testing")
    # Evaluate the model on the test set
    predict(net, test_loader)


