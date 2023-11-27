from utils.read import read_data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.utils
from torch.utils.data import Dataset
import numpy as np
from  torch.utils.data import DataLoader,random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, dir_path, resize=(30, 30)):
        self.images, self.labels = self.load_data(dir_path, resize)
        self.labels = self.convert_alpha_to_num(self.labels)

    def load_data(self, dir_path, resize):
        images, labels = read_data(dir_path, flatten=0, grayscale=0, resize=resize)
        images_np = np.stack(images, axis=0)
        images_tensor = torch.tensor(images_np).float()
        images_tensor = images_tensor.permute(0, 3, 1, 2)
        return images_tensor, labels

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define convolutional layers, activation functions, pooling layers, and fully connected layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,kernel_size=3, padding=1,stride=1)#out:6x30x30
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)#out:6x15x15
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1,stride=1)#out:16x15x15,after pool:16x7x7
        self.fc1 = nn.Linear(784, 120)  # Adjust input size to match your data
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)  # Adjust the output size to match the number of classes

    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1,784)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Training
def fit(model, criterion, optimizer, train_loader,valid_loader, num_epochs=10,patience=5):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print(f'Accuracy: {accuracy*100:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

def main():
    # Load the full training dataset
    full_train_dataset = CustomDataset("../dataset/train")
    test_dataset = CustomDataset("../dataset/test")

    # Calculate the sizes for training and validation sets (80-20 split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create data loaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    # Initialize the model, loss criterion, and optimizer
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the model using the training and validation loaders
    net=fit(net, criterion, optimizer, train_loader, val_loader, num_epochs=50,patience=5)

    # Evaluate the model on the test set
    predict(net, test_loader)


if __name__ == '__main__':
    main()

