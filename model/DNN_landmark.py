from utils.read import read_data
import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils
import numpy as np
from  torch.utils.data import DataLoader,random_split, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture
class SimpleLandmarkClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleLandmarkClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def fit(model, criterion, optimizer, train_loader,valid_loader, num_epochs,patience):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    train_losses = []
    valid_losses = []
    accuracies = []         

    #Training
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        validation_loss, accuracy =comp_val_loss(model,valid_loader,criterion)
        valid_losses.append(validation_loss)
        accuracies.append(accuracy)
        
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
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


    # Learning curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(accuracies, label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Learning Curve')

    return model

def comp_val_loss(model,valid_loader,criterion):
# Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = []
        valid_loss = 0.0
        total = 0
        correct = 0
        for inputs, labels in valid_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        validation_loss = valid_loss / len(valid_loader)
        accuracy = correct / total

    return validation_loss , accuracy

def predict(model, test_loader, y_test):
    # Use the best model for evaluation on the test set
    model.eval()
    y_test_pred = []
    with torch.no_grad():
        total = 0
        correct = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_test_pred.extend(predicted.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate test accuracy and generate confusion matrix
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    label_encoder = LabelEncoder()
    # Alphabet -> Number for confusion Matrix
    label_encoder.classes_ = [chr(ord('A') + i) for i in range(26)]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate precision, recall, f1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')

    print(f'Weighted Precision: {precision:.4f}')
    print(f'Weighted Recall: {recall:.4f}')
    print(f'Weighted F1 Score: {f1:.4f}')



def DNN_landmark():
    train_file = '/Users/jeongyoon/Desktop/GitBlog/MSAI_349_FinalProject-1/dataset/train_landmarks.csv'
    test_file = '/Users/jeongyoon/Desktop/GitBlog/MSAI_349_FinalProject-1/dataset/test_landmarks.csv'
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    #Drop the header
    X = train.drop('label', axis=1).values
    y = train['label'].values

    X_test = test.drop('label', axis=1).values
    y_test = test['label'].values

    # Label encoding (Alphabet -> Number)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y_test = label_encoder.transform(y_test)

    # Divide into validation data/ training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform to Tensor
    X_train, X_test, y_train, y_test, X_valid, y_valid = map(torch.tensor, (X_train, X_test, y_train, y_test, X_valid, y_valid))

    # Data Loader
    train_dataset = TensorDataset(X_train.float(), y_train.long())
    valid_dataset = TensorDataset(X_valid.float(), y_valid.long())
    test_dataset = TensorDataset(X_test.float(), y_test.long())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X_train.size(1)
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = len(np.unique(y))

    # Initialize the model, loss criterion, and optimizer
    net = SimpleLandmarkClassifier(input_size, hidden_size1, hidden_size2, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    print("In training")
    # Train the model using the training and validation loaders
    net=fit(net, criterion, optimizer, train_loader, valid_loader, num_epochs=100,patience=10)
    print("In testing")
    # Evaluate the model on the test set
    predict(net, test_loader, y_test)


