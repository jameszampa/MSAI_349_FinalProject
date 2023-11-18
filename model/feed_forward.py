import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from utils.read import read_df
from utils.evaluate import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
token_label_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, scaler):
        """
        :param dataset: list: [[label], [feature1, feature2, ..., featureN]]
        :param scaler:
        """
        self.dataset = dataset
        self.sc = scaler

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw = self.dataset.iloc[idx, 1:]
        if type(idx) == int:
            raw = np.array(raw).reshape(1, -1)
        raw = self.sc.transform(raw)
        data = torch.tensor(raw, dtype=torch.float32)
        label_token = token_label_map.index(self.dataset.iloc[idx, 0])
        label = torch.tensor(label_token, dtype=torch.float32)
        return data, label


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(784, 2)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 24)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x


def train(dataloader, model, loss_func, optimizer):
    model.train()
    train_loss = []
    now = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        y = y.type(torch.LongTensor)
        loss = loss_func(pred.squeeze(1), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Show progress and record training loss
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            iters = 10 * len(X)
            then = datetime.datetime.now()
            iters /= (then - now).total_seconds()
            # print(f"loss: {loss:>6f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
            now = then
            train_loss.append(loss)
    return train_loss


def valid(dataloader, model, loss_func):
    num_batches = 0
    model.eval()
    valid_loss = 0
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.type(torch.LongTensor)
            # Record loss
            valid_loss += loss_func(pred.squeeze(1), y).item()
            # Record prediction
            _, predicted_class_index = torch.max(pred[0], 1)  # Get the predicted class index
            predicted_class = token_label_map[predicted_class_index]
            predictions.append(predicted_class)
            ground_truth.append(int(y.item()))
            num_batches = num_batches + 1
    # Calculate loss
    valid_loss /= num_batches
    print(f"\nValid Avg Loss: {valid_loss:>8f}")
    # Calculate F1
    f1 = f1_score(ground_truth, predictions, average='micro')
    return valid_loss, f1


def test(dataloader, model, loss_func):
    num_batches = 0
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.type(torch.LongTensor)
            test_loss += loss_func(pred.squeeze(1), y).item()
            num_batches = num_batches + 1
    test_loss /= num_batches
    print(f"Test Avg Loss: {test_loss:>8f}\n")
    return test_loss


def generate_plots(train_loss, valid_loss, test_loss, lr_rate, epochs):
    # Plot the training loss
    plt.plot([i for i in range(len(train_loss))], train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('train_loss_' + str(lr_rate) + '_' + str(epochs) + '.png')
    plt.clf()
    # Plot the valid loss
    plt.plot([i for i in range(len(valid_loss))], valid_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.savefig('valid_loss_' + str(lr_rate) + '_' + str(epochs) + '.png')
    plt.clf()
    # Plot the testing loss
    plt.plot([i for i in range(len(test_loss))], test_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.savefig('test_loss_' + str(lr_rate) + '_' + str(epochs) + '.png')
    plt.clf()
    # Plot train and validation on the same figure
    plt.plot([i for i in range(len(train_loss))], train_loss, label='Training')
    plt.plot([i for i in range(len(valid_loss))], valid_loss, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig('train_valid_loss_' + str(lr_rate) + '_' + str(epochs) + '.png')
    plt.clf()
    # Plot train, valid, and test on the same figure
    plt.plot([i for i in range(len(train_loss))], train_loss, label='Training')
    plt.plot([i for i in range(len(valid_loss))], valid_loss, label='Validation')
    plt.plot([i for i in range(len(test_loss))], test_loss, label='Test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation and Test Loss')
    plt.savefig('train_valid_test_loss_' + str(lr_rate) + '_' + str(epochs) + '.png')
    plt.clf()


def feed_forward(lr_rate, epochs):
    if_lr_decay = False
    stopping_criteria = False

    train_data = read_df('../sign_mnist_train_tiny.csv')
    valid_data = read_df('../sign_mnist_valid_tiny.csv')
    test_data = read_df('../sign_mnist_test_tiny.csv')

    # Define scaler, fit it to the training data and use the same scaler to transform the test and validation data
    sc = MinMaxScaler()
    sc.fit_transform([data[1:] for data in train_data.values])

    # Load dataset
    train_dataset = Dataset(train_data, sc)
    valid_dataset = Dataset(valid_data, sc)
    test_dataset = Dataset(test_data, sc)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Define model loss, optimizer, etc.
    ff = FeedForward().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ff.parameters(), lr=lr_rate)

    if if_lr_decay:
        # Define learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 0.965
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=30)  # 0.97 lr_rate=0.2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)  # 0.965

    # Start training
    train_loss, valid_loss, test_loss = [], [], []
    finished_epoch = 0
    if stopping_criteria:
        best_val_loss = float('inf')
        patience_counter = 0
        desired_f1 = 0.9
    for t in range(epochs):
        print(f"Epoch {t + 1}\n------------------------------- \n")
        losses = train(train_loader, ff, loss_func, optimizer)
        # Record learning rate decay
        if if_lr_decay:
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print("Epoch %d: SGD lr %.4f -> %.4f" % (t, before_lr, after_lr))
        # Record loss
        train_loss.append(sum(losses)/len(losses))
        valid_loss_epoch, valid_f1 = valid(valid_loader, ff, loss_func)
        valid_loss.append(valid_loss_epoch)
        test_loss.append(test(test_loader, ff, loss_func))
        finished_epoch += 1
        # Stopping criteria
        if stopping_criteria:
            # Achieved desired performance
            if valid_f1 >= desired_f1:
                print("Desired performance achieved.")
                break
            # Early stopping
            patience = 5  # Number of epochs to wait for improvement
            if valid_loss_epoch < best_val_loss:
                best_val_loss = valid_loss_epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping: No improvement for {} epochs.".format(patience))
                    break
    print("\n-------------------------------")
    print("Valid Avg Losses: ", valid_loss)
    print("Test Avg Losses: ", test_loss)
    print("Done!")

    # Draw the plot
    # generate_plots(train_loss, valid_loss, test_loss, lr_rate, epochs)

    # Test
    ff.eval()
    predictions = []
    ground_truth = []
    for x, y in test_dataset:
        with torch.no_grad():
            x = x.to(device)
            pred = ff(x)
            _, predicted_class_index = torch.max(pred[0], 0)  # Get the predicted class index
            predicted_class = token_label_map[predicted_class_index]
            predictions.append(predicted_class)
            ground_truth.append(int(y.item()))

    # Evaluate
    evaluate('feed_forward', ground_truth, predictions)


if __name__ == "__main__":
    feed_forward(0.1, 100)