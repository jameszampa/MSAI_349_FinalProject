import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from utils.read import read_data, read_features_labels
from utils.preprocessing import DimensionReduction
from sklearn.preprocessing import StandardScaler
from utils.evaluate import evaluate


def svm_with_image():
    # Load a dataset
    print("Load a dataset")
    X_train, Y_train = read_data('../dataset/train', flatten=1, grayscale=1, resize=(50, 50))
    X_test, Y_test = read_data('../dataset/test', flatten=1, grayscale=1, resize=(50, 50))

    # Preprocessing
    print("Preprocessing")
    # Grayscale, Histogram equalization, Resize in the data reading process
    # Scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # PCA
    pca = DimensionReduction(X_train, 588)
    X_train = pca.pca_transform(X_train)

    # Create an SVM model
    print("Create an SVM model")
    svm_model = LinearSVC(dual="auto", random_state=0, tol=0.01, fit_intercept=True, intercept_scaling=50, max_iter=1000)

    # Train the model
    print("Train the model")
    svm_model.fit(X_train, Y_train)

    # Make predictions on the test set
    print("Make predictions on the test set")
    X_test_t = scaler.transform(X_test)
    X_test_t = pca.pca_transform(X_test_t)
    y_pred = svm_model.predict(X_test_t)

    # Evaluate the accuracy
    print("Evaluate the accuracy")
    accuracy = accuracy_score(Y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    evaluate('svm_image', Y_test, y_pred)

    # Return bad cases for analysis
    indices = np.where(Y_test != y_pred)
    return X_test[indices], Y_test[indices], y_pred[indices]


def svm_with_landmarks():
    # Load a dataset
    print("Load a dataset")
    X_train, Y_train = read_features_labels('../dataset/train_landmarks.csv')
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test, Y_test = read_features_labels('../dataset/test_landmarks.csv')
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print ("Preprocessing")
    # Preprocessing
    # Scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # Create an SVM model
    print("Create an SVM model")
    svm_model = LinearSVC(dual="auto", random_state=0, tol=0.001, fit_intercept=True, intercept_scaling=1, max_iter=1000)

    # Train the model
    print("Train the model")
    svm_model.fit(X_train, Y_train)

    # Make predictions on the test set
    print("Make predictions on the test set")
    X_test_t = scaler.transform(X_test)
    y_pred = svm_model.predict(X_test_t)

    print ("Evaluate the accuracy")
    # Evaluate the accuracy
    accuracy = accuracy_score(Y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    evaluate('svm', Y_test, y_pred)

    # Return bad cases for analysis
    indices = np.where(Y_test != y_pred)
    return X_test[indices], Y_test[indices], y_pred[indices]


if __name__ == '__main__':
    svm_with_image()

