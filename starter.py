from model.DNN_image import DNN_image
from model.DNN_landmark import DNN_landmark
from model.DNN_image_binary import DNN_image_binary
from utils.preprocessing import run_preprocessing
from utils.preprocessing import YOLO_preprocessing

def main():
    # DNN_image()
    # DNN_image()
    # DNN_landmark()

    #binary image with PCA
    # train_pca , test_pca, train_labels , test_labels = run_preprocessing()
    # DNN_image_binary(train_pca, test_pca, train_labels , test_labels)

    YOLO_preprocessing()


if __name__ == "__main__":
    main()
    