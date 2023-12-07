from model.DNN_image import DNN_image
from model.DNN_landmark import DNN_landmark
from model.DNN_image_binary import DNN_image_binary
from model.CNN_image_YOLO import CNN_image_YOLO
from utils.preprocessing import run_preprocessing

def main():
    # DNN_image()
    # DNN_image()
    # DNN_landmark()

    #binary image with PCA
    train_pca , test_pca, train_labels , test_labels = run_preprocessing()
    DNN_image_binary(train_pca, test_pca, train_labels , test_labels)

    # YOLO_preprocessing()
    # CNN_image_YOLO()


if __name__ == "__main__":
    main()
    