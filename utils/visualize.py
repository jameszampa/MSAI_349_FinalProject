# Visualize image from pixels
import cv2


def show_image(image):
    img = image.reshape(28, 28).astype('float32')
    cv2.imwrite('../image.png', img)  # save image


if __name__ == "__main__":
    from utils.read import read_df
    train_df = read_df('../dataset/sign_mnist_train.csv')
    show_image(train_df.values[0][1:])
