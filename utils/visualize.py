# Visualize image from pixels
import cv2


def show_image(image):
    img = image.reshape(28, 28).astype('float32')
    cv2.imwrite('../image.png', img)  # save image


def visualize_hand_landmarks(image_file_path, landmarks):
    img = cv2.imread(image_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width = img.shape[1]
    height = img.shape[0]
    for landmark in landmarks:
        norm_x, norm_y, _ = landmark
        x = int(norm_x * width)
        y = int(norm_y * height)
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
    return img


if __name__ == "__main__":
    from utils.read import read_df
    train_df = read_df('../dataset/sign_mnist_train.csv')
    show_image(train_df.values[0][1:])
