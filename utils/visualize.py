import cv2


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

