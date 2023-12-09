"""
Readme - Install yolo (miniconda) => Python version > 3.8

conda create --name yolov5-env python=3.8
conda activate yolov5-env

conda install -c pytorch pytorch torchvision torchaudio cpuonly
pip install matplotlib pillow

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -U -r requirements.txt

Change each path for your local environment

Extract the bounding box from YOLO, align the object in center (if it's exist), make background color as white.

"""
import os
import torch
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s')  # You can choose different versions

def runYOLO(type):
    # Setting your path for dataset folder
    output_folder = f'/Users/jeongyoon/Desktop/GitBlog/MSAI_349_FinalProject-1/dataset/{type}_YOLO_new_new'

    # Loop through each image in the dataset
    dataset_folder = f'/Users/jeongyoon/Desktop/GitBlog/MSAI_349_FinalProject-1/dataset/{type}'
    for label_folder in os.listdir(dataset_folder):
        if label_folder == '.DS_Store':  # Skip .DS_Store folder
            continue

        label_path = os.path.join(dataset_folder, label_folder)
        print("type , label : " , type, label_folder)

        # Loop through each image in the label folder
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = Image.open(img_path)

            # Inference
            results = model(img)

            # Get folder name as label
            label = label_folder
            

            # Check if any object is detected
            if len(results.xyxy[0]) > 0:
                # Process each object
                for i, (bbox, _) in enumerate(zip(results.xyxy[0].cpu().numpy(), results.names)):
                    # Extract object image
                    object_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                    # Calculate padding to center the object
                    pad_width = (img.width - object_img.width) // 2
                    pad_height = (img.height - object_img.height) // 2

                    # Create a new image with white background
                    new_img = Image.new("RGB", (img.width, img.height), "white")

                    # Paste the object image onto the new image at the center
                    new_img.paste(object_img, (pad_width, pad_height))

                    # Save the processed image
                    output_label_folder = os.path.join(output_folder, 'YOLO_new_new', f'{type}', label)
                    os.makedirs(output_label_folder, exist_ok=True)
                    output_img_path = os.path.join(output_label_folder, f'object_{i}_{img_name}')
                    
                    new_img.save(output_img_path)
            else:
                # If no object is detected, save the original image
                output_label_folder = os.path.join(output_folder, 'YOLO_new_new', f'{type}', label)
                os.makedirs(output_label_folder, exist_ok=True)
                output_img_path = os.path.join(output_label_folder, f'original_{img_name}')
                
                img.save(output_img_path)

if __name__ == "__main__":
    runYOLO("train")
    runYOLO("test")

