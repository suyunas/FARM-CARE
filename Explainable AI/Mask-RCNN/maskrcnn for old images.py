import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if needed
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):  # Check if it's a file, not a directory
                try:
                    img = Image.open(file_path)
                    images.append(img)
                    filenames.append(filename)
                except IOError:
                    print(f"Could not read image: {file_path}")
    return images, filenames

# Function to transform images for Mask R-CNN
def transform_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)

# Function to apply Mask R-CNN to an image
def apply_mask_rcnn(model, image):
    with torch.no_grad():
        prediction = model(image)
    return prediction

def get_grid_cell_coordinates(image_shape, grid_size=(2, 3)):
    height, width = image_shape
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]
    return [(i * cell_height, j * cell_width, (i+1) * cell_height, (j+1) * cell_width) for i in range(grid_size[0]) for j in range(grid_size[1])]

def is_object_in_target_cell(box, target_cell):
    # Check if the center of the box is within the target cell
    center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return target_cell[0] <= center_y < target_cell[2] and target_cell[1] <= center_x < target_cell[3]

def visualize(image, prediction, output_path, filename, animal_classes, folder_name, index):
    image_np = np.array(image)
    image_shape = image_np.shape[:2]
    grid_cells = get_grid_cell_coordinates(image_shape)
    target_cell = grid_cells[4]  # (0,1,1)
    black_canvas = np.zeros_like(image_np)

    for element in range(len(prediction[0]['boxes'])):
        box = prediction[0]['boxes'][element].detach().numpy()
        mask = prediction[0]['masks'][element, 0].detach().numpy()
        score = prediction[0]['scores'][element].detach().numpy()
        label = prediction[0]['labels'][element].detach().numpy()

        if label in animal_classes and is_object_in_target_cell(box, target_cell):
            # Apply a threshold to the mask for sharper boundary
            mask = mask > 0.7

            # Smoothing edges of the mask with a smaller kernel
            mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (35, 35), 0)

            # Create a 3-channel mask for color image
            mask_3channel = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2)

            # Apply mask to the image
            black_canvas = np.where(mask_3channel > 0.5, image_np, black_canvas)

    # Rename the image with folder name and number
    new_filename = f"{folder_name}_{index:03d}.jpg"
    image_with_mask = Image.fromarray(black_canvas)
    image_with_mask.save(os.path.join(output_path, new_filename))
    
    # Delete the original file after saving the new one
    original_file_path = os.path.join(output_path, filename)
    if os.path.exists(original_file_path):
        os.remove(original_file_path)

# Example usage
animal_classes = [17, 18, 19, 20, 21, 22, 23, 24, 25]  # Example: cat, dog, horse, etc. indices from COCO

def process_folder(folder, model, animal_classes):
    images, filenames = load_images_from_folder(folder)
    folder_name = os.path.basename(folder)  # Get the folder name for renaming images
    
    for index, (image, filename) in enumerate(zip(images, filenames), start=1):
        transformed_image = transform_image(image)
        prediction = apply_mask_rcnn(model, transformed_image)
        visualize(image, prediction, folder, filename, animal_classes, folder_name, index)

def apply_mask_rcnn_to_all_folders(root_folder, model, animal_classes):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if dirnames:
            for subfolder in dirnames:
                subfolder_path = os.path.join(dirpath, subfolder)
                print(f"Processing folder: {subfolder_path}")
                process_folder(subfolder_path, model, animal_classes)
                print(f"Finished processing folder: {subfolder_path}")

if __name__ == '__main__':
    # Load the pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Ask the user to input the root folder
    root_folder = r'E:\FarmCare\B4_SOWS\4_MVD2\OF62\test'

    # Apply Mask R-CNN to all folders and subfolders provided by the user
    apply_mask_rcnn_to_all_folders(root_folder, model, animal_classes)