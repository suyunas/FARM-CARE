import cv2
import os
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
from scipy import ndimage

# Load pre-trained Mask R-CNN
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to extract frames from video
def extract_frames(video_path, output_folder, frame_rate=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    interval = int(fps / frame_rate)  # Calculate interval between frames
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    extracted_frame_paths = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    success, frame = cap.read()
    while success:
        if frame_count % interval == 0:
            frame_filename = f"{video_name}_frame{frame_count}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frame_paths.append(frame_path)
        
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    return extracted_frame_paths

# Function to detect and filter motion blur
def is_blurry(image_path, threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()  # Compute Laplacian variance
    return laplacian_var < threshold  # True if image is blurry

# Function to transform images for Mask R-CNN
def transform_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)

# Function to apply Mask R-CNN to an image
def apply_mask_rcnn(model, image):
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Function to check if object is in the target grid cell
def get_grid_cell_coordinates(image_shape, grid_size=(2, 3)):
    height, width = image_shape
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]
    return [(i * cell_height, j * cell_width, (i+1) * cell_height, (j+1) * cell_width) for i in range(grid_size[0]) for j in range(grid_size[1])]

def is_object_in_target_cell(box, target_cell):
    center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return target_cell[0] <= center_y < target_cell[2] and target_cell[1] <= center_x < target_cell[3]

# Function to visualize Mask R-CNN output
def visualize(image, prediction, output_path, filename, animal_classes, video_name, frame_number):
    image_np = np.array(image)
    image_shape = image_np.shape[:2]
    grid_cells = get_grid_cell_coordinates(image_shape)
    target_cell = grid_cells[4]  # Target grid cell (0, 1, 1)
    black_canvas = np.zeros_like(image_np)

    # Debug: Print the number of detections
    num_detections = len(prediction[0]['boxes'])
    print(f"Number of detections: {num_detections}")

    for element in range(num_detections):
        box = prediction[0]['boxes'][element].detach().numpy()
        mask = prediction[0]['masks'][element, 0].detach().numpy()
        score = prediction[0]['scores'][element].detach().numpy()
        label = prediction[0]['labels'][element].detach().numpy()

        # Debug: Print the detection details
        print(f"Box: {box}, Score: {score}, Label: {label}")

        if score >= 0.5 and label in animal_classes and is_object_in_target_cell(box, target_cell):
            # Apply threshold to mask
            mask = mask > 0.7

            # Smoothing mask edges
            mask_blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=7)

            # Create 3-channel mask for color image
            mask_3channel = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2)

            # Apply mask to black canvas
            black_canvas = np.where(mask_3channel > 0.5, image_np, black_canvas)

    if np.any(black_canvas):
        # Save new image with mask and rename
        new_filename = f"{video_name}_frame{frame_number}.jpg"
        image_with_mask = Image.fromarray(black_canvas)
        image_with_mask.save(os.path.join(output_path, new_filename))
    else:
        print(f"No significant masks found for {filename}")

# Main processing function
def process_video(video_path, output_folder, animal_classes, frame_rate=3, blur_threshold=100):
    # Step 1: Extract frames from video
    frame_paths = extract_frames(video_path, output_folder, frame_rate)
    
    # Step 2: Filter out motion-blurred frames
    filtered_frames = [frame for frame in frame_paths if not is_blurry(frame, blur_threshold)]
    
    # Step 3: Apply Mask R-CNN to remaining frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    for frame_path in filtered_frames:
        frame_number = int(os.path.splitext(os.path.basename(frame_path))[0].split("frame")[1])
        
        image = Image.open(frame_path)
        transformed_image = transform_image(image)
        prediction = apply_mask_rcnn(model, transformed_image)
        
        visualize(image, prediction, output_folder, frame_path, animal_classes, video_name, frame_number)
    
    print(f"Finished processing video: {video_name}")

# Example usage
if __name__ == '__main__':
    video_path = r'E:\FarmCare\B4_SOWS\4_MVD2\OF62\test\OF62_move D2 (3).MP4'  # Input video file
    output_folder = r'E:\FarmCare\B4_SOWS\4_MVD2\OF62\test'  # Output folder for extracted images

    animal_classes = [17, 18, 19, 20, 21, 22, 23, 24, 25]  # Example COCO animal class IDs (cat, dog, horse, etc.)
    
    process_video(video_path, output_folder, animal_classes, frame_rate=3, blur_threshold=100)
