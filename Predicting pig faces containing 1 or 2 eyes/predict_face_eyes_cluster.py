# cluster the similar images

import os
import cv2
import csv
import random
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from torchvision.models import vgg16
from PIL import Image

###

# Load pre-trained VGG16 model
model_vgg16 = vgg16(pretrained=True)
model_vgg16.classifier = model_vgg16.classifier[:-3]  # Remove the last three layers (including the final softmax)

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to VGG16 input size
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize according to ImageNet standards
])

def extract_features(cropped_img):
    try:
        # Convert cropped_img to PIL Image if it's not already
        if not isinstance(cropped_img, Image.Image):
            cropped_img = Image.fromarray(cropped_img)
        
        # Apply transformation
        img_tensor = transform(cropped_img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Set model to evaluation mode
        model_vgg16.eval()
        
        # Extract features using VGG16 model
        with torch.no_grad():
            features = model_vgg16(img_tensor)
        
        return features.squeeze().numpy()  # Squeeze to remove batch dimension and convert to numpy array
    
    except Exception as e:
        print(f"Error in extract_features: {e}")
        return None


def get_representative_images(detections, num_clusters):
    # Extract features for each cropped_img
    features = [extract_features(detection[1]) for detection in detections if detection[1] is not None]
    features = [f for f in features if f is not None]  # Remove None entries
    
    if len(features) == 0:
        print("No valid features extracted. Exiting.")
        return []
    
    # Convert list of arrays to 2D array
    features = np.stack(features, axis=0)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    
    # Get the indices of the representative images (one from each cluster)
    representative_indices = []
    for cluster in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster)[0]
        representative_indices.append(random.choice(cluster_indices))
    
    # Select representative images based on indices
    detections_to_save = [detections[i] for i in representative_indices]
    return detections_to_save

####

def ensure_output_folder(root):
    output_folder = os.path.join(root, 'seg')
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def process_images_in_folder(image_paths, csv_writer, subfolder_name, output_folder, area_threshold_min, area_threshold_max, net, ln):
    detections = []
    frame_count = 0

    for image_path in image_paths:
        print(f"Processing image: {image_path}")  # Log image path
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            continue

        H, W, _ = img.shape
        frame_count += 1

        try:
            results = model(img, conf=0.9, verbose=False)

            if not results:
                #print("No results from model inference")
                continue

            for result in results:
                if not result or not result.masks:
                    #print("No masks in result")
                    continue

                for mask in result.masks.data:
                    mask = mask.cpu().numpy()
                    mask = cv2.resize(mask, (W, H))
                    binary_mask = (mask > 100 / 255).astype(np.uint8) * 255

                    dilation_kernel_size = (9, 9)
                    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
                    binary_mask = cv2.dilate(binary_mask, dilation_kernel, iterations=1)

                    masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        area = w * h

                        if area_threshold_min <= area <= area_threshold_max:
                            cropped_img = masked_img[y:y + h, x:x + w]
                            cropped_img_resized = cv2.resize(cropped_img, (256, 256))

                            # Perform eye detection
                            image_inf, boxes = perform_inference(cropped_img_resized, net, ln, inference_res=256, swapRB=True)
                            boxes = sanitize_boxes(boxes)
                            if len(boxes) > 0:
                                frame_name = f"{subfolder_name}_frame{frame_count}"
                                detections.append((img.copy(), cropped_img_resized, binary_mask.copy(), frame_name, area))
                                # Write to CSV
                                csv_writer.writerow([frame_name, area])
                                print(f"Detection added: {frame_name}, area: {area}")
                            else:
                                print(f"No eyes detected in {frame_name}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    return detections


def process_video(video_path, csv_writer, subfolder_name, output_folder, area_threshold_min, area_threshold_max, net, ln):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    frame_count = 0
    detections = []

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        frame_count += 1
        H, W, _ = img.shape
        try:
            results = model(img, conf=0.7, verbose=False)

            if not results:
                #print("No results from model inference")
                continue

            for result in results:
                if not result or not result.masks:
                    #print("No masks in result")
                    continue

                for mask in result.masks.data:
                    mask = mask.cpu().numpy()
                    mask = cv2.resize(mask, (W, H))
                    binary_mask = (mask > 100 / 255).astype(np.uint8) * 255

                    dilation_kernel_size = (9, 9)
                    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
                    binary_mask = cv2.dilate(binary_mask, dilation_kernel, iterations=1)

                    masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        area = w * h

                        if area_threshold_min <= area <= area_threshold_max:
                            cropped_img = masked_img[y:y + h, x:x + w]
                            cropped_img_resized = cv2.resize(cropped_img, (256, 256))

                            # Perform eye detection
                            image_inf, boxes = perform_inference(cropped_img_resized, net, ln, inference_res=256, swapRB=True)
                            boxes = sanitize_boxes(boxes)
                            if len(boxes) > 0:
                                # Construct frame_name
                                frame_name = f"{subfolder_name}_{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count}"
                                detections.append((img.copy(), cropped_img_resized, binary_mask.copy(), frame_name, area))
                                # Write to CSV
                                csv_writer.writerow([frame_name, area])
                                print(f"Detection added: {frame_name}, area: {area}")
                            else:
                                print(f"No eyes detected in frame {frame_count} of video {video_path}")

        except Exception as e:
            print(f"Error processing frame {frame_count} of video {video_path}: {e}")

    cap.release()
    cv2.destroyAllWindows()

    return detections




def perform_inference(image, net, ln, inference_res=256, swapRB=True):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (inference_res, inference_res), swapRB=swapRB, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    confidence_T = 0.01
    threshold_T = 0.01
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence >= confidence_T:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_T, threshold_T)
    return image, boxes

def sanitize_boxes(boxes):
    for i, b in enumerate(boxes):
        boxes[i] = [0 if x <= 0 else x for x in b]
    return boxes

def filter_images_with_eyes(detections, net, ln):
    valid_detections = []
    for img, cropped_img, binary_mask, frame_name, area in detections:
        image_inf, boxes = perform_inference(cropped_img, net, ln, inference_res=256, swapRB=True)
        boxes = sanitize_boxes(boxes)
        if len(boxes) > 0:
            valid_detections.append((img, cropped_img, binary_mask, frame_name, area))
            print(f"Detection with eyes: {frame_name}")

    return valid_detections

# Load the YOLOv8 model for face detection
model_path = r'C:\Users\syedu\Desktop\Coding\acode\best10_t23_8x.pt'
model = YOLO(model_path)

# Load the YOLOv3 model for eye detection
configPath = r'C:\Users\syedu\Desktop\Coding\acode\eye_model\pigeyes-yolov3-tiny.cfg'
weightsPath = r'C:\Users\syedu\Desktop\Coding\acode\eye_model\pigeyes-yolov3-tiny_final.weights'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln1 = net.getLayerNames()
ln = [ln1[i - 1] for i in net.getUnconnectedOutLayers()]

# Input folder containing videos and images
input_folder = r'J:\F2-Daughters\B3\D2'

# Define area thresholds
area_threshold_gilt_min = 20000
area_threshold_gilt_max = 2600000
area_threshold_other_min = 20000
area_threshold_other_max = 2600000

# Walk through all directories and files in the input folder
for root, dirs, files in os.walk(input_folder):
     
    if 'extra' in dirs:
        dirs.remove('extra')  # Skip the 'extra' folder if it exists in the current directory

    if 'seg' in dirs:
        dirs.remove('seg')  # Skip the 'seg' folder

    subfolder_name = os.path.relpath(root, input_folder).replace(os.sep, '_')
    output_folder = ensure_output_folder(root)  # Ensure output folder in each subfolder

    # CSV file to save frame names and areas
    csv_file = os.path.join(output_folder, 'detections.csv')

    # Create and write CSV header
    with open(csv_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Frame Name', 'Area'])

        detections = []
        image_paths = []

        for file in files:
            file_path = os.path.join(root, file)

            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                print(f"Processing video: {file_path}")
                video_detections = process_video(file_path, csv_writer, subfolder_name, output_folder, area_threshold_gilt_min, area_threshold_gilt_max, net, ln)
                detections.extend(video_detections)
            #elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            #    image_paths.append(file_path)

            if image_paths:
                image_detections = process_images_in_folder(image_paths, csv_writer, subfolder_name, output_folder, area_threshold_gilt_min, area_threshold_gilt_max, net, ln)
                detections.extend(image_detections)

            # Randomly select 30% of detections to save
            if detections:  # Ensure detections list is not empty                
                # Define the number of clusters (or diverse images) you want
                num_clusters = min(len(detections), 20)

                # Get representative images
                detections_to_save = get_representative_images(detections, num_clusters)

                print(f"Detections to save: {len(detections_to_save)}")  # Debug print for detections to save

                for img, cropped_img, binary_mask, frame_name, area in detections_to_save:

                    output_image_path = os.path.join(output_folder, f"{frame_name}.jpg")
                    output_cropped_path = os.path.join(output_folder, f"{frame_name}_cropped.jpg")
                    output_mask_path = os.path.join(output_folder, f"{frame_name}.png")

                    if not cv2.imwrite(output_cropped_path, cropped_img):
                        print(f"Failed to save cropped image: {output_cropped_path}")

                print(f"Saved {len(detections_to_save)} out of {len(detections)} detections for {subfolder_name}")
                del detections[:]
            else:
                print(f"No valid detections with eyes for {subfolder_name}")