import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd
import cv2
from sklearn.metrics import classification_report
from datetime import datetime  # Import datetime for timestamping


class EmotipigCNN(nn.Module):
    def __init__(self, num_classes):  # , initial_bias):
        super(EmotipigCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)  # Adjust the dimensions based on your input size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.flatten(x.permute(0, 2, 3, 1))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class SowStress:
    def __init__(self):
        self.img_rows, self.img_cols = 256, 256

        self.model_path = r'F:\videos\From Mark'
        filepath = os.path.join(self.model_path, 'pytorch_emotipig.pth')
        self.model = torch.load(filepath)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize((self.img_rows, self.img_cols)),  # Resize to the input size of the model
            transforms.ToTensor(),  # Convert the image to a tensor
        ])

    def process_img(self, image):
        image = Image.fromarray(image)
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch).cpu().detach().numpy().item()

        return output


def main():
    # Load Excel file
    excel_file = r'F:\videos\test_set.xlsx'  # Update this with your actual file path
    df = pd.read_excel(excel_file)

    # Check if the required columns are present
    if 'Path' not in df.columns or 'Condition' not in df.columns:
        print("Excel file must contain 'Path' and 'Condition' columns.")
        return

    # Create an instance of the SowStress class
    ss = SowStress()

    # Store predictions, true conditions, and model confidence
    predictions = []
    true_conditions = df['Condition'].tolist()
    model_confidence = []

    # Process each image
    for index, row in df.iterrows():
        image_path = row['Path']
        if os.path.exists(image_path):
            # Load the image
            image = cv2.imread(image_path)
            # Process the image and get the prediction
            prediction = ss.process_img(image)
            predictions.append(1 if prediction >= 0.9 else 0)  # Assuming threshold of 0.5 for binary classification
            model_confidence.append(prediction)  # Store the confidence score
        else:
            print(f"Image path {image_path} does not exist. Skipping...")
            predictions.append(None)  # Append None if image doesn't exist
            model_confidence.append(None)

    # Remove None predictions
    valid_indices = [i for i in range(len(predictions)) if predictions[i] is not None]
    true_conditions = [true_conditions[i] for i in valid_indices]
    predictions = [predictions[i] for i in valid_indices]
    model_confidence = [model_confidence[i] for i in valid_indices]

    # Generate classification report
    report = classification_report(true_conditions, predictions, target_names=['Stressed', 'Unstressed'])

    # Create a timestamp for the report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file_path = os.path.join(os.path.dirname(excel_file), f'classification_report_{timestamp}.txt')

    # Save the report to a text file
    with open(report_file_path, 'w') as f:
        f.write(report)

    print(f'Classification report saved to {report_file_path}')

    # Save results to a new Excel file with additional columns
    result_df = df.copy()  # Copy the original dataframe
    result_df['Predicted Label'] = predictions
    result_df['Model Confidence'] = model_confidence
    result_file_path = os.path.join(os.path.dirname(excel_file), f'results_{timestamp}.xlsx')
    result_df.to_excel(result_file_path, index=False)

    print(f'Results saved to {result_file_path}')


if __name__ == '__main__':
    main()
