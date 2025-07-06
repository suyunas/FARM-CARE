import os
import pandas as pd

# Criteria lists for classification
#zero_criteria = ['YF12749', 'YF12876A', 'YF12892', 'YF12825B', 'YF12876B', 'YF12447', 'YF12612', 'YF12752', 'YF13730']
zero_criteria = ['YF12749', 'YF12876A', 'YF12892', 'YF12825B', 'YF12876B', 'YF12447', 'YF12612', 'YF12752', 'YF13770', 'YF12697']
one_criteria = ['YF12811', 'OF62', 'OF174', 'YF13921', 'YF13922', 'YF12746', 'YF12750', 'PF27', 'YF13730', 'YF13825', 'PF25']

# List of expected subfolder names within 'sow' folders
#expected_subfolders = ['D0', 'D1', 'D2', 'D3', 'D70', 'D90']
expected_subfolders = ['D70']

def get_condition(image_name):
    """Return 0 or 1 based on the image name and matching criteria."""
    for criteria in zero_criteria:
        if criteria in image_name:
            return 0
    for criteria in one_criteria:
        if criteria in image_name:
            return 1
    return None  # If no match, return None

def process_images(start_path):
    """Process images from 'sow' folders and their specific subfolders."""
    data = []

    for batch in os.listdir(start_path):
        batch_path = os.path.join(start_path, batch)

        # Check if the batch path is a directory
        if os.path.isdir(batch_path):
            # Check for the 'sow' folder
            sow_path = os.path.join(batch_path, 'sow')
            if os.path.isdir(sow_path):
                print(f"Processing 'sow' folder: {sow_path}")

                for subfolder in expected_subfolders:
                    subfolder_path = os.path.join(sow_path, subfolder)
                    
                    if os.path.isdir(subfolder_path):
                        print(f"Processing subfolder: {subfolder_path}")

                        # Walk through animal folders and their subdirectories
                        for animal_root, animal_dirs, animal_files in os.walk(subfolder_path):
                            print(f"Current animal directory: {animal_root}")

                            # Skip 'extra' folders
                            animal_dirs[:] = [d for d in animal_dirs if d.lower() != 'extra']
                            print(f"Filtered animal subdirectories: {animal_dirs}")

                            for file in sorted(animal_files):
                                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    image_path = os.path.join(animal_root, file)
                                    condition = get_condition(file)
                                    if condition is not None:
                                        data.append([image_path, condition])

    # Create a DataFrame with image paths and conditions
    df = pd.DataFrame(data, columns=['Path', 'Condition'])
    return df

def main():
    # Specify the starting path (root directory) where the batch folders are located
    start_path = r'F:\videos'  # Modify this to the correct path

    # Process images and get the DataFrame
    df = process_images(start_path)

    # Save the DataFrame to an Excel file in the same folder as the input path
    output_path = os.path.join(start_path, 'image_conditions.xlsx')
    df.to_excel(output_path, index=False)

    print(f"Excel file created: {output_path}")

if __name__ == "__main__":
    main()
