import os
import shutil

def organize_images_by_prefix(folder_path):
    # Define the paths for the new 'pre' and 'post' folders
    pre_folder = os.path.join(folder_path, 'pre')
    post_folder = os.path.join(folder_path, 'post')

    # Create the 'pre' and 'post' folders if they don't already exist
    os.makedirs(pre_folder, exist_ok=True)
    os.makedirs(post_folder, exist_ok=True)

    # Iterate over all files in the given folder
    for filename in os.listdir(folder_path):
        # Get the full path of the file
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image and move it to the appropriate folder
        if os.path.isfile(file_path):
            if 'pre' in filename.lower():
                shutil.move(file_path, os.path.join(pre_folder, filename))
            elif 'post' in filename.lower():
                shutil.move(file_path, os.path.join(post_folder, filename))

# Example usage
folder_path = r"E:\FarmCare\videos\B3\sow\D70\YF12825\seg"  # Replace this with the path to your folder
organize_images_by_prefix(folder_path)