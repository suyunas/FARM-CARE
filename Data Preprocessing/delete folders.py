import os
import shutil

def delete_seg_folders(root_dir):
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Iterate through the folder names
        for dirname in dirnames:
            if dirname == 'seg':  # If the folder is named 'seg'
                folder_to_delete = os.path.join(dirpath, dirname)
                print(f"Deleting folder: {folder_to_delete}")
                shutil.rmtree(folder_to_delete)  # Delete the folder and its contents

if __name__ == "__main__":
    root_directory = r'J:\F2-Daughters\extra\B1\extra\D0'  # The current directory, modify this if necessary
    delete_seg_folders(root_directory)
