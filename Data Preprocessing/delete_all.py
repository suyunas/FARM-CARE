import os

def delete_files_in_folder(folder_path):
    # Loop through the directory tree using os.walk
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

if __name__ == "__main__":
    # Specify the folder path you want to clean up
    folder_path = r'I:\check'

    if os.path.isdir(folder_path):
        delete_files_in_folder(folder_path)
        print("All files have been deleted. Folders remain intact.")
    else:
        print("Invalid folder path.")