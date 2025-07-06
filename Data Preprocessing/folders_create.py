import os

def get_folders(path):
    """Returns a list of folder names in the given directory path."""
    try:
        # List all items in the directory
        items = os.listdir(path)
        # Filter out only folders
        folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
        return folders
    except FileNotFoundError:
        print("The specified path does not exist.")
        return []
    except PermissionError:
        print("Permission denied to access the specified path.")
        return []

def create_folders(source_folders, destination_path):
    """Creates folders in the destination path with names from source_folders."""
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    
    for folder in source_folders:
        folder_path = os.path.join(destination_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

def main():
    source_path = r'I:\FarmCare\B3\f2sows\D0'.strip()
    destination_path = r'I:\FarmCare\B3\f2sows\B3_GILTS_DT90_230124'.strip()

    # Get folder names from the source directory
    source_folders = get_folders(source_path)

    if source_folders:
        # Create similar empty folders in the destination directory
        create_folders(source_folders, destination_path)
    else:
        print("No folders found in the source directory.")

if __name__ == "__main__":
    main()
