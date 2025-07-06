import os
import re

def rename_folders(path):
    """Renames folders in the specified directory based on the pattern starting with 'RF'."""
    try:
        # List all items in the directory
        items = os.listdir(path)
        # Filter out only folders (excluding subfolders)
        folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
        
        # Regular expression to match 'RF' followed by digits
        pattern = re.compile(r'(RF\d+)')
        
        for folder in folders:
            # Search for 'RF' pattern in the folder name
            match = pattern.search(folder)
            if match:
                new_name = match.group(1)  # Extract matched part (e.g., 'RF98' or 'RF982')
                old_folder_path = os.path.join(path, folder)
                new_folder_path = os.path.join(path, new_name)
                
                # Rename the folder
                if not os.path.exists(new_folder_path):
                    os.rename(old_folder_path, new_folder_path)
                    print(f"Renamed '{folder}' to '{new_name}'")
                else:
                    print(f"Folder '{new_name}' already exists. Skipping rename for '{folder}'.")
            else:
                print(f"No 'RF' pattern found in '{folder}'. Skipping.")
    
    except FileNotFoundError:
        print("The specified path does not exist.")
    except PermissionError:
        print("Permission denied to access the specified path.")

def main():
    folder_path = r'I:\FarmCare\B5\f2sows\BATCH 5_GILTS_MOVE_D2'.strip()
    rename_folders(folder_path)

if __name__ == "__main__":
    main()
