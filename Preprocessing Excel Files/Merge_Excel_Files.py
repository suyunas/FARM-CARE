import pandas as pd
import os

# Prompt for folder path
folder_path = input("Enter the folder path containing Excel files to merge: ").strip()

# Validate the folder path
if not os.path.isdir(folder_path):
    print("Invalid folder path.")
    exit()

# Get all .xlsx files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

if not file_list:
    print("No Excel files found in the specified folder.")
    exit()

print(f"Found {len(file_list)} Excel files: {file_list}")

# List to store DataFrames
dfs = []

# Read and store each Excel file
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    try:
        df = pd.read_excel(file_path)
        dfs.append(df)
        print(f"Loaded: {filename}")
    except Exception as e:
        print(f"Could not read {filename}: {e}")

# Merge all DataFrames
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged file
output_path = os.path.join(folder_path, "merged_output.xlsx")
merged_df.to_excel(output_path, index=False)

print(f"Merged Excel file saved at: {output_path}")
