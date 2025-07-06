import pandas as pd
import os

def count_labels_in_excel(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_excel(file_path)
                columns = [col.strip() for col in df.columns]  # Remove whitespace
                if 'Condition' in columns:
                    df.columns = columns  # Normalize column names
                    counts = df['Condition'].value_counts()
                    count_0 = counts.get(0, 0)
                    count_1 = counts.get(1, 0)
                    print(f"File: {filename}")
                    print(f"  Label 0: {count_0}")
                    print(f"  Label 1: {count_1}\n")
                else:
                    print(f"File: {filename} - Skipped (no 'Condition' column)\n")
            except Exception as e:
                print(f"Error processing file {filename}: {e}\n")

# Example usage:
# Replace with the actual folder containing your Excel files
count_labels_in_excel(r'J:\test')
