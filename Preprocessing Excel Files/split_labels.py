import pandas as pd
import os
from datetime import datetime

def split_excel_file(excel_file):
    # Load the Excel file
    df = pd.read_excel(excel_file)

    # Check if the required 'Condition' column is present
    if 'Condition' not in df.columns:
        print("Excel file must contain a 'Condition' column.")
        return

    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Separate the data based on the 'Condition' column
    condition_1_df = df[df['Condition'] == 1]  # Rows where Condition is 1
    condition_0_df = df[df['Condition'] == 0]  # Rows where Condition is 0

    # Define the output file paths
    base_dir = os.path.dirname(excel_file)
    file_1_path = os.path.join(base_dir, f'condition_1_{timestamp}.xlsx')
    file_0_path = os.path.join(base_dir, f'condition_0_{timestamp}.xlsx')

    # Save the two DataFrames to new Excel files
    condition_1_df.to_excel(file_1_path, index=False)
    condition_0_df.to_excel(file_0_path, index=False)

    print(f"Excel file for Condition 1 saved at: {file_1_path}")
    print(f"Excel file for Condition 0 saved at: {file_0_path}")

# Define the path to your Excel file
excel_file = r'F:\videos\test_set - Copy.xlsx'  # Replace with your actual file path

# Call the function to split the Excel file
split_excel_file(excel_file)
