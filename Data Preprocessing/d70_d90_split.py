import pandas as pd
import os

# Load the input data
input_file = r'E:\FarmCare\batch_5_post.xlsx'  # Change this to the actual file path
df = pd.read_excel(input_file)

# Debug: Print the first few rows of the input data
print("First few rows of the input data:")
print(df.head())

# Split the DataFrame based on '1_D70' and '2_D90' in the Image Name column
df_1_D70 = df[df['Path'].str.contains(r'\\1_D70\\')]
df_2_D90 = df[df['Path'].str.contains(r'\\2_D90\\')]

# Determine the directory of the input file
input_dir = os.path.dirname(input_file)

# Define the output file paths for each subset
output_file_1_D70 = os.path.join(input_dir, 'images_1_D70.xlsx')
output_file_2_D90 = os.path.join(input_dir, 'images_2_D90.xlsx')

# Save the split DataFrames to separate Excel files
df_1_D70.to_excel(output_file_1_D70, index=False)
df_2_D90.to_excel(output_file_2_D90, index=False)

print(f"Images with '1_D70' saved to: {output_file_1_D70}")
print(f"Images with '2_D90' saved to: {output_file_2_D90}")
