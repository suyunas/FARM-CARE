import pandas as pd
import os

# Specify the path to your Excel file
file_path = 'E:\\FarmCare\\image_labels.xlsx'  # Update this to your file path

# Change the working directory to the folder where the Excel file exists
directory = os.path.dirname(file_path)
os.chdir(directory)

# Load the Excel file
df = pd.read_excel(file_path)

# Split the DataFrame based on 'Path' containing 'pre' or 'post'
df_pre = df[df['Path'].str.contains('pre', case=False, na=False)]
df_post = df[df['Path'].str.contains('post', case=False, na=False)]

# Create combined DataFrames for pre and post
df_combined = pd.concat([df_pre, df_post])

# Split df_combined into batches based on the specified criteria
batch_1_4_combined = df_combined[df_combined['Path'].str.contains('GF187|YF12595|YF12749|YF12811|YF12892|OF174|YF12825B|YF12612|YF13921|YF13922|YF12750|YF12752|PF27|YF13730|YF13770|PF25|YF12697', case=False, na=False)]
batch_5_combined = df_combined[df_combined['Path'].str.contains('YF12876A|OF62|YF12876B|YF12447|YF12746|YF13835', case=False, na=False)]

# Now filter these combined batches to only include 'post'
batch_1_4_post = batch_1_4_combined[batch_1_4_combined['Path'].str.contains('post', case=False, na=False)]
batch_5_post = batch_5_combined[batch_5_combined['Path'].str.contains('post', case=False, na=False)]

# Save the resulting DataFrames to new Excel files
batch_1_4_combined.to_excel('batch_1_4_combined.xlsx', index=False)
batch_5_combined.to_excel('batch_5_combined.xlsx', index=False)
batch_1_4_post.to_excel('batch_1_4_post.xlsx', index=False)
batch_5_post.to_excel('batch_5_post.xlsx', index=False)
