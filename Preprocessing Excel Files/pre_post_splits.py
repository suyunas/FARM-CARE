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

# Split df_post into two DataFrames based on '1_D70' and '2_D90'
df_post_1_D70 = df_post[df_post['Path'].str.contains('1_D70', case=False, na=False)]
df_post_2_D90 = df_post[df_post['Path'].str.contains('2_D90', case=False, na=False)]

# Split df_pre into batches based on the specified criteria
batch_1 = df_pre[df_pre['Path'].str.contains('GF187|YF12595|YF12749|YF12811|YF12892', case=False, na=False)]
batch_2 = df_pre[df_pre['Path'].str.contains('OF174|YF12825B|YF12612|YF13921', case=False, na=False)]
batch_3 = df_pre[df_pre['Path'].str.contains('YF13922|YF12750|YF12752|PF27', case=False, na=False)]
batch_4 = df_pre[df_pre['Path'].str.contains('YF13730|YF13770|PF25|YF12697', case=False, na=False)]
batch_5 = df_pre[df_pre['Path'].str.contains('YF12876A|OF62|YF12876B|YF12447|YF12746|YF13835', case=False, na=False)]

# Split df_pre into two DataFrames based on '1_D70' and '2_D90'
df_pre_1_D70 = df_pre[df_pre['Path'].str.contains('1_D70', case=False, na=False)]
df_pre_2_D90 = df_pre[df_pre['Path'].str.contains('2_D90', case=False, na=False)]

# Count the number of images in each batch
count_batch_1 = batch_1.shape[0]
count_batch_2 = batch_2.shape[0]
count_batch_3 = batch_3.shape[0]
count_batch_4 = batch_4.shape[0]
count_batch_5 = batch_5.shape[0]

# Save the resulting DataFrames to new Excel files
df_pre.to_excel('pre_images.xlsx', index=False)
df_post.to_excel('post_images.xlsx', index=False)
df_post_1_D70.to_excel('post_images_1_D70.xlsx', index=False)
df_post_2_D90.to_excel('post_images_2_D90.xlsx', index=False)
df_pre_1_D70.to_excel('pre_images_1_D70.xlsx', index=False)
df_pre_2_D90.to_excel('pre_images_2_D90.xlsx', index=False)
batch_1.to_excel('batch_1.xlsx', index=False)
batch_2.to_excel('batch_2.xlsx', index=False)
batch_3.to_excel('batch_3.xlsx', index=False)
batch_4.to_excel('batch_4.xlsx', index=False)
batch_5.to_excel('batch_5.xlsx', index=False)

# # Print the counts and the first few rows of each DataFrame for verification
# print(f"Number of 'pre' images: {df_pre.shape[0]}")
# print(f"Number of 'post' images: {df_post.shape[0]}")
# print(f"Number of 'post' images with '1_D70': {df_post_1_D70.shape[0]}")
# print(f"Number of 'post' images with '2_D90': {df_post_2_D90.shape[0]}")
# print(f"Number of 'pre' images with '1_D70': {df_pre_1_D70.shape[0]}")
# print(f"Number of 'pre' images with '2_D90': {df_pre_2_D90.shape[0]}")
# print(f"Number of images in Batch 1: {count_batch_1}")
# print(f"Number of images in Batch 2: {count_batch_2}")
# print(f"Number of images in Batch 3: {count_batch_3}")
# print(f"Number of images in Batch 4: {count_batch_4}")
# print(f"Number of images in Batch 5: {count_batch_5}")

# print("\nBatch 1 DataFrame:")
# print(batch_1.head())
# print("\nBatch 2 DataFrame:")
# print(batch_2.head())
# print("\nBatch 3 DataFrame:")
# print(batch_3.head())
# print("\nBatch 4 DataFrame:")
# print(batch_4.head())
# print("\nBatch 5 DataFrame:")
# print(batch_5.head())
