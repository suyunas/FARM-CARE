import os
import pandas as pd
from collections import defaultdict

def count_frames_from_excel(excel_file):
    # Read the existing Excel file
    df = pd.read_excel(excel_file)

    # Initialize a dictionary to hold frame counts
    frame_counts = defaultdict(lambda: {'D70_pre': 0, 'D70_post': 0, 'D90_pre': 0, 'D90_post': 0})

    # List of animal IDs to check in the paths
    animal_ids = ['GF187', 'YF12749', 'YF12876A', 'YF12892', 'YF12876A', 'YF12876B', 'YF12447', 'YF12612', 'YF12752', 'YF13730', 'YF12697',
                  'YF12595', 'YF12811', 'OF62', 'OF174', 'YF13921', 'YF13922', 'YF12746', 'YF12750', 'YF13770', 'PF27', 'YF13835', 'PF25']

    # Iterate over the image paths in the DataFrame
    for path in df['Path']:
        # Determine the day (70 or 90) based on the directory path
        day = None
        if '_D70' in path:
            day = 'D70'
        elif '_D90' in path:
            day = 'D90'

        # Determine if the state is 'pre' or 'post'
        state = 'pre' if 'pre' in path.lower() else 'post'

        # Identify the animal ID in the path
        for animal_id in animal_ids:
            if animal_id in path:
                # Update the frame count for this animal, day, and state
                key = f'{day}_{state}'
                frame_counts[animal_id][key] += 1

    return frame_counts

def create_summary_excel(output_folder, frame_counts):
    # Create a DataFrame from the frame counts dictionary
    summary_data = []
    for animal_id, counts in frame_counts.items():
        summary_data.append({
            'Animal ID': animal_id,
            'D70_pre': counts['D70_pre'],
            'D70_post': counts['D70_post'],
            'D90_pre': counts['D90_pre'],
            'D90_post': counts['D90_post'],
        })
    
    df = pd.DataFrame(summary_data)

    # Define the output Excel file path
    excel_path = os.path.join(output_folder, 'frame_summary.xlsx')

    # Save the DataFrame to an Excel file
    df.to_excel(excel_path, index=False)

    print(f"Summary Excel file created at: {excel_path}")

if __name__ == '__main__':
    # Input paths from the user on the same line
    output_folder = r'E:\FarmCare'  # Output folder path
    existing_excel_file = r'E:\FarmCare\image_labels.xlsx'  # Path to the existing Excel file

    # Count the frames using the existing Excel file
    frame_counts = count_frames_from_excel(existing_excel_file)

    # Create a summary Excel file with the frame counts
    create_summary_excel(output_folder, frame_counts)