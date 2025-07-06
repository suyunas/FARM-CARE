import pandas as pd
from sklearn.metrics import classification_report
import glob
import os

# Function to process each file
def process_file(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Filter out rows where 'AnimalID' is 'unknown' or missing
    df = df[df['AnimalID'].notna() & (df['AnimalID'].str.strip().str.lower() != 'unknown')]
    
    # Extract and format the model name from the filename
    base_parts = os.path.basename(file_path).replace('.csv', '').split('_')[2:]
    base_model_name = '_'.join(base_parts).lower()
    
    # Extract the true labels and AnimalIDs
    y_true = df['Condition']
    all_animal_ids = df['AnimalID'].unique()
    
    # Initialize a DataFrame for prediction summaries with 'AnimalID' and 'Condition' as initial columns
    summary_df = df[['AnimalID', 'Condition']].drop_duplicates().reset_index(drop=True)
    
    # Calculate total images for each AnimalID
    total_images = df.groupby('AnimalID').size().reindex(all_animal_ids, fill_value=0).reset_index(name='Total')
    
    # Merge total images into the summary DataFrame
    summary_df = summary_df.merge(total_images, on='AnimalID', how='left')
    
    # Initialize a list to store individual reports with model name
    reports_with_model_name = []
    
    # Loop through the columns that contain predictions to generate reports and summaries
    for col in df.columns[3:]:
        y_pred = df[col]
        
        # Generate the classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Extract the required metrics for each class
        class_metrics = {}
        for cls in report:
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                class_metrics[cls] = {
                    "precision": round(report[cls]['precision'], 3),
                    "recall": round(report[cls]['recall'], 3),
                    "f1-score": round(report[cls]['f1-score'], 3)
                }
        
        # Add macro average separately
        macro_avg = {
            "precision": round(report['macro avg']['precision'], 3),
            "recall": round(report['macro avg']['recall'], 3),
            "f1-score": round(report['macro avg']['f1-score'], 3)
        }
        class_metrics['macro avg'] = macro_avg
        
        # Convert the metrics to DataFrame and add model name as a column
        report_df = pd.DataFrame(class_metrics).T.reset_index().rename(columns={'index': 'class'})
        model_name = f"{base_model_name}_{col.lower()}"
        report_df['model'] = model_name
        reports_with_model_name.append(report_df)
        
        # Generate prediction summary with count of correctly classified images per AnimalID
        correct_predictions = df[df['Condition'] == df[col]]
        correct_count = correct_predictions.groupby('AnimalID').size().reindex(all_animal_ids, fill_value=0).reset_index(name=f'Correct_{model_name}')
        
        # Merge the correctly classified counts into the summary DataFrame
        summary_df = summary_df.merge(correct_count, on='AnimalID', how='left')
    
    return reports_with_model_name, summary_df

# Function to process all CSV files in a specified folder
def process_all_files_in_folder(folder_path):
    # Set up lists to collect data for consolidated output
    all_reports = []
    all_summaries = []
    
    # Find all CSV files in the specified folder (no subdirectories)
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    # Process each file
    for file_path in csv_files:
        reports, prediction_summary = process_file(file_path)
        
        # Append each report to the reports list
        for report in reports:
            all_reports.append(report)
        
        # Append each summary to the summaries list
        all_summaries.append(prediction_summary)
    
    # Combine all the reports into a single DataFrame
    final_report_df = pd.concat(all_reports, ignore_index=True)
    
    # Combine all summaries, grouping by AnimalID and Condition to aggregate across files
    final_summary_df = pd.concat(all_summaries).groupby(['AnimalID', 'Condition']).first().reset_index()

    # Save the consolidated DataFrames to CSV files
    classification_report_file = os.path.join(folder_path, 'all_classification_report.csv')
    prediction_summary_file = os.path.join(folder_path, 'all_prediction_summary.csv')
    
    # Save the consolidated reports and summaries
    final_report_df.to_csv(classification_report_file, index=False)
    final_summary_df.to_csv(prediction_summary_file, index=False)
    
    print(f"Consolidated classification report saved to {classification_report_file}")
    print(f"Consolidated prediction summary saved to {prediction_summary_file}")

    # Process the prediction summary for the switch logic
    process_prediction_summary_switch(final_summary_df, folder_path)

    # Generate classification report for the prediction summary switch file
    generate_classification_report_from_switch(os.path.join(folder_path, 'prediction_summary_switch.csv'))

# Function to process the prediction summary switch logic
def process_prediction_summary_switch(summary_df, folder_path):
    # Define the output file path
    output_file = os.path.join(folder_path, 'prediction_summary_switch.csv')
    
    # Remove the 'Total' column if it exists
    if 'Total' in summary_df.columns:
        summary_df = summary_df.drop(columns=['Total'])
    
    # Define the range of columns to be processed (from D onwards)
    columns_to_process = summary_df.columns[2:]  # Columns D to AM
    
    # Define a function to switch values based on the Condition
    def process_row(row):
        for col in columns_to_process:
            if row['Condition'] == 1:
                # If Condition is 1, set column value to 1 if it was greater than 0, otherwise 0
                row[col] = 1 if row[col] > 0 else 0
            else:
                # If Condition is 0, set column value to 0 if it was greater than 0, otherwise 1
                row[col] = 0 if row[col] > 0 else 1
        return row
    
    # Apply the function to each row
    summary_df = summary_df.apply(process_row, axis=1)
    
    # Save the updated data to a new CSV file
    summary_df.to_csv(output_file, index=False)
    
    print(f'Processed data saved to {output_file}')

# Function to generate a classification report from the processed prediction summary
def generate_classification_report_from_switch(file_path):
    # Load the processed prediction summary CSV file
    df = pd.read_csv(file_path)
    
    # Extract the true labels and predictions
    y_true = df['Condition']
    predictions = df.columns[2:]  # Columns from D onwards
    
    # Initialize a list to store individual classification reports
    all_reports = []
    
    # Loop through the prediction columns to generate reports
    for col in predictions:
        y_pred = df[col]
        
        # Generate the classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Extract the required metrics for each class
        class_metrics = {}
        for cls in report:
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                class_metrics[cls] = {
                    "precision": round(report[cls]['precision'], 3),
                    "recall": round(report[cls]['recall'], 3),
                    "f1-score": round(report[cls]['f1-score'], 3)
                }
        
        # Add macro average separately
        macro_avg = {
            "precision": round(report['macro avg']['precision'], 3),
            "recall": round(report['macro avg']['recall'], 3),
            "f1-score": round(report['macro avg']['f1-score'], 3)
        }
        class_metrics['macro avg'] = macro_avg
        
        # Convert the metrics to DataFrame and add column name as a model identifier
        report_df = pd.DataFrame(class_metrics).T.reset_index().rename(columns={'index': 'class'})
        report_df['model'] = col
        
        # Append the report to the list
        all_reports.append(report_df)
    
    # Combine all reports into a single DataFrame
    final_report_df = pd.concat(all_reports, ignore_index=True)
    
    # Define output file path for the classification report
    output_file = os.path.join(os.path.dirname(file_path), 'classification_report_from_switch.csv')
    
    # Save the classification report to a new CSV file
    final_report_df.to_csv(output_file, index=False)
    
    print(f'Classification report saved to {output_file}')

# Define folder path
folder_path = r'J:\F2-Daughters\B1\clustering_results_D70'
process_all_files_in_folder(folder_path)
