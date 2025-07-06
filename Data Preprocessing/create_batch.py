import os
import pandas as pd
import itertools

def generate_train_test_sets(base_folder, batch_files, train_count=7, test_count=2):
    # Generate all combinations of batches (7 for training and 2 for testing)
    combinations = list(itertools.combinations(batch_files, train_count + test_count))

    batch_num = 1
    for comb in combinations:
        # Split into training and testing batches
        train_batches = comb[:train_count]
        test_batches = comb[train_count:]
        
        # Create training data from selected batches
        train_data = []
        for batch in train_batches:
            batch_file_path = os.path.join(base_folder, batch)
            df = pd.read_excel(batch_file_path)
            df['Batch'] = batch  # Add batch column for identification
            train_data.append(df)
        
        # Combine all training data into a single DataFrame
        train_data_df = pd.concat(train_data, ignore_index=True)
        # Save training data to Excel
        train_file_path = os.path.join(base_folder, f'train_data_batch_{batch_num}.xlsx')
        train_data_df.to_excel(train_file_path, index=False)
        print(f"Training data saved to {train_file_path}")
        
        # Create testing data from selected batches
        test_data = []
        for batch in test_batches:
            batch_file_path = os.path.join(base_folder, batch)
            df = pd.read_excel(batch_file_path)
            df['Batch'] = batch  # Add batch column for identification
            test_data.append(df)
        
        # Combine all testing data into a single DataFrame
        test_data_df = pd.concat(test_data, ignore_index=True)
        # Save testing data to Excel
        test_file_path = os.path.join(base_folder, f'test_data_batch_{batch_num}.xlsx')
        test_data_df.to_excel(test_file_path, index=False)
        print(f"Testing data saved to {test_file_path}")
        
        batch_num += 1

# Example usage
base_folder = "J:\\"  # Set to your actual base folder path
batch_files = [f for f in os.listdir(base_folder) if f.startswith("Batch_") and f.endswith(".xlsx")]

# Generate all train-test combinations
generate_train_test_sets(base_folder, batch_files)
