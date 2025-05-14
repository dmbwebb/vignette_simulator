import pandas as pd
import glob
import os
import re # Import the re module for regex operations

# Define the path to the folder containing the CSV files
folder_path = "outputs/curated_outputs"

# Define the pattern for the files to be concatenated
file_pattern = "summary_*.csv"

# Create the full path pattern
full_pattern = os.path.join(folder_path, file_pattern)

# Get a list of all files matching the pattern
file_list = glob.glob(full_pattern)

# Initialize an empty list to store DataFrames
dfs = []

# Regex to extract date and time from filename (e.g., summary_20250513_182327.csv)
filename_pattern = re.compile(r"summary_(\d{8}_\d{6})\.csv")

# Loop through the file list and read each CSV into a DataFrame
if not file_list:
    print(f"No files found matching the pattern: {full_pattern}")
else:
    for file in file_list:
        try:
            # Extract the date-time stamp from the filename
            match = filename_pattern.search(os.path.basename(file))
            timestamp = None
            if match:
                timestamp_str = match.group(1)
                # Reformat to a more standard YYYY-MM-DD HH:MM:SS
                timestamp = f"{timestamp_str[:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]} {timestamp_str[9:11]}:{timestamp_str[11:13]}:{timestamp_str[13:15]}"
            else:
                print(f"Warning: Could not extract timestamp from filename: {file}")

            df = pd.read_csv(file)
            
            # Add the timestamp as a new column
            if timestamp:
                df['source_timestamp'] = timestamp
            else:
                df['source_timestamp'] = pd.NaT # Or some other placeholder if preferred

            dfs.append(df)
            print(f"Successfully read {file} and added timestamp: {timestamp}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Concatenate all DataFrames in the list
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)

        # Define the path for the output file
        output_file_path = os.path.join(folder_path, "combined_summary_with_timestamps.csv") # Changed output filename

        # Write the combined DataFrame to a new CSV file
        try:
            combined_df.to_csv(output_file_path, index=False)
            print(f"Successfully concatenated files into {output_file_path}")
            print(f"The combined file has {len(combined_df)} rows.")
        except Exception as e:
            print(f"Error writing combined file: {e}")
    else:
        print("No dataframes to concatenate.") 