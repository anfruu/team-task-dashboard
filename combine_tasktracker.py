
import os
import pandas as pd

# Folder containing the Excel files
folder_path = r"C:\\Users\\ansalaza\\TEAM - Task Tracker"  # <-- Change this to your folder path
output_folder = r"C:\\Users\\ansalaza\\Dashboard PY"       # <-- Where you want the combined file saved

# Sheets to process
sheets = [
    'Task Tracker - Monday',
    'Task Tracker - Tuesday',
    'Task Tracker - Wednesday',
    'Task Tracker - Thursday',
    'Task Tracker - Friday'
]

# Initialize an empty list to store dataframes
data_frames = []

# Loop through all Excel files in the folder
for file in os.listdir(folder_path):
    if file.endswith('.xlsx') or file.endswith('.xlsm'):
        file_path = os.path.join(folder_path, file)
        try:
            # Read each sheet
            for sheet in sheets:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl', usecols='A:D')
                    df.columns = ['Task ID', 'Team Member', 'Task Description', 'Task Type']
                    # Extract day from sheet name
                    day = sheet.replace('Task Tracker - ', '').strip()
                    df['Day'] = day
                    # Drop rows where Task ID is NaN
                    df = df.dropna(subset=['Task ID'])
                    # Add Duration and Volume columns as 0
                    df['Duration'] = 0
                    df['Volume'] = 0
                    data_frames.append(df)
                except Exception:
                    continue  # Skip if sheet not found
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Combine all dataframes
if data_frames:
    combined_df = pd.concat(data_frames, ignore_index=True)
    # Save to Excel
    output_file = os.path.join(output_folder, 'Combined_TaskTracker_Test.xlsx')
    combined_df.to_excel(output_file, index=False)
    print(f"✅ Combined file created: {output_file} with {len(combined_df)} rows.")
else:
    print("⚠ No data found to combine.")
