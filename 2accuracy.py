#this file create the accuracy files



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from scipy import stats
from pathlib import Path
from typing import Tuple, Optional, Dict

classroom_output_path = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy\results_output_for_accuracy_Classroom.xlsx"
cafe_output_path = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy\results_output_for_accuracy_Cafe.xlsx"    
output_path = r"C:\Users\peleg\Desktop\Lub\comparison_cafe_vs_classroom.png"
stats_output_path = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy\results_output_for_accuracy_all.xlsx"
CONDITION_FILE_PATH = r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\assignment_log.csv"

# CONSTANTS
ADHD = 0
Control = 1
CAFE = "cafe"
CLASSROOM = "classroom"
NUMBER_TEST = 1
NUMBER_RETEST = 2
MAX_TRIAL_ID = 30
MAX_QUESTION_ID = 7777

# PATHS - Import from config
try:
    from config import *
    print(" Configuration loaded successfully")
except ImportError:
    print(" Warning: config.py not found, using default paths")
    # Keep original paths as fallback
    CAFE_DATA_PATH = r"C:\Users\peleg\Desktop\Lub\assignment_log_cafe.csv"
    CLASSROOM_DATA_PATH = r"C:\Users\peleg\Desktop\Lub\assignment_log_classroom.csv"
    DATA_PATH_ADHD = r"C:\Users\peleg\Desktop\Lub\ADHD_group.xlsx"
    DATA_PATH = r"C:\Users\peleg\Desktop\Lub\All_Answers"
    OUTPUT_DIR = r"C:\Users\peleg\Desktop\Lub\Accuracy_Analysis"

# UTILITY FUNCTIONS

def load_metadata_files(cafe_log_path: str, classroom_log_path: str) -> dict:
    # Loading all the data from the environments
    return {
        'cafe': pd.read_csv(cafe_log_path),
        'classroom': pd.read_csv(classroom_log_path)
    }

def load_condition_mapping(condition_file_path: str) -> dict:
    # Loading CSV that maps the conditions of trial to subject
    if not os.path.exists(condition_file_path):
        print(f"ERROR: File does not exist: {condition_file_path}")
        return {}
    
    try:
        condition_df = pd.read_csv(condition_file_path)
    except Exception as e:
        print(f" ERROR loading file: {e}")
        return {}

    required_columns = ['Subject', 'Session', 'Condition']
    missing_columns = [col for col in required_columns if col not in condition_df.columns]

    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        return {}

    condition_df = condition_df.dropna(subset=['Subject', 'Session', 'Condition'])

    # Build dictionary mapping (subject, session) to condition string
    condition_map = {}
    for _, row in condition_df.iterrows():
        try:
            key = (int(row['Subject']), int(row['Session']))
            condition_map[key] = str(row['Condition']).strip()
        except (ValueError, TypeError):
            continue

    print(f" Loaded {len(condition_map)} condition mappings")
    return condition_map

def extract_subject_number(filepath: str) -> int:
    # Returning the number of the subject as int
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

def load_file(file_path: str) -> pd.DataFrame:
    #Loading a file for processing
    try:
        return pd.read_excel(file_path)
    except Exception:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f" Failed to load: {os.path.basename(file_path)} {e}")
            return None

def test_or_retest(file_path: str) -> int:
    # Identify if test or retest
    return NUMBER_RETEST if "retest" in os.path.basename(file_path).lower() \
        else NUMBER_TEST

def identify_envi(participant_id: int, session: int, metadata: dict) -> str:
    # Identify if cafe or classroom environment
    #first cafe:
    cafe_match = metadata['cafe'][
        (metadata['cafe']['Subject'] == participant_id) &
        (metadata['cafe']['Session'] == session)
    ]
    if not cafe_match.empty:
        return "CAFE"
#second classroom
    classroom_match = metadata['classroom'][
        (metadata['classroom']['Subject'] == participant_id) &
        (metadata['classroom']['Session'] == session)
    ]
    if not classroom_match.empty:
        return "CLASSROOM"

    return None

def identify_ADHD_or_Control(participant_id: int) -> int:
    # Identify if ADHD or Control group
    ADHD_Control = pd.read_excel(DATA_PATH_ADHD)
    matching_row = ADHD_Control[
        (ADHD_Control['Part_id'] == participant_id) &
        (ADHD_Control['which'] == 'V')
    ]
    return ADHD if not matching_row.empty \
        else Control

def parse_condition_sequence(condition_str: str) -> tuple:
    # Identify order of the experiment (R, IR, M)
    conditions = []
    i = 0
    while i < len(condition_str):
        if i < len(condition_str) - 1 and condition_str[i:i + 2] == 'IR':
            conditions.append('IR')
            i += 2
        else:
            conditions.append(condition_str[i])
            i += 1

    if len(conditions) != 3:
        print(f"Warning: Expected 3 conditions, got {len(conditions)} from '{condition_str}'")
        return (None, None, None)

    return tuple(conditions)

def identify_condition_for_trial(condition_str: str, trial_id: int) -> str:
    # Identify which condition belongs to each trial (R, IR, M) based on trial ID
    if trial_id < 1 or trial_id > 30:
        return None

    cond_1, cond_2, cond_3 = parse_condition_sequence(condition_str)

    if 1 <= trial_id <= 10:
        return cond_1
    elif 11 <= trial_id <= 20:
        return cond_2
    elif 21 <= trial_id <= 30:
        return cond_3

    return None

# ACCURACY CALCULATION FUNCTIONS

def filtered_data_frame_for_accuracy(file_to_read: pd.DataFrame) -> pd.DataFrame:
    # Filter what data to read - only valid trials and questions
    return file_to_read[
        (file_to_read['TrialID'] <= MAX_TRIAL_ID) &
        (file_to_read['QuestionID'] < MAX_QUESTION_ID)
    ]

def calculate_accuracy_for_trials(filtered_df: pd.DataFrame, trial_ids: list) -> float:
    # Calculate the accuracy for trials in subject data
    if not trial_ids:
        return 0.0

    trial_data = filtered_df[filtered_df['TrialID'].isin(trial_ids)]

    if len(trial_data) == 0:
        return 0.0

    correct = trial_data[trial_data['SingleChoiceAccurate'] == True]
    return (len(correct) / len(trial_data)) * 100

def calculate_avg_accuracy_for_condition(filtered_df: pd.DataFrame,
                                         condition_str: str,
                                         condition_type: str) -> float:
    # Calculate the average accuracy for a specific condition type (R, IR, or M)
    condition_trials = []
    for trial_id in range(1, 31):
        trial_condition = identify_condition_for_trial(condition_str, trial_id)
        if trial_condition == condition_type:
            condition_trials.append(trial_id)

    return calculate_accuracy_for_trials(filtered_df, condition_trials)

# MAIN PROCESSING FUNCTIONS

    # Running all the files and returning the aggregated data
    all_results = []
    metadata = load_metadata_files(CAFE_DATA_PATH, CLASSROOM_DATA_PATH)

    condition_map = load_condition_mapping(CONDITION_FILE_PATH)

    if not condition_map:
        print("ERROR: Condition map is empty!")
        return pd.DataFrame()

    # Collect all files to process
    xlsx_files = glob.glob(os.path.join(DATA_PATH, '*.xlsx'))
    csv_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))
    all_files_to_process = xlsx_files + csv_files

    # Process each file
    for file in all_files_to_process:
        subject_number = extract_subject_number(file)
        session = test_or_retest(file)
        condition_str = condition_map.get((subject_number, session))

        if condition_str is None:
            continue

        which_envi = identify_envi(subject_number, session, metadata)
        is_adhd = identify_ADHD_or_Control(subject_number)

        raw_df = load_file(file)
        if raw_df is None:
            continue

        filtered_df = filtered_data_frame_for_accuracy(raw_df)

        if filtered_df.empty:
            continue

        # Calculate accuracy for each condition type
        for condition_type in ['R', 'IR', 'M']:
            avg_accuracy = calculate_avg_accuracy_for_condition(
                filtered_df, condition_str, condition_type
            )

            all_results.append({
                'Subject': subject_number,
                'Session': session,
                'Environment': which_envi,
                'Group': is_adhd,
                'Condition': condition_type,
                'Condition_Sequence': condition_str,
                'Accuracy': avg_accuracy
            })

    if not all_results:
        print("ERROR: No results generated!")
        return pd.DataFrame()

    return pd.DataFrame(all_results)

def results_for_one_env(which_envi: str) -> pd.DataFrame:
    #Process and return results for a specific environment (cafe or classroom)
    results = []

    # Select the correct condition file based on environment
    if which_envi.lower() == "cafe":
        CONDITION_FILE_PATH = CAFE_DATA_PATH
    else:
        CONDITION_FILE_PATH = CLASSROOM_DATA_PATH

    metadata = load_metadata_files(CAFE_DATA_PATH, CLASSROOM_DATA_PATH)
    condition_map = load_condition_mapping(CONDITION_FILE_PATH)

    if not condition_map:
        print("ERROR: Condition map is empty!")
        return pd.DataFrame()

    # Collect all files to process
    xlsx_files = glob.glob(os.path.join(DATA_PATH, '*.xlsx'))
    csv_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))
    all_files_to_process = xlsx_files + csv_files

    files_processed = 0
    files_skipped = 0

    # Process each file, filtering by environment
    for file in all_files_to_process:
        subject_number = extract_subject_number(file)
        session = test_or_retest(file)
        condition_str = condition_map.get((subject_number, session))

        if condition_str is None:
            continue

        file_environment = identify_envi(subject_number, session, metadata)
        # Skip if file is not from the requested environment
        if file_environment != which_envi.upper():
            files_skipped += 1
            continue

        is_adhd = identify_ADHD_or_Control(subject_number)

        raw_df = load_file(file)
        if raw_df is None:
            files_skipped += 1
            continue

        filtered_df = filtered_data_frame_for_accuracy(raw_df)

        if filtered_df.empty:
            files_skipped += 1
            continue

        # Calculate accuracy for each condition type
        for condition_type in ['R', 'IR', 'M']:
            avg_accuracy = calculate_avg_accuracy_for_condition(
                filtered_df, condition_str, condition_type
            )

            results.append({
                'Subject': subject_number,
                'Session': session,
                'Environment': file_environment,
                'Group': is_adhd,
                'Condition': condition_type,
                'Condition_Sequence': condition_str,
                'Accuracy': avg_accuracy
            })
            files_processed += 1

    if not results:
        print(" ERROR: No results generated!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(f" Processed {files_processed} entries from {which_envi}")
    return df

def calculate_std_per_subject_from_raw_data(output_path: str):
    """Calculate standard deviation per subject from raw trial data - FOR ACCURACY"""
    all_results = []
    metadata = load_metadata_files(CAFE_DATA_PATH, CLASSROOM_DATA_PATH)
    
    cafe_condition_map = load_condition_mapping(CAFE_DATA_PATH)
    classroom_condition_map = load_condition_mapping(CONDITION_FILE_PATH)
    
    xlsx_files = glob.glob(os.path.join(DATA_PATH, '*.xlsx'))
    csv_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))
    all_files = xlsx_files + csv_files
    
    print("\n" + "="*80)
    print("CALCULATING SD AND MEDIAN FROM RAW TRIAL DATA - ACCURACY")
    print("="*80)
    
    for file in all_files:
        subject_number = extract_subject_number(file)
        session = test_or_retest(file)
        environment = identify_envi(subject_number, session, metadata)
        
        if environment is None:
            continue
        
        if environment == "CAFE":
            condition_map = cafe_condition_map
        else:
            condition_map = classroom_condition_map
        
        condition_str = condition_map.get((subject_number, session))
        if condition_str is None:
            continue
        
        raw_df = load_file(file)
        if raw_df is None:
            continue
        

        filtered_df = filtered_data_frame_for_accuracy(raw_df)
        if filtered_df.empty:
            continue
        
        print(f"Processing Subject {subject_number}, Session {session}, Environment {environment}")
        
        #calcultae for each trail
        for condition_type in ['R', 'IR', 'M']:
            condition_trials = []
            condition_accuracies = []
            
        
            for trial_id in range(1, 31):
                trial_condition = identify_condition_for_trial(condition_str, trial_id)
                if trial_condition == condition_type:
                    # acuuracy calculate for trail
                    trial_data = filtered_df[filtered_df['TrialID'] == trial_id]
                    if len(trial_data) > 0:
                        correct = trial_data[trial_data['SingleChoiceAccurate'] == True]
                        accuracy = (len(correct) / len(trial_data)) * 100
                        condition_trials.append(trial_id)
                        condition_accuracies.append(accuracy)
            
            if len(condition_accuracies) > 0:
                mean_val = np.mean(condition_accuracies)
                median_val = np.median(condition_accuracies)
                std_val = np.std(condition_accuracies, ddof=1) if len(condition_accuracies) > 1 else 0.0
                
                all_results.append({
                    'Subject': subject_number,
                    'Session': session,
                    'Environment': environment,
                    'Condition': condition_type,
                    'Mean': mean_val,
                    'Median': median_val,
                    'SD': std_val,
                    'N_Trials': len(condition_accuracies),
                })
        
        # Overall statistics
        all_accuracies = []
        for trial_id in range(1, 31):
            trial_data = filtered_df[filtered_df['TrialID'] == trial_id]
            if len(trial_data) > 0:
                correct = trial_data[trial_data['SingleChoiceAccurate'] == True]
                accuracy = (len(correct) / len(trial_data)) * 100
                all_accuracies.append(accuracy)
        
        if len(all_accuracies) > 0:
            all_results.append({
                'Subject': subject_number,
                'Session': session,
                'Environment': environment,
                'Condition': 'Overall',
                'Mean': np.mean(all_accuracies),
                'Median': np.median(all_accuracies),
                'SD': np.std(all_accuracies, ddof=1) if len(all_accuracies) > 1 else 0.0,
                'N_Trials': len(all_accuracies),
            })
    
    results_df = pd.DataFrame(all_results)
    
    # Sort results
    condition_order = {'Overall': 0, 'R': 1, 'IR': 2, 'M': 3}
    results_df['Condition_Order'] = results_df['Condition'].map(condition_order)
    results_df = results_df.sort_values(['Subject', 'Session', 'Environment', 'Condition_Order'])
    results_df = results_df.drop('Condition_Order', axis=1)
    
    # Round values
    results_df['Mean'] = results_df['Mean'].round(4)
    results_df['Median'] = results_df['Median'].round(4)
    results_df['SD'] = results_df['SD'].round(4)
    
    results_df.to_excel(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    print(f"Total rows: {len(results_df)}")
    
    return results_df

def calculate_cv(mean, std):
    # Calculate coefficient of variation (CV) as percentage
    if mean == 0 or pd.isna(mean):
        return 0.0
    return (std / mean) * 100

def main():
        #  Cafe
    cafe_df = results_for_one_env("cafe")
    cafe_df.to_excel(cafe_output_path, index=False)
#  Classroom
    classroom_df = results_for_one_env("classroom")
    classroom_df.to_excel(classroom_output_path, index=False)

#combined
    combiend_df= pd.concat([cafe_df, classroom_df],ignore_index=True)
    stats_df = calculate_std_per_subject_from_raw_data(stats_output_path)


if __name__== "__main__":
    main()

