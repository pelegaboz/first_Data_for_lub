"""
Interest Level Analysis
Analyzes interest levels of participants across different conditions and environments
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from pathlib import Path
# Get the base directory (where the script is located)
BASE_DIR = Path(__file__).parent



# CONFIGURATION - Import from config file
try:
    from config import *
    print(" Configuration loaded successfully")
except ImportError:
    print(" Warning: config.py not found, using default paths")
    # Constants
    ADHD = 0
    Control = 1
    CAFE = "cafe"
    CLASSROOM = "classroom"
    NUMBER_TEST = 1
    NUMBER_RETEST = 2
    INTRESED_LEVEL_QUASTION = 7777
    MAX_TRIAL_ID = 30
    # Paths - relative to script location
    CAFE_DATA_PATH = BASE_DIR / "Input_Data" / "assignment_log_cafe.csv"
    CLASSROOM_DATA_PATH = BASE_DIR / "Input_Data" / "assignment_log_classroom.csv"
    DATA_PATH_ADHD = BASE_DIR / "Input_Data" / "ADHD_group.xlsx"
    DATA_PATH = BASE_DIR / "Input_Data" / "All_Answers"
def load_metadata_files(cafe_log_path: str, classroom_log_path: str):
    """Load metadata files for both environments"""
    return {
        'cafe': pd.read_csv(cafe_log_path),
        'classroom': pd.read_csv(classroom_log_path)
    }


def extract_subject_number(filepath: str) -> int:
    """Extract subject number from filename"""
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None


def load_file(file_path: str):
    """Load Excel or CSV file"""
    try:
        return pd.read_excel(file_path)
    except Exception:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"✗ Failed to load: {os.path.basename(file_path)} {e}")
            return None


def test_or_retest(file_path: str) -> int:
    """Determine if file is test or retest"""
    if "retest" in os.path.basename(file_path).lower():
        return NUMBER_RETEST
    else:
        return NUMBER_TEST


def identify_envi(participant_id: int, session: int, metadata):
    """Identify environment (Cafe or Classroom) for participant"""
    cafe_match = metadata['cafe'][
        (metadata['cafe']['Subject'] == participant_id) &
        (metadata['cafe']['Session'] == session)
    ]
    if not cafe_match.empty:
        return "CAFE"
    
    classroom_match = metadata['classroom'][
        (metadata['classroom']['Subject'] == participant_id) &
        (metadata['classroom']['Session'] == session)
    ]
    if not classroom_match.empty:
        return "CLASSROOM"
    
    return None


def identify_ADHD_or_Control(participant_id: int):
    """Identify if participant is ADHD or Control group"""
    ADHD_Control = pd.read_excel(DATA_PATH_ADHD)
    matching_row = ADHD_Control[
        (ADHD_Control['Part_id'] == participant_id) &
        (ADHD_Control['which'] == 'V')
    ]
    if not matching_row.empty:
        return ADHD
    else:
        return Control


def fillterd_data_frame_for_intrest(file_to_read):
    """Filter dataframe for interest level questions"""
    return file_to_read[
        (file_to_read['TrialID'] <= MAX_TRIAL_ID) &
        (file_to_read['QuestionID'] == INTRESED_LEVEL_QUASTION)
    ]

def interest_level_for_one_question(number_of_quastion: int, filltered_df) -> int:
    """Get interest level for a single question"""
    question_rows = filltered_df[filltered_df['TrialID'] == number_of_quastion]

    if question_rows.empty:
        return 0

    row = question_rows.iloc[0]
    for index in range(2, 9):
        column = f"ReportedAnswer{index}"
        if column in row.index and row[column] == True:
            return index - 1
    return 0

def parse_condition_sequence(condition_str: str) -> tuple:
    """Parse condition sequence string into tuple"""
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


def load_condition_mapping(condition_file_path: str) -> dict:
    """Load condition mapping from CSV file"""
    if not os.path.exists(condition_file_path):
        print(f"✗ ERROR: File does not exist: {condition_file_path}")
        return {}

    try:
        condition_df = pd.read_csv(condition_file_path)
    except Exception as e:
        print(f"✗ ERROR loading file: {e}")
        return {}

    required_columns = ['Subject', 'Session', 'Condition']
    missing_columns = [col for col in required_columns if col not in condition_df.columns]

    if missing_columns:
        print(f"✗ ERROR: Missing columns: {missing_columns}")
        return {}

    condition_df = condition_df.dropna(subset=['Subject', 'Session', 'Condition'])

    condition_map = {}
    for _, row in condition_df.iterrows():
        try:
            key = (int(row['Subject']), int(row['Session']))
            condition_map[key] = str(row['Condition']).strip()
        except (ValueError, TypeError):
            continue

    print(f"✓ Loaded {len(condition_map)} condition mappings")
    return condition_map


def identify_condition_for_trial(condition_str: str, trial_id: int) -> str:
    """Identify condition for specific trial"""
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


def calculate_avg_interest_for_condition(filtered_df, condition_str: str, condition_type: str) -> float:
    """Calculate average interest level for specific condition"""
    condition_trials = []
    for trial_id in range(1, 31):
        trial_condition = identify_condition_for_trial(condition_str, trial_id)
        if trial_condition == condition_type:
            condition_trials.append(trial_id)

    if not condition_trials:
        return 0

    total_interest = 0
    count = 0

    for trial_id in condition_trials:
        level = interest_level_for_one_question(trial_id, filtered_df)
        if level is not None and level > 0:
            total_interest += level
            count += 1

    if count == 0:
        return 0

    return total_interest / len(condition_trials)



def results_For_one_envi(which_envi: str):
    """Process files for one environment"""
    results = []
    if which_envi.lower() == "cafe":
        metadata = load_metadata_files(CAFE_DATA_PATH, CLASSROOM_DATA_PATH)
        CONDITION_FILE_PATH = CAFE_DATA_PATH
        condition_map = load_condition_mapping(CONDITION_FILE_PATH)
    else:
        metadata = load_metadata_files(CAFE_DATA_PATH, CLASSROOM_DATA_PATH)
        CONDITION_FILE_PATH = CLASSROOM_DATA_PATH
        condition_map = load_condition_mapping(CONDITION_FILE_PATH)
    
    if not condition_map:
        print("✗ ERROR: Condition map is empty!")
        return pd.DataFrame()

    xlsx_files = glob.glob(os.path.join(DATA_PATH, '*.xlsx'))
    csv_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))
    all_files_to_process = xlsx_files + csv_files

    files_processed = 0
    files_skipped = 0

    for file in all_files_to_process:
        subject_number = extract_subject_number(file)
        session = test_or_retest(file)
        condition_str = condition_map.get((subject_number, session))

        if condition_str is None:
            continue

        file_environment = identify_envi(subject_number, session, metadata)
        if file_environment != which_envi.upper():
            files_skipped += 1
            continue
        
        is_adhd = identify_ADHD_or_Control(subject_number)

        raw_df = load_file(file)
        if raw_df is None:
            files_skipped += 1
            continue

        filtered_df = fillterd_data_frame_for_intrest(raw_df)

        if filtered_df.empty:
            files_skipped += 1
            continue

        for condition_type in ['R', 'IR', 'M']:
            avg_interest = calculate_avg_interest_for_condition(
                filtered_df, condition_str, condition_type
            )

            results.append({
                'Subject': subject_number,
                'Session': session,
                'Environment': file_environment,
                'Group': is_adhd,
                'Condition': condition_type,
                'Condition_Sequence': condition_str,
                'Interest_Level': avg_interest
            })
            files_processed += 1

    if not results:
        print("✗ ERROR: No results generated!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    print(f"✓ Processed {files_processed} entries from {which_envi}")
    return df


def calculate_std_per_subject_from_raw_data(output_path: str):
    """Calculate standard deviation per subject from raw trial data"""
    all_results = []
    metadata = load_metadata_files(CAFE_DATA_PATH, CLASSROOM_DATA_PATH)
    
    cafe_condition_map = load_condition_mapping(CAFE_DATA_PATH)
    classroom_condition_map = load_condition_mapping(CLASSROOM_DATA_PATH)
    
    xlsx_files = glob.glob(os.path.join(DATA_PATH, '*.xlsx'))
    csv_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))
    all_files = xlsx_files + csv_files
    
    print("\n" + "="*80)
    print("CALCULATING SD AND MEDIAN FROM RAW TRIAL DATA")
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
        
        filtered_df = fillterd_data_frame_for_intrest(raw_df)
        if filtered_df.empty:
            continue
        
        print(f"Processing Subject {subject_number}, Session {session}, Environment {environment}")
        
        for condition_type in ['R', 'IR', 'M']:
            condition_trials = []
            condition_values = []
            
            for trial_id in range(1, 31):
                trial_condition = identify_condition_for_trial(condition_str, trial_id)
                if trial_condition == condition_type:
                    level = interest_level_for_one_question(trial_id, filtered_df)
                    if level is not None and level > 0:
                        condition_trials.append(trial_id)
                        condition_values.append(level)
            
            if len(condition_values) > 0:
                mean_val = np.mean(condition_values)
                median_val = np.median(condition_values)
                std_val = np.std(condition_values, ddof=1) if len(condition_values) > 1 else 0.0
                
                all_results.append({
                    'Subject': subject_number,
                    'Session': session,
                    'Environment': environment,
                    'Condition': condition_type,
                    'Mean': mean_val,
                    'Median': median_val,
                    'SD': std_val,
                    'N_Trials': len(condition_values),
                })
        
        # Overall statistics
        all_values = []
        for trial_id in range(1, 31):
            level = interest_level_for_one_question(trial_id, filtered_df)
            if level is not None and level > 0:
                all_values.append(level)
        
        if len(all_values) > 0:
            all_results.append({
                'Subject': subject_number,
                'Session': session,
                'Environment': environment,
                'Condition': 'Overall',
                'Mean': np.mean(all_values),
                'Median': np.median(all_values),
                'SD': np.std(all_values, ddof=1) if len(all_values) > 1 else 0.0,
                'N_Trials': len(all_values),
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


def main():
    """Main execution function"""
    # create Cafe's files
    print("Creating Cafe data...")
    cafe_df = results_For_one_envi("cafe")
    cafe_output_path = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl\results_output_for_intrestLVL_cafe.xlsx"
    cafe_df.to_excel(cafe_output_path, index=False)
    print(f" Cafe data saved to: {cafe_output_path}")

    # create Classroom's files
    print("\nCreating Classroom data...")
    classroom_df = results_For_one_envi("classroom")
    classroom_output_path = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl\results_output_for_intrestLVL_classroom.xlsx"
    classroom_df.to_excel(classroom_output_path, index=False)
    print(f"Classroom data saved to: {classroom_output_path}")

    # create complete table
    stats_output_path = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl\results_output_for_intrestLVL_all.xlsx"
    stats_df = calculate_std_per_subject_from_raw_data(stats_output_path)
    print(" All analyses completed!")


#  Tests 
def test_interest_level_for_one_question():
    df = pd.DataFrame({
        'TrialID':           [5],
        'QuestionID':        [7777],
        'ReportedAnswer2':   [False],
        'ReportedAnswer3':   [False],
        'ReportedAnswer4':   [True],
        'ReportedAnswer5':   [False],
        'ReportedAnswer6':   [False],
        'ReportedAnswer7':   [False],
        'ReportedAnswer8':   [False],
    })
    assert interest_level_for_one_question(5, df) == 3

    assert interest_level_for_one_question(99, df) == 0

    df_no_true = df.copy()
    df_no_true['ReportedAnswer4'] = False
    assert interest_level_for_one_question(5, df_no_true) == 0

if __name__ == "__main__":
    main()