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

# CONSTANTS
ADHD = 0
Control = 1
CAFE = "cafe"
CLASSROOM = "classroom"
NUMBER_TEST = 1
NUMBER_RETEST = 2
MAX_TRIAL_ID = 30

# PATHS - Import from config
try:
    from config import *
    print("✓ Configuration loaded successfully")
except ImportError:
    print("⚠ Warning: config.py not found, using default paths")
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

def extract_subject_number(filepath: str) -> int:
    # Returning the number of the subject as int
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

def load_file(file_path: str) -> pd.DataFrame:
    # Loading a file for processing
    try:
        return pd.read_excel(file_path)
    except Exception:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"✗ Failed to load: {os.path.basename(file_path)} {e}")
            return None

def test_or_retest(file_path: str) -> int:
    # Identify if test or retest
    return NUMBER_RETEST if "retest" in os.path.basename(file_path).lower() \
        else NUMBER_TEST

def identify_envi(participant_id: int, session: int, metadata: dict) -> str:
    # Identify if cafe or classroom environment
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
        (file_to_read['QuestionID'] < 7777)
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

    CONDITION_FILE_PATH = r"C:\Users\peleg\Desktop\Lub\assignment_log.csv"
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
        print("✗ ERROR: No results generated!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(f"✓ Processed {files_processed} entries from {which_envi}")
    return df

def independent_ttest_adhd_vs_control(df, environment):
#returning the results of indipendent t-test
    env_data = df[df['Environment'] == environment.upper()].copy()

    if env_data.empty:
        print(f" No data found for environment: {environment}")
        return None

    subject_avg = env_data.groupby(['Subject', 'Group'])['Accuracy'].mean().reset_index()

    adhd_group = subject_avg[subject_avg['Group'] == ADHD]['Accuracy']
    control_group = subject_avg[subject_avg['Group'] == Control]['Accuracy']

    n_adhd = len(adhd_group)
    n_control = len(control_group)
    print(f"Number of ADHD subjects: {n_adhd}")
    print(f"Number of Control subjects: {n_control}")

    adhd_mean = adhd_group.mean()
    adhd_std = adhd_group.std()
    control_mean = control_group.mean()
    control_std = control_group.std()

    t_statistic, p_value = stats.ttest_ind(adhd_group, control_group, equal_var=True)

    pooled_std = np.sqrt(((n_adhd - 1) * adhd_std ** 2 + (n_control - 1) * control_std ** 2) /
                         (n_adhd + n_control - 2))
    cohens_d = (adhd_mean - control_mean) / pooled_std

    print(f"INDEPENDENT T-TEST: ADHD vs CONTROL ({environment.upper()})")
    print(f"\n{'ADHD GROUP':-^80}")
    print(f"  n = {n_adhd}")
    print(f"  Mean = {adhd_mean:.4f}%")
    print(f"  SD = {adhd_std:.4f}")
    print(f"  Variance = {adhd_std ** 2:.4f}")

    print(f"\n{'CONTROL GROUP':-^80}")
    print(f"  n = {n_control}")
    print(f"  Mean = {control_mean:.4f}%")
    print(f"  SD = {control_std:.4f}")
    print(f"  Variance = {control_std ** 2:.4f}")

    print(f"\n{'TEST RESULTS':-^80}")
    print(f"  t-statistic = {t_statistic:.4f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  degrees of freedom = {n_adhd + n_control - 2}")
    print(f"  Mean difference = {abs(adhd_mean - control_mean):.4f}%")

    if p_value < 0.001:
        significance = "*** (p < 0.001) - Highly significant!"
    elif p_value < 0.01:
        significance = "** (p < 0.01) - Very significant!"
    elif p_value < 0.05:
        significance = "* (p < 0.05) - Significant!"
    else:
        significance = "n.s. (p >= 0.05) - Not significant"

    print(f"  Significance: {significance}")

    print(f"\n{'EFFECT SIZE (Cohen d)':-^80}")
    print(f"  Cohen's d = {cohens_d:.4f}")

    if abs(cohens_d) < 0.2:
        effect_interpretation = "Negligible effect"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "Small effect"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "Medium effect"
    else:
        effect_interpretation = "Large effect"

    print(f"  Interpretation: {effect_interpretation}")

    print(f"\n{'CONCLUSION':-^80}")

    return {
        'environment': environment,
        'adhd_n': n_adhd,
        'adhd_mean': adhd_mean,
        'adhd_std': adhd_std,
        'control_n': n_control,
        'control_mean': control_mean,
        'control_std': control_std,
        't_statistic': t_statistic,
        'p_value': p_value,
        'df': n_adhd + n_control - 2,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }

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
                    level = intrest_level_for_one_quastion(trial_id, filtered_df)
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
            level = intrest_level_for_one_quastion(trial_id, filtered_df)
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
    # print(f"\n✓ Results saved to: {output_path}")
    print(f"Total rows: {len(results_df)}")
    
    return results_df

def calculate_std_per_subject_from_raw_data(output_path: str):
    """Calculate standard deviation per subject from raw trial data - FOR ACCURACY"""
    all_results = []
    metadata = load_metadata_files(CAFE_DATA_PATH, CLASSROOM_DATA_PATH)
    
    cafe_condition_map = load_condition_mapping(CAFE_DATA_PATH)
    classroom_condition_map = load_condition_mapping(CLASSROOM_DATA_PATH)
    
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
    #create_overall_comparison_chart(cafe_df, classroom_df,output_path)


if __name__== "__main__":
    main()


"""



# VISUALIZATION FUNCTIONS

def load_condition_mapping(condition_file_path: str) -> dict:
    # Loading CSV that maps the conditions of trial to subject
    if not os.path.exists(condition_file_path):
        print(f"ERROR: File does not exist: {condition_file_path}")
        return {}

    try:
        condition_df = pd.read_csv(condition_file_path)
    except Exception as e:
        print(f"✗ ERROR loading file: {e}")
        return {}

    required_columns = ['Subject', 'Session', 'Condition']
    missing_columns = [col for col in required_columns if col not in condition_df.columns]

    if missing_columns:
        print(f" ERROR: Missing columns: {missing_columns}")
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

    print(f"✓ Loaded {len(condition_map)} condition mappings")
    return condition_map



def slope_bar_accuracy(df: pd.DataFrame):
    # Create slope graph showing accuracy change between cafe and classroom for each subject
    
    # Pivot data to have one row per subject with cafe and classroom columns
    pivot_df = df.pivot_table(
        values="Accuracy",
        index="Subject",
        columns="Environment",
        aggfunc="mean"
    )
    group_map = df.drop_duplicates('Subject').set_index('Subject')['Group']
    pivot_df = pivot_df.dropna()

    print(f"Found {len(pivot_df)} participants with data from both environments")
    print(f"Participants: {list(pivot_df.index)}")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create unique color for each subject
    n_subjects = len(pivot_df)
    colors = plt.cm.tab20(np.linspace(0, 1, n_subjects))

    # Plot line for each subject connecting cafe to classroom
    for idx, subject_id in enumerate(pivot_df.index):
        if subject_id not in group_map.index:
            print(f"Warning: Subject {subject_id} not in group_map, skipping")
            continue

        cafe_value = pivot_df.loc[subject_id, 'CAFE']
        classroom_value = pivot_df.loc[subject_id, 'CLASSROOM']

        group = group_map.loc[subject_id]
        color = colors[idx]

        # Draw line from cafe (x=0) to classroom (x=1)
        ax.plot([0, 1], [cafe_value, classroom_value],
                marker='o', markersize=8, linewidth=2.5,
                alpha=0.7, color=color, label=f'Subject {subject_id}')

        # Add subject ID label on the left
        ax.text(-0.05, cafe_value, str(subject_id),
                fontsize=9, ha='right', va='center',
                color=color, fontweight='bold')

        print(f"Subject {subject_id}: CAFE={cafe_value:.2f}%, CLASSROOM={classroom_value:.2f}%, "
              f"Group={'ADHD' if group == 0 else 'Control'}")

    # Configure axes
    ax.set_xlim(-0.3, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Cafe', 'Classroom'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Change by Environment\n(Each Subject in Different Color)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(40, 100)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'slope_graph_accuracy.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Slope graph saved to: {save_path}")

def comparison_bar_chart_accuracy(df: pd.DataFrame, comparison_type: str):
    # Create bar chart comparing accuracy by group, environment, or session
    
    if comparison_type.lower() == 'group':
        # Average per subject first, then group by ADHD/Control
        avg_per_subject = df.groupby(['Group', 'Subject'])['Accuracy'].mean().reset_index()
        grouped_data = avg_per_subject.groupby('Group').agg(
            mean=('Accuracy', 'mean'),
            std=('Accuracy', 'std'),
            count=('Subject', 'nunique')
        ).reset_index()

        # Map numeric group codes to labels
        grouped_data['Group'] = grouped_data['Group'].map({0: 'ADHD', 1: 'Control'})
        avg_per_subject['Group'] = avg_per_subject['Group'].map({0: 'ADHD', 1: 'Control'})
        category_col = 'Group'

        title = 'Average Accuracy:\nADHD vs Control'
        xlabel = 'Group'
        colors = ['#E74C3C', '#3498DB']

    elif comparison_type.lower() == 'environment':
        # Average per subject first, then group by environment
        avg_per_subject = df.groupby(['Environment', 'Subject'])['Accuracy'].mean().reset_index()
        grouped_data = avg_per_subject.groupby('Environment').agg(
            mean=('Accuracy', 'mean'),
            std=('Accuracy', 'std'),
            count=('Subject', 'nunique')
        ).reset_index()

        category_col = 'Environment'
        title = 'Average Accuracy:\nCafe vs Classroom'
        xlabel = 'Environment'
        colors = ['#F39C12', '#27AE60']

    elif comparison_type.lower() == 'session':
        # Average per subject first, then group by test/retest
        avg_per_subject = df.groupby(['Session', 'Subject'])['Accuracy'].mean().reset_index()
        grouped_data = avg_per_subject.groupby('Session').agg(
            mean=('Accuracy', 'mean'),
            std=('Accuracy', 'std'),
            count=('Subject', 'nunique')
        ).reset_index()

        # Map numeric session codes to labels
        grouped_data['Session'] = grouped_data['Session'].map({1: 'Test', 2: 'Retest'})
        avg_per_subject['Session'] = avg_per_subject['Session'].map({1: 'Test', 2: 'Retest'})
        category_col = 'Session'

        title = 'Average Accuracy:\nTest vs Retest'
        xlabel = 'Session'
        colors = ['#F39C12', '#27AE60']

    else:
        print(f"Error: comparison_type must be 'group', 'environment', or 'session'")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    x_positions = np.arange(len(grouped_data))

    # Create bars with error bars
    bars = ax.bar(x_positions, grouped_data['mean'],
                  color=colors[:len(grouped_data)], alpha=0.6, edgecolor='black',
                  linewidth=1.5, label='Mean')

    ax.errorbar(x_positions, grouped_data['mean'], yerr=grouped_data['std'],
                fmt='none', ecolor='black', capsize=8, capthick=2, alpha=0.7)

    # Add individual subject IDs with jitter
    for i, category in enumerate(grouped_data.iloc[:, 0]):
        subjects_in_category = avg_per_subject[avg_per_subject[category_col] == category]
        x_jitter = np.random.normal(i + 0.15, 0.04, size=len(subjects_in_category))

        for x_jittered, (_, row) in zip(x_jitter, subjects_in_category.iterrows()):
            ax.text(x_jittered, row['Accuracy'], str(row['Subject']),
                    fontsize=8, ha='center', va='center',
                    color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.7, linewidth=1),
                    zorder=3)

    # Add mean value and sample size labels above bars
    for i, (mean_val, count) in enumerate(zip(grouped_data['mean'], grouped_data['count'])):
        ax.text(i, mean_val + grouped_data['std'].iloc[i] + 2,
                f'{mean_val:.2f}%\n(n={int(count)})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Configure axes
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(grouped_data.iloc[:, 0], fontsize=13)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"comparison_accuracy_{comparison_type.lower()}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Graph saved as: {filename}")
    print("\nStatistics:")
    print(grouped_data.to_string(index=False))

def comparison_by_condition_accuracy(df: pd.DataFrame):
    # Create bar chart comparing accuracy across condition types (R, IR, M)
    
    # Average per subject first
    avg_per_subject = df.groupby(['Condition', 'Subject'])['Accuracy'].mean().reset_index()
    grouped_data = df.groupby('Condition')['Accuracy'].agg(['mean', 'std', 'count']).reset_index()

    # Order conditions as R, IR, M
    order = ['R', 'IR', 'M']
    grouped_data['Condition'] = pd.Categorical(grouped_data['Condition'],
                                               categories=order, ordered=True)
    grouped_data = grouped_data.sort_values('Condition')

    title = 'Average Accuracy:\nR vs IR vs M'
    xlabel = 'Condition Type'
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    fig, ax = plt.subplots(figsize=(12, 8))

    x_positions = np.arange(len(grouped_data))
    
    # Create bars with error bars
    bars = ax.bar(x_positions, grouped_data['mean'],
                  color=colors[:len(grouped_data)], alpha=0.6,
                  edgecolor='black', linewidth=1.5, label='Mean')

    ax.errorbar(x_positions, grouped_data['mean'], yerr=grouped_data['std'],
                fmt='none', ecolor='black', capsize=8, capthick=2, alpha=0.7)

    # Add individual subject IDs with jitter
    for i, condition in enumerate(grouped_data['Condition']):
        subjects_in_condition = avg_per_subject[avg_per_subject['Condition'] == condition]
        x_jitter = np.random.normal(i + 0.15, 0.04, size=len(subjects_in_condition))

        for x_jittered, (_, row) in zip(x_jitter, subjects_in_condition.iterrows()):
            ax.text(x_jittered, row['Accuracy'], str(row['Subject']),
                    fontsize=8, ha='center', va='center',
                    color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.7, linewidth=1),
                    zorder=3)

    # Add mean value and sample size labels above bars
    for i, (mean_val, count) in enumerate(zip(grouped_data['mean'], grouped_data['count'])):
        ax.text(i, mean_val + grouped_data['std'].iloc[i] + 2,
                f'{mean_val:.2f}%\n(n={int(count)})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Configure axes
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(grouped_data['Condition'], fontsize=13)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'comparison_accuracy_condition.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Graph saved!")
    print("\nStatistics:")
    print(grouped_data.to_string(index=False))

def histogram_accuracy_by_environment(df: pd.DataFrame):
    # Create histograms showing distribution of accuracy scores for cafe and classroom
    
    # Calculate average accuracy per subject for each environment
    cafe_avg_per_subject = df[df['Environment'] == 'CAFE'].groupby('Subject')['Accuracy'].mean()
    classroom_avg_per_subject = df[df['Environment'] == 'CLASSROOM'].groupby('Subject')['Accuracy'].mean()

    n_subjects_cafe = len(cafe_avg_per_subject)
    n_subjects_classroom = len(classroom_avg_per_subject)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bins = np.arange(0, 105, 5)  # 0, 5, 10, 15, ..., 100

    # CAFE histogram
    counts_cafe, bins_cafe, patches_cafe = axes[0].hist(
        cafe_avg_per_subject, bins=bins, color='#F39C12', alpha=0.7,
        edgecolor='black', linewidth=1.2
    )

    # Add count labels on top of each bar
    for count, patch in zip(counts_cafe, patches_cafe):
        height = patch.get_height()
        if height > 0:
            axes[0].text(patch.get_x() + patch.get_width() / 2, height + 0.3,
                         f'{int(height)}', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

    # Configure cafe histogram
    axes[0].set_xlabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Subjects', fontsize=13, fontweight='bold')
    axes[0].set_title(f'Average Accuracy per Subject - CAFE\n(n={n_subjects_cafe} subjects)',
                      fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_xlim(0, 100)

    # CLASSROOM histogram
    counts_classroom, bins_classroom, patches_classroom = axes[1].hist(
        classroom_avg_per_subject, bins=bins, color='#27AE60', alpha=0.7,
        edgecolor='black', linewidth=1.2
    )

    # Add count labels on top of each bar
    for count, patch in zip(counts_classroom, patches_classroom):
        height = patch.get_height()
        if height > 0:
            axes[1].text(patch.get_x() + patch.get_width() / 2, height + 0.3,
                         f'{int(height)}', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

    # Configure classroom histogram
    axes[1].set_xlabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Subjects', fontsize=13, fontweight='bold')
    axes[1].set_title(f'Average Accuracy per Subject - CLASSROOM\n(n={n_subjects_classroom} subjects)',
                      fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_xlim(0, 100)

    plt.tight_layout()

    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'histogram_accuracy_by_environment.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Histogram saved to: {save_path}")

def create_overall_comparison_chart(cafe_data, classroom_data, output_path):
    #cafe
    cafe_per_subject_session = cafe_data.groupby(['Subject', 'Session'])['Accuracy'].mean().reset_index()
    cafe_per_subject = cafe_per_subject_session.groupby('Subject')['Accuracy'].mean()

    cafe_overall_mean = cafe_data['Accuracy'].mean()
    cafe_overall_std = cafe_data['Accuracy'].std()
    cafe_cv = calculate_cv(cafe_overall_mean, cafe_overall_std)
    cafe_n = len(cafe_per_subject)

    # Classroom
    classroom_per_subject_session = classroom_data.groupby(['Subject', 'Session'])['Accuracy'].mean().reset_index()
    classroom_per_subject = classroom_per_subject_session.groupby('Subject')['Accuracy'].mean()

    classroom_overall_mean = classroom_data['Accuracy'].mean()
    classroom_overall_std = classroom_data['Accuracy'].std()
    classroom_cv = calculate_cv(classroom_overall_mean, classroom_overall_std)
    classroom_n = len(classroom_per_subject)

    environments = ['Cafe', 'Classroom']
    means = [cafe_overall_mean, classroom_overall_mean]
    stds = [cafe_overall_std, classroom_overall_std]
    cvs = [cafe_cv, classroom_cv]
    sample_sizes = [cafe_n, classroom_n]

    colors = ['#F39C12', '#27AE60']  # Orange for Cafe, Green for Classroom

    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(environments))
    bars = ax.bar(x, means, color=colors, width=0.6,
                  edgecolor='black', linewidth=2, alpha=0.8)

    ax.errorbar(x, means, yerr=stds, fmt='none',
                ecolor='black', capsize=15, linewidth=2.5, alpha=0.8)

    for i, (bar, mean, std, cv, n) in enumerate(zip(bars, means, stds, cvs, sample_sizes)):
        height = bar.get_height()

        ax.text(bar.get_x() + bar.get_width() / 2, height + std + 3,
                f'{mean:.2f}%',
                ha='center', va='bottom', fontsize=16, fontweight='bold',
                color=colors[i])

        ax.text(bar.get_x() + bar.get_width() / 2, height * 0.5,
                f'SD: {std:.2f}',
                ha='center', va='center', fontsize=11,
                color='darkred', fontweight='bold')

        ax.text(bar.get_x() + bar.get_width() / 2, height * 0.2,
                f'n = {n}',
                ha='center', va='center', fontsize=10,
                color='darkblue', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          alpha=0.8))

    overall_avg = sum(means) / len(means)
    ax.axhline(y=overall_avg, color='purple', linestyle=':',
               linewidth=2.5, alpha=0.6)

    ax.set_ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Comparison\nCafe vs Classroom',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(environments, fontsize=14, fontweight='bold')
    ax.set_ylim(0, min(100, max(means) + max(stds) + 10))

    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    difference = abs(cafe_overall_mean - classroom_overall_mean)
    better_env = "Cafe" if cafe_overall_mean > classroom_overall_mean else "Classroom"

    info_text = f'Difference: {difference:.2f}%\n'
    info_text += f'Higher accuracy: {better_env}\n'
    info_text += f'Total participants: {cafe_n}'

    ax.text(0.8, 0.01, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat',
                      edgecolor='black', linewidth=1.5, alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Overall accuracy comparison chart saved to: {output_path}")
    plt.show()

"""