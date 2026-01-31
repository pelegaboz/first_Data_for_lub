# ADHD Research Data Analysis System

## Overview
Comprehensive analysis system for data that I worked on proccesing in Golumbic lab. the data comparing participant performance and interest levels across two learning environments (Cafe vs Classroom). The system processes raw participant data, calculates accuracy and interest level metrics, performs statistical analyses, and generates visualizations.
---
Key Objectives:

Accuracy Analysis: Compare cognitive performance accuracy across environments and specific trial conditions.

Interest Level Assessment: Analyze subjective interest levels reported by participants on a scale of 1-7.

Environmental Comparison: Determine if auditory distractions (Cafe) versus a quiet setting (Classroom) differentially affect ADHD and Control groups.

Methodology: The system processes raw participant logs, calculates performance metrics, merges demographic data, performs statistical testing (paired t-tests), and generates visualizations.
## Folder & Module Structure
The project uses a relative path structure centered around a base directory defined in config.py.

Project_Root/
├── Input_Data/                     # Raw data sources
│   ├── All_Answers/                # Individual participant Excel/CSV files
│   ├── ADHD_group.xlsx             # Group classification (ADHD/Control)
│   ├── assignment_log.csv          # Master log of sessions and conditions
│   └── participants_info.xlsx      # Demographic and version info 
│
├── Output_Results/                 # Generated analysis products
│   ├── accuracy/                   # Processed accuracy Excel files
│   ├── interestlvl/                # Processed interest level Excel files
│   ├── Visualization/              # Generated plots and histograms
│   ├── statistical_tests/          # T-test summaries and significance tables
│   └── data_peleg.xlsx             # Final merged master dataset
│
├── Core Scripts/
│   ├── config.py                   # Central configuration (paths, constants)
│   ├── 1create_data.py             # Data preparation and splitting 
│   ├── 2accuracy.py                # Accuracy metric calculation
│   ├── 2interest_level.py          # Interest level metric calculation
│   ├── 3changes_in_data.py         # Data merging and master file creation
│   ├── 4visualization.py           # Graph generation
│   └── 5statistical_tests.py       # Statistical analysis
│
└── requirements.txt                # Python dependencies
## System Requirements

### Prerequisites
- Python 3.8 or higher
- Windows/Mac/Linux OS
- Approximately 500MB free disk space

### Required Python Libraries
```
pandas
numpy
matplotlib
scipy
openpyxl
pathlib
```

All dependencies are listed in `requirements.txt` and will be installed automatically.

---



Expected output:
```
All output directories created successfully
Configuration loaded successfully
All input files found
```

## Running the Analysis

### Complete Analysis Pipeline (Recommended)

Run the scripts in the following order:

Key Workflow Stages
The analysis pipeline must be executed in the following order to ensure data dependencies are met:

# Stage 1: Data Import & Split
Script: 1create_data.py

Action: Reads the main assignment_log and participants_info.
- Key Functions:
- `create_file_ADHD(from_df)` - Extracts ADHD classification data
- `test_files_in_both_environments(assignment_log_df)` - **NEW:** Validates file presence in both environments
- `create_file_assignment_log_per_envi(assignment_log_general, envi, output_path)` - Splits logs by environment

Outcome: Splits the log into two environment-specific files: assignment_log_cafe.csv and assignment_log_classroom.csv. This step is essential for the subsequent parallel processing of environments.

# Stage 2: Metric Calculation (Processing)
Two scripts run independently to process raw participant files from the All_Answers directory:

Accuracy: 2accuracy.py filters for valid trials (TrialID <= 30) and calculates the percentage of correct answers (SingleChoiceAccurate) for the Test and Retest sessions.
Interest level: 2interest_level.py extracts responses to the specific interest question (QuestionID == 7777) and converts boolean flags into a 1-7 numerical scale.
- Key Functions accuracy:
- `results_for_one_env(which_envi)` - Processes single environment
- `calculate_std_per_subject_from_raw_data(output_path)` - Calculates statistics
- `load_condition_mapping(condition_file_path)` - Maps conditions to subjects
- `identify_condition_for_trial(condition_str, trial_id)` - Determines trial condition
- `calculate_accuracy_for_trials(filtered_df, trial_ids)` - Computes accuracy

- Key Functions interst level:
- `results_For_one_envi(which_envi)` - Environment-specific analysis
- `calculate_std_per_subject_from_raw_data(output_path)` - Statistical calculations
- `interest_level_for_one_question(number_of_quastion, filltered_df)` - Extracts interest rating
- `calculate_avg_interest_for_condition(filtered_df, condition_str, condition_type)` - Averages by condition



# Stage 3: Data Integration
Script: 3changes_in_data.py

Action: Merges the output from Stage 2 into a single master file (data_peleg.xlsx).

Details:

Combines Cafe and Classroom data.

Adds demographic info (ADHD/Control) and experiment versions.

Calculates standard deviations (SD) and medians per subject.
- Key Functions:
- `check_and_create_main_data()` - Creates or validates master file
- `change_acc_per_condition(main_df, cafe_df, classroom_df)` - Updates accuracy data
- `change_interest_lvl_per_condition(main_df, interest_lvl_df)` - Updates interest data
- `change_SD_interest_lvl(df, from_df)` - Updates standard deviations
- `change_median_interest_lvl(df, from_df)` - Updates median values
- `change_ADHD(df, from_df)` - Updates group classifications
- `change_version(df, from_df)` - Updates experiment versions


# Stage 4: Visualization
Script: 4visualization.py

Action: Generates static plots saved to Output_Results/Visualization.

Outputs:

Slope Graphs: Individual trajectory of change between Cafe and Classroom.

Grouped Bar Charts: Comparisons by Group, Environment, and Session.

Histograms: Distribution of scores across subjects.


- Key Functions:
- `load_all_data()` - Loads all processed data
- `slope_graph_accuracy(df, save_path)` - Individual trajectories
- `comparison_acc_by_condition_and_environment(df, save_path)` - Grouped bar charts
- `histogram_acc_by_environment(df, save_path)` - Distribution plots
- `comparison_interest_by_group(df, save_path)` - Group comparisons
- `create_all_visualizations()` - Master execution function


# Stage 5: Statistical Analysis
Script: 5statistical_tests.py

Action: Performs Paired T-Tests to determine statistical significance.

Metrics: Calculates p-values and Cohen's d (effect size) for both Accuracy and Interest Levels, comparing Cafe vs. Classroom globally and per condition.
- Key Functions:
- `perform_paired_ttest(cafe_data, classroom_data, metric_name)` - Executes t-test
- `ttest_accuracy_by_environment()` - Accuracy comparisons
- `ttest_interest_by_environment()` - Interest comparisons
- `create_summary_table(accuracy_results, interest_results, output_path)` - Combined summary

- Statistical Methods:
- Paired t-tests (within-subjects design)
- Cohen's d effect size calculation
- Significance levels: p < 0.05 (*), p < 0.01 (**), p < 0.001 (***)


# Key Definitions & Configuration
The following parameters are defined in config.py and used throughout the system:

Group Classifications
0 (ADHD): Participants diagnosed with ADHD.

1 (Control): Control group participants.

Experimental Conditions
The experiment includes three distinct conditions that appear in varying sequences (e.g., R-IR-M):

R: Regular/Base condition.

IR: Irrelevant/Interference condition.

M: Music/Mixed condition.

Environments

Cafe: Simulates a noisy, public environment.


Classroom: Represents a standard, quiet academic setting.

Technical Parameters
MAX_TRIAL_ID = 30: The cutoff for valid trials included in accuracy calculations.

INTRESED_LEVEL_QUASTION = 7777: The unique Question ID used to query the user's subjective interest level.

BASE_DIR: Dynamically calculates the project root path to ensure the code runs on any machine without path errors.






## Output File Reference - for your convinience!!

### Accuracy Files
- `results_output_for_accuracy_Cafe.xlsx` - Cafe environment accuracy data
- `results_output_for_accuracy_Classroom.xlsx` - Classroom environment accuracy data
- `results_output_for_accuracy_all.xlsx` - Complete accuracy statistics (Mean, SD, Median)

### Interest Level Files
- `results_output_for_intrestLVL_cafe.xlsx` - Cafe environment interest data
- `results_output_for_intrestLVL_classroom.xlsx` - Classroom environment interest data
- `results_output_for_intrestLVL_all.xlsx` - Complete interest statistics

### Master Data File
- `data_peleg.xlsx` - Aggregated dataset combining all metrics

### Statistical Analysis
- `ttest_accuracy_results.xlsx` - Accuracy t-test results
- `ttest_interest_results.xlsx` - Interest t-test results
- `ttest_summary_all.xlsx` - Combined statistical summary

### Visualizations
**Accuracy Charts:**
- slope_graph_accuracy.png
- accuracy_by_environment.png
- accuracy_by_session.png
- accuracy_by_condition.png
- accuracy_histogram.png
- condition_and_environment.png

**Interest Level Charts:**
- interest_by_condition_and_environment.png
- interest_histogram.png
- interest_slope_bar.png
- interest_by_group.png
- interest_by_session.png

