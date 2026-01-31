"""
Configuration File for Analysis Scripts
Contains all constants, paths, and settings used across the project

IMPORTANT: All paths are now RELATIVE to the project directory.
This makes the code portable - anyone can run it from any location!
"""

from pathlib import Path
import os

# BASE DIRECTORY - The foundation for all relative paths

# Get the directory where THIS config file is located
# This will be the base for all other paths
BASE_DIR = Path(__file__).parent

# Example: If config.py is in C:\Projects\Analysis\
# Then BASE_DIR = C:\Projects\Analysis\
# If someone else has it in D:\MyWork\Study\
# Then BASE_DIR = D:\MyWork\Study\
# The code works the same in both cases!


# CONSTANTS - Group and Environment Identifiers

# Group identifiers
ADHD = 0
Control = 1

# Environment identifiers
CAFE = "cafe"
CLASSROOM = "classroom"

# Session identifiers
NUMBER_TEST = 1
NUMBER_RETEST = 2

# Trial settings
MAX_TRIAL_ID = 30

# Question identifiers
INTRESED_LEVEL_QUASTION = 7777  # Question ID for interest level questions


# INPUT PATHS - Data Sources (NOW RELATIVE!)

# Environment assignment logs
# OLD: r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\assignment_log_cafe.csv"
# NEW: Works from any location!
CAFE_DATA_PATH = BASE_DIR / "Input_Data" / "assignment_log_cafe.csv"
CLASSROOM_DATA_PATH = BASE_DIR / "Input_Data" / "assignment_log_classroom.csv"

# ADHD group classification
DATA_PATH_ADHD = BASE_DIR / "Input_Data" / "ADHD_group.xlsx"

# Participants information
PARTICIPANTS_INFO = BASE_DIR / "Input_Data" / "participants_info.xlsx"

# Raw data directory (contains all subject files)
DATA_PATH = BASE_DIR / "Input_Data" / "All_Answers"

# Combined condition log
CONDITION_FILE_PATH = BASE_DIR / "Input_Data" / "assignment_log.csv"


# OUTPUT PATHS - Results and Visualizations (NOW RELATIVE!)

# Main output directory
OUTPUT_DIR = BASE_DIR / "Output_Results"
OUTPUT_DIR_VISUALIZATION = BASE_DIR / "Output_Results" / "Visualization"

# Accuracy results
ACCURACY_OUTPUT_DIR = BASE_DIR / "Output_Results" / "accuracy"
ACCURACY_CAFE_OUTPUT = BASE_DIR / "Output_Results" / "accuracy" / "results_output_for_accuracy_Cafe.xlsx"
ACCURACY_CLASSROOM_OUTPUT = BASE_DIR / "Output_Results" / "accuracy" / "results_output_for_accuracy_Classroom.xlsx"
ACCURACY_ALL_OUTPUT = BASE_DIR / "Output_Results" / "accuracy" / "results_output_for_accuracy_all.xlsx"

# Interest level results
INTEREST_OUTPUT_DIR = BASE_DIR / "Output_Results" / "interestlvl"
INTEREST_CAFE_OUTPUT = BASE_DIR / "Output_Results" / "interestlvl" / "results_output_for_intrestLVL_cafe.xlsx"
INTEREST_CLASSROOM_OUTPUT = BASE_DIR / "Output_Results" / "interestlvl" / "results_output_for_intrestLVL_classroom.xlsx"
INTEREST_ALL_OUTPUT = BASE_DIR / "Output_Results" / "interestlvl" / "results_output_for_intrestLVL_all.xlsx"

# Main combined data file
MAIN_DATA = BASE_DIR / "Output_Results" / "data_peleg.xlsx"


# ANALYSIS SETTINGS (Optional - can be customized)

# Statistical significance levels
ALPHA_LEVEL = 0.05
HIGHLY_SIGNIFICANT = 0.001
VERY_SIGNIFICANT = 0.01

# Visualization settings
FIGURE_DPI = 300
FIGURE_SIZE_LARGE = (12, 8)
FIGURE_SIZE_MEDIUM = (10, 6)

# Color schemes
COLORS_GROUP = ['#E74C3C', '#3498DB']  # ADHD (red), Control (blue)
COLORS_ENVIRONMENT = ['#F39C12', '#27AE60']  # Cafe (orange), Classroom (green)
COLORS_CONDITION = ['#E74C3C', '#3498DB', '#2ECC71']  # R, IR, M


# DATA VALIDATION SETTINGS

# Expected column names in data files
EXPECTED_COLUMNS_ACCURACY = ['TrialID', 'QuestionID', 'SingleChoiceAccurate']
EXPECTED_COLUMNS_INTEREST = ['TrialID', 'QuestionID', 'ReportedAnswer2', 
                              'ReportedAnswer3', 'ReportedAnswer4', 'ReportedAnswer5',
                              'ReportedAnswer6', 'ReportedAnswer7', 'ReportedAnswer8']

# Expected columns in metadata files
EXPECTED_METADATA_COLUMNS = ['Subject', 'Session', 'Condition']


# HELPER FUNCTION

def create_output_directories():
    """
    Create all output directories if they don't exist.
    This is safe to run multiple times - it won't overwrite existing data.
    """
    directories = [
        OUTPUT_DIR,
        OUTPUT_DIR_VISUALIZATION,
        ACCURACY_OUTPUT_DIR,
        INTEREST_OUTPUT_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        # parents=True: creates parent directories if needed
        # exist_ok=True: doesn't raise error if directory already exists
    
    print(" All output directories created/verified")
    print(f" Base directory: {BASE_DIR}")


# PATH CONVERSION HELPER (for debugging)

def print_all_paths():
    """
    Print all configured paths - useful for debugging.
    Shows both the relative Path objects and their absolute versions.
    """
    print("\n" + "="*80)
    print("CONFIGURED PATHS")
    print("="*80)
    
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Absolute: {BASE_DIR.resolve()}")
    
    print("\n--- INPUT PATHS ---")
    input_paths = {
        "CAFE_DATA_PATH": CAFE_DATA_PATH,
        "CLASSROOM_DATA_PATH": CLASSROOM_DATA_PATH,
        "DATA_PATH_ADHD": DATA_PATH_ADHD,
        "PARTICIPANTS_INFO": PARTICIPANTS_INFO,
        "DATA_PATH": DATA_PATH,
        "CONDITION_FILE_PATH": CONDITION_FILE_PATH
    }
    
    for name, path in input_paths.items():
        exists = " EXISTS" if path.exists() else "✗ NOT FOUND"
        print(f"{name:25} {exists}")
        print(f"  {path}")
    
    print("\n--- OUTPUT PATHS ---")
    output_paths = {
        "OUTPUT_DIR": OUTPUT_DIR,
        "ACCURACY_OUTPUT_DIR": ACCURACY_OUTPUT_DIR,
        "INTEREST_OUTPUT_DIR": INTEREST_OUTPUT_DIR,
        "OUTPUT_DIR_VISUALIZATION": OUTPUT_DIR_VISUALIZATION
    }
    
    for name, path in output_paths.items():
        exists = " EXISTS" if path.exists() else "✗ NOT CREATED YET"
        print(f"{name:30} {exists}")
        print(f"  → {path}")
    
    print("\n" + "="*80)


# CONFIGURATION VALIDATION

if __name__ == "__main__":
    """
    Test configuration when running this file directly.
    Run: python config.py
    """
    print("CONFIGURATION VALIDATION")
    
    print(f"\n Project Base Directory:")
    print(f"   {BASE_DIR}")
    print(f"   (Absolute: {BASE_DIR.resolve()})")
    
    # Check if input files exist
    print("\n Checking INPUT files:")
    input_files = {
        "Cafe assignment log": CAFE_DATA_PATH,
        "Classroom assignment log": CLASSROOM_DATA_PATH,
        "ADHD group file": DATA_PATH_ADHD,
        "Participants info": PARTICIPANTS_INFO,
        "Raw data directory": DATA_PATH,
        "Condition file": CONDITION_FILE_PATH
    }
    
    all_exist = True
    for name, path in input_files.items():
        exists = path.exists()
        all_exist = all_exist and exists
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
        if not exists:
            print(f"     Missing: {path}")
    
    # Create output directories
    print("\n Creating OUTPUT directories:")
    create_output_directories()
    
    # Summary
    print("\n" + "="*80)
    if all_exist:
        print("Configuration loaded successfully!")
        print("All input files found!")
    else:
        print(" Configuration loaded, but some input files are missing.")
        print("  Make sure to place your data files in the Input_Data folder.")
