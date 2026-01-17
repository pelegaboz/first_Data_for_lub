"""
Configuration File for Analysis Scripts
Contains all constants, paths, and settings used across the project
"""

# ============================================================================
# CONSTANTS - Group and Environment Identifiers
# ============================================================================

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


# ============================================================================
# INPUT PATHS - Data Sources
# ============================================================================

# Environment assignment logs
CAFE_DATA_PATH = r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\assignment_log_cafe.csv"
CLASSROOM_DATA_PATH = r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\assignment_log_classroom.csv"

# ADHD group classification
DATA_PATH_ADHD = r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\ADHD_group.xlsx"

# Raw data directory (contains all subject files)
DATA_PATH = r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\All_Answers"

# Combined condition log (if needed)
CONDITION_FILE_PATH = r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\assignment_log.csv"


# ============================================================================
# OUTPUT PATHS - Results and Visualizations
# ============================================================================

# Main output directory
OUTPUT_DIR = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results"
OUTPUT_DIR_VISUALIZATION = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\Visualization"

# Accuracy results
ACCURACY_OUTPUT_DIR = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy"
ACCURACY_CAFE_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy\results_output_for_accuracy_Cafe.xlsx"
ACCURACY_CLASSROOM_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy\results_output_for_accuracy_Classroom.xlsx"
ACCURACY_ALL_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy\results_output_for_accuracy_all.xlsx"

# Interest level results
INTEREST_OUTPUT_DIR = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl"
INTEREST_CAFE_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl\results_output_for_intrestLVL_cafe.xlsx"
INTEREST_CLASSROOM_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl\results_output_for_intrestLVL_classroom.xlsx"
INTEREST_ALL_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl\results_output_for_intrestLVL_all.xlsx"
MAIN_DATA= r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\data_peleg.xlsx"


# ============================================================================
# ANALYSIS SETTINGS (Optional - can be customized)
# ============================================================================

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


# ============================================================================
# DATA VALIDATION SETTINGS
# ============================================================================

# Expected column names in data files
EXPECTED_COLUMNS_ACCURACY = ['TrialID', 'QuestionID', 'SingleChoiceAccurate']
EXPECTED_COLUMNS_INTEREST = ['TrialID', 'QuestionID', 'ReportedAnswer2', 
                              'ReportedAnswer3', 'ReportedAnswer4', 'ReportedAnswer5',
                              'ReportedAnswer6', 'ReportedAnswer7', 'ReportedAnswer8']

# Expected columns in metadata files
EXPECTED_METADATA_COLUMNS = ['Subject', 'Session', 'Condition']


# ============================================================================
# HELPER FUNCTION (Optional)
# ============================================================================

def create_output_directories():
    """Create all output directories if they don't exist"""
    import os
    directories = [
        OUTPUT_DIR,
        ACCURACY_OUTPUT_DIR,
        INTEREST_OUTPUT_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ All output directories created/verified")


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

if __name__ == "__main__":
    """Test configuration when running this file directly"""
    import os
    
    print("="*80)
    print("CONFIGURATION VALIDATION")
    print("="*80)
    
    # Check if input files exist
    print("\nChecking input files:")
    input_files = {
        "Cafe assignment log": CAFE_DATA_PATH,
        "Classroom assignment log": CLASSROOM_DATA_PATH,
        "ADHD group file": DATA_PATH_ADHD,
        "Raw data directory": DATA_PATH
    }
    
    for name, path in input_files.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
    
    # Create output directories
    print("\nCreating output directories:")
    create_output_directories()
    
    print("\n" + "="*80)
    print("Configuration loaded successfully!")
    print("="*80)