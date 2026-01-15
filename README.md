# ADHD Research Data Analysis System

## Overview
Comprehensive analysis system for ADHD research data, comparing participant performance and interest levels across two learning environments (Cafe vs Classroom).

---

## System Requirements

### Prerequisites
- Python 3.8 or higher
- Windows OS
- Approximately 500MB free disk space

### Required Python Libraries
All dependencies are listed in `requirements.txt` and will be installed automatically.

---

## Installation Guide

### Step 1: Verify Python Installation
Open Command Prompt (CMD) and run:
```bash
python --version
```
Expected output: Python 3.8.x or higher

If Python is not installed, download from: https://www.python.org/downloads/

### Step 2: Install Required Libraries
Navigate to the project directory and run:
```bash
cd path\to\Analysis_Code
pip install -r requirements.txt
```
Installation time: Approximately 2-3 minutes

### Step 3: Configure File Paths
1. Open `config.py` in a text editor
2. Locate the line:
   ```python
   BASE_PATH = r"C:\Users\YourName\Desktop\Analysis_Code"
   ```
3. Replace `YourName` with your actual Windows username
4. Save the file (Ctrl+S)

### Step 4: Verify Configuration
Run the configuration verification:
```bash
python config.py
```

Expected output:
```
✓ All output directories created successfully
✓ Base Path: [your path]
✓ Input Data: [your path]
✓ All Answers: [your path]
```

If you see any "✗" marks, the path configuration is incorrect.

---

## Running the Analysis

### Method 1: Automated Execution (Recommended)

#### Option A: Using Batch File
Double-click on `START.bat` to run all analyses automatically.

#### Option B: Using Python Command
```bash
python main.py
```

This will execute all three analysis modules in sequence:
1. Accuracy Analysis (2-3 minutes)
2. Interest Level Analysis (2-3 minutes)
3. Data Updates (1 minute)

Total execution time: 5-7 minutes

### Method 2: Individual Module Execution

#### Accuracy Analysis
```bash
python accuracy.py
```

**Purpose:** Calculates accuracy percentages for participant responses

**Output Files:**
- `results_output_for_accuracy_Cafe.xlsx`
- `results_output_for_accuracy_Classroom.xlsx`
- Charts in `Output_Results/Accuracy_Analysis/`

**Execution Time:** 2-3 minutes

---

#### Interest Level Analysis
```bash
python interest_level.py
```

**Purpose:** Analyzes participant interest levels across different conditions

**Output Files:**
- `results_output_for_intrest_level_Cafe.xlsx`
- `results_output_for_intrest_level_Classroom.xlsx`
- `statistics_by_environment_and_condition.xlsx`
- Charts in `Output_Results/Interest_Level/`

**Execution Time:** 2-3 minutes

---

#### Data Updates
```bash
python changes_in_data.py
```

**Purpose:** Updates and synchronizes data across different files

**Output Files:**
- Updates `data_peleg.xlsx` with new analysis results

**Execution Time:** 1 minute

---

## Directory Structure

```
Analysis_Code/
│
├── Core Scripts
│   ├── main.py                     # Master execution script
│   ├── accuracy.py                 # Accuracy analysis module
│   ├── interest_level.py           # Interest level analysis module
│   └── changes_in_data.py          # Data synchronization module
│
├── Configuration Files
│   ├── config.py                   # Path and constant definitions
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # This documentation
│
├── Utilities
│   └── START.bat                   # Windows batch launcher
│
├── Input_Data/                     # Input data directory
│   ├── All_Answers/                # Participant response files
│   ├── ADHD_group.xlsx             # Group classification data
│   ├── assignment_log.csv          # Complete assignment log
│   ├── assignment_log_cafe.csv     # Cafe environment assignments
│   ├── assignment_log_classroom.csv# Classroom environment assignments
│   └── data_peleg.xlsx             # Master data file
│
└── Output_Results/                 # Analysis output directory
    ├── Accuracy_Analysis/          # Accuracy charts and data
    └── Interest_Level/             # Interest level charts and data
   └── data_peleg.xlsx             # Master data file
```

---

## Module Documentation

### accuracy.py

**Primary Functions:**
- `results_for_all_files()` - Processes all participant files
- `results_for_one_env(environment)` - Analyzes single environment data
- `comparison_bar_chart_accuracy(df, comparison_type)` - Generates comparison charts
- `slope_bar_accuracy(df)` - Creates individual subject trajectory plots
- `independent_ttest_adhd_vs_control(df, environment)` - Statistical comparison

**Output Columns:**
- `Subject` - Participant ID number
- `Session` - Test session (1=Test, 2=Retest)
- `Environment` - Testing environment (CAFE/CLASSROOM)
- `Group` - Participant group (0=ADHD, 1=Control)
- `Condition` - Trial condition (R/IR/M)
- `Accuracy` - Percentage accuracy score

---

### interest_level.py

**Primary Functions:**
- `results_For_one_envi(environment)` - Calculates mean interest levels
- `calculate_std_per_subject_from_raw_data(output_path)` - Computes statistical measures
- `comparison_by_condition(df)` - Generates condition comparisons
- `slop_bar(df)` - Creates trajectory visualizations
- `histogram_interest_by_environment(df)` - Distribution analysis

**Output Columns:**
- `Subject` - Participant ID number
- `Session` - Test session
- `Environment` - Testing environment
- `Group` - Participant group
- `Condition` - Trial condition (R/IR/M/Overall)
- `Interest_Level` - Mean interest level score
- `SD` - Standard deviation
- `Median` - Median interest level

---

### changes_in_data.py

**Primary Functions:**
- `change_acc_per_condition(main_df, cafe_df, classroom_df)` - Updates accuracy by condition
- `change_intrest_lvl_per_condition(main_df, intrest_df)` - Updates interest levels
- `change_SD_intrest_lvl(df, from_df)` - Updates standard deviations
- `change_median_intrest_lvl(df, from_df)` - Updates median values

**Usage:**
Execute only when master data file requires updates from analysis results.

---

## Data Requirements

### Input File Format

**Participant Response Files (All_Answers/):**
Required columns:
- `TrialID` - Trial identification number (1-30)
- `QuestionID` - Question identification number
- `SingleChoiceAccurate` - Response accuracy (Boolean)
- `ReportedAnswer2-8` - Interest level responses (Boolean)

**Assignment Log Files:**
Required columns:
- `Subject` - Participant ID
- `Session` - Session number
- `Condition` - Condition sequence string

**Group Classification File:**
Required columns:
- `Part_id` - Participant ID
- `which` - Classification indicator ('V' for valid entries)

---

## Troubleshooting

### Common Issues and Solutions

**Issue: "No module named 'pandas'"**
Solution:
```bash
pip install pandas numpy matplotlib scipy openpyxl xlrd
```

**Issue: "File not found" or "Path does not exist"**
Solution:
1. Open `config.py`
2. Verify `BASE_PATH` is correct
3. Run `python config.py` to verify all paths

**Issue: No output files generated**
Solution:
1. Verify files exist in `Input_Data/All_Answers/`
2. Check input file format matches requirements
3. Review error messages in console output

**Issue: Script appears frozen**
Solution:
- Processing large datasets takes time (2-5 minutes is normal)
- If exceeds 10 minutes, press Ctrl+C and check for errors

**Issue: Missing visualizations**
Solution:
1. Verify sufficient data exists (minimum 5 participants)
2. Check for PNG files in output directories
3. Review console for matplotlib errors

**Issue: Permission denied when writing files**
Solution:
1. Close any open Excel files in output directories
2. Run Command Prompt as Administrator
3. Verify write permissions on output directories

---

## Output File Locations

All analysis outputs are saved in the `Output_Results/` directory:

**Accuracy Analysis:**
- Excel tables: `Output_Results/`
- Charts: `Output_Results/Accuracy_Analysis/`

**Interest Level Analysis:**
- Excel tables: `Output_Results/`
- Charts: `Output_Results/Interest_Level/`
- Statistics: `Output_Results/Interest_Level/statistics_by_environment_and_condition.xlsx`

---

## Performance Notes

**Typical Execution Times:**
- Accuracy analysis: 2-3 minutes
- Interest level analysis: 2-3 minutes
- Data updates: 1 minute
- Complete automated run: 5-7 minutes

**Resource Usage:**
- Memory: ~500MB RAM
- Disk space: ~100MB for outputs
- CPU: Standard processing (single-threaded)

---

## Version Information

**System Version:** 1.0
**Last Updated:** January 2026
**Python Compatibility:** 3.8+
**Operating System:** Windows 10/11

---

## Contact Information

**Developer:** [Your Name]
**Email:** [Your Email]
**Date Created:** [Current Date]

---

## Important Notes

1. **Data Backup:** Always maintain backup copies of original input data
2. **File Integrity:** Do not modify files in `Input_Data/` during analysis
3. **Output Management:** Output files can be safely deleted; they will be regenerated
4. **Execution Order:** For first-time analysis, run `main.py` to generate all outputs
5. **Path Configuration:** Verify path configuration before first execution

---

## Pre-Delivery Checklist

- [ ] Python 3.8+ installed and verified
- [ ] All dependencies from requirements.txt installed
- [ ] config.py updated with correct paths
- [ ] Test run of `python config.py` successful
- [ ] Test run of `python accuracy.py` completed
- [ ] Test run of `python interest_level.py` completed
- [ ] All input data files present in Input_Data/
- [ ] Output directories created successfully

---

**End of Documentation**