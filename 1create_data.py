"""
this file creating the  data files:
adhd
assignment log for each environment
all the files neceseries for running the next scripts
I creating the files from: assignment_log, participants_info that given to me
"""
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).parent

CAFE= "Cafe"
CLASSROOM= "Classroom"

PARTICIPANTS_INFO = BASE_DIR / "Input_Data" / "participants_info.xlsx"
ASSIGNMENT_LOG_PATH = BASE_DIR / "Input_Data" / "assignment_log.csv"
OUTPUT_DATA = BASE_DIR / "Input_Data" / "ADHD_group.xlsx"
OUTPUT_PATH_FOR_CAFE = BASE_DIR / "Input_Data" / "assignment_log_cafe.csv"
OUTPUT_PATH_FOR_CLASSROOM = BASE_DIR / "Input_Data" / "assignment_log_classroom.csv"

def create_file_ADHD(from_df):
       
    #taking the columns of subject id and ADHD
    new_df = from_df[['Part_id', 'ADHD?']].copy()
    new_df = new_df.rename(columns={'ADHD?': 'which'})
    
    
    return new_df

def create_file_assignment_log_per_envi(assignment_log_general,envi,output_path):
    #creating the assignment_log that needed
    filtered_df_by_demend= assignment_log_general[assignment_log_general['Environment']== envi]
    filtered_df_by_demend.to_csv(output_path, index= False) #creating the file
    

    
def main():
    #creating ADHD file
    basic_df= pd.read_excel(PARTICIPANTS_INFO,sheet_name="Sheet1")
    df = create_file_ADHD(basic_df)
    df.to_excel(OUTPUT_DATA, index=False)
    #creating assignment log files
    assignment_log_df= pd.read_csv(ASSIGNMENT_LOG_PATH)
    create_file_assignment_log_per_envi(assignment_log_df,CAFE,OUTPUT_PATH_FOR_CAFE)  
    create_file_assignment_log_per_envi(assignment_log_df,CLASSROOM,OUTPUT_PATH_FOR_CLASSROOM)    
    print("created the files: assignment log cafe, assignment log classroom")
    
    
# Tests 
def test_adhd_file_columns_and_rename():
    sample = pd.DataFrame({
        "Part_id": [101, 102, 103],
        "ADHD?":   ["Yes", "No", "Yes"],
        "Age":     [25, 30, 22],
    })
    result = create_file_ADHD(sample)
    assert list(result.columns) == ["Part_id", "which"]
    assert list(result["which"]) == ["Yes", "No", "Yes"]


def test_assignment_log_filters_correctly(tmp_path):
    sample = pd.DataFrame({
        "Assignment_id": [1, 2, 3, 4],
        "Environment":   ["Cafe", "Classroom", "Cafe", "Classroom"],
        "Student_id":    [101, 102, 103, 104],
    })
    out = tmp_path / "cafe.csv"
    create_file_assignment_log_per_envi(sample, "Cafe", out)
    result = pd.read_csv(out)
    assert (result["Environment"] == "Cafe").all()
    assert len(result) == 2


def test_original_df_not_modified():
    sample = pd.DataFrame({
        "Part_id": [101, 102, 103],
        "ADHD?":   ["Yes", "No", "Yes"],
        "Age":     [25, 30, 22],
    })
    original_copy = sample.copy()
    create_file_ADHD(sample)
    pd.testing.assert_frame_equal(sample, original_copy)


if __name__== "__main__":
    main()
