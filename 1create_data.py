"""
this file creating the  data files:
adhd
assignment log for each environment
all the files neceseries for running the next scripts
"""
import pandas as pd

CAFE= "Cafe"
CLASSROOM= "Classroom"


PARTICIPANTS_INFO= r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\participants_info.xlsx"
ASSIGNMENT_LOG_PATH= r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\assignment_log.csv"
OUTPUT_DATA= r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\ADHD_group.xlsx"
OUTPUT_PATH_FOR_CAFE=r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\assignment_log_cafe.csv"
OUTPUT_PATH_FOR_CLASSROOM=r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\assignment_log_classroom.csv"



def create_file_ADHD(from_df):
       
    #taking the columns of subject id and ADHD
    new_df = from_df[['Part_id', 'ADHD?']].copy()
    new_df = new_df.rename(columns={'ADHD?': 'which'})
    
    
    return new_df

def create_file_assignment_log(assignment_log_general,envi,output_path):
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
    create_file_assignment_log(assignment_log_df,CAFE,OUTPUT_PATH_FOR_CAFE)  
    create_file_assignment_log(assignment_log_df,CLASSROOM,OUTPUT_PATH_FOR_CLASSROOM)    
    print("creating the files: assignment log cafe, assignment log classroom")
if __name__== "__main__":
    main()
