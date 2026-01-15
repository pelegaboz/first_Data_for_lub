"""
runing first- creating ADHD file
"""
import pandas as pd

PARTICIPANTS_INFO= r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\participants_info.xlsx"
OUTPUT_DATA= r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\ADHD_group.xlsx"



def create_file_ADHD(from_df):
       
    #taking the columns of subject id and ADHD
    new_df = from_df[['Part_id', 'ADHD?']].copy()
    new_df = new_df.rename(columns={'ADHD?': 'which'})
    
    
    return new_df

def main():
    basic_df= pd.read_excel(PARTICIPANTS_INFO,sheet_name="Sheet1")
    df = create_file_ADHD(basic_df)
    df.to_excel(OUTPUT_DATA, index=False)
    
if __name__== "__main__":
    main()
