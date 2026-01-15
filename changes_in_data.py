import pandas as pd
import numpy as np
import os

# CONFIGURATION - Import from config file
MAIN_DATA = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\data_peleg.xlsx"
DATA_PATH_ADHD = r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\ADHD_group.xlsx"
PARTICIPANTS_INFO= r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\participants_info.xlsx"
try:
    from config import *
    print("Configuration loaded successfully")
except ImportError:
    print("Warning: config.py not found, using default paths")
    # Constants
    ADHD = 0
    Control = 1
    CAFE = "cafe"
    CLASSROOM = "classroom"
    NUMBER_TEST = 1
    NUMBER_RETEST = 2
    MAX_TRIAL_ID = 30
    # Paths
    MAIN_DATA = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\data_peleg.xlsx"
    DATA_PATH_ADHD = r"C:\Users\peleg\Desktop\Analysis_Code\Input_Data\ADHD_group.xlsx"
    ACCURACY_CAFE_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy\results_output_for_accuracy_Cafe.xlsx"
    ACCURACY_CLASSROOM_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy\results_output_for_accuracy_Classroom.xlsx"
    ACCURACY_ALL_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\accuracy\results_output_for_accuracy_all.xlsx"
    INTEREST_CAFE_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl\results_output_for_intrestLVL_cafe.xlsx"
    INTEREST_CLASSROOM_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl\results_output_for_intrestLVL_classroom.xlsx"
    INTEREST_ALL_OUTPUT = r"C:\Users\peleg\Desktop\Analysis_Code\Output_Results\interestlvl\results_output_for_intrestLVL_all.xlsx"
    

def change_envi(df):
    """
    Normalize environment names
    """
    df['Environment'] = df['Environment'].replace({
       "CLASSROOM": "Classroom",
       "CAFE": "Cafe"
    })
    return df

def create_main_data_file():
    """
   create the final sheet from accuracy and interest level
    """
    
    print("CREATING MAIN_DATA FILE FROM SCRATCH")
    print("="*80)
    
    # 1. reading accuracy files
    print("\n1. Loading accuracy data...")
    cafe_acc = pd.read_excel(ACCURACY_CAFE_OUTPUT)
    classroom_acc = pd.read_excel(ACCURACY_CLASSROOM_OUTPUT)
    
    all_acc = pd.concat([cafe_acc, classroom_acc], ignore_index=True)
    print(f"   ✓ Total accuracy records: {len(all_acc)}")
    
    # 2. reading interest level files
    print("\n2. Loading interest level data...")
    cafe_int = pd.read_excel(INTEREST_CAFE_OUTPUT)
    classroom_int = pd.read_excel(INTEREST_CLASSROOM_OUTPUT)
    
    all_int = pd.concat([cafe_int, classroom_int], ignore_index=True)
    print(f"   ✓ Total interest records: {len(all_int)}")
    
    # 3.create dataframe
    print("\n3. Creating main structure...")
    
    # Subject + Environment + Session
    main_df = all_acc[['Subject', 'Environment', 'Session', 'Group']].drop_duplicates()
    
    # changing the name of the columns
    main_df = main_df.rename(columns={
        'Subject': 'subject_id',
        'Environment': 'environment',
        'Session': 'session',
        'Group': 'group'
    })
    
    # environment= small leters
    main_df['environment'] = main_df['environment'].str.lower()
    
    print(f"   ✓ Unique participants: {len(main_df)}")
    
    # 4. all accuracy
    print("\n4. Calculating overall accuracy...")
    acc_overall = all_acc.groupby(['Subject', 'Environment'])['Accuracy'].mean().reset_index()
    acc_overall.columns = ['subject_id', 'environment', 'q_acc_all']
    acc_overall['environment'] = acc_overall['environment'].str.lower()
    acc_overall['q_acc_all'] = acc_overall['q_acc_all'] / 100  
    
    # 5.avarege interest level for each subject
    print("\n5. Calculating overall interest level...")
    int_overall = all_int.groupby(['Subject', 'Environment'])['Interest_Level'].mean().reset_index()
    int_overall.columns = ['subject_id', 'environment', 'interest_mean_all']
    int_overall['environment'] = int_overall['environment'].str.lower()
    
    # 6. marging all to main_df
    print("\n6. Merging all data...")
    main_df = main_df.merge(
        acc_overall[['subject_id', 'environment', 'q_acc_all']],
        on=['subject_id', 'environment'],
        how='left'
    )
    
    main_df = main_df.merge(
        int_overall[['subject_id', 'environment', 'interest_mean_all']],
        on=['subject_id', 'environment'],
        how='left'
    )
    
    # 7.all the demended columns
    print("\n7. Adding placeholder columns...")
    columns_to_add = [
        'q_acc_r', 'q_acc_ir', 'q_acc_m',
        'interest_mean_r', 'interest_mean_ir', 'interest_mean_m',
        'interest_sd_all', 'interest_sd_r', 'interest_sd_ir', 'interest_sd_m',
        'interest_median_all'
    ]
    
    for col in columns_to_add:
        if col not in main_df.columns:
            main_df[col] = None
    
    # 8. saving
    print("\n8. Saving MAIN_DATA file...")
    main_df.to_excel(MAIN_DATA, index=False)
    
    print(f"✓ MAIN_DATA file created successfully!")
    print(f"✓ File saved to: {MAIN_DATA}")
    print(f"✓ Total rows: {len(main_df)}")
    
    return main_df

def check_and_create_main_data():
    """
   if MAIN_DATA dosent exist- create it
    """
    # check if exist
    if not os.path.exists(MAIN_DATA):
        print(f" MAIN_DATA file not found at: {MAIN_DATA}")
        print("Creating new MAIN_DATA file...")
        return create_main_data_file()
    
    # check if empty
    try:
        main_df = pd.read_excel(MAIN_DATA)
        if len(main_df) == 0 or main_df.empty:
            print(f" MAIN_DATA file is empty!")
            print("Recreating MAIN_DATA file...")
            return create_main_data_file()
        else:
            print(f" MAIN_DATA file exists with {len(main_df)} rows")
            return main_df
    except Exception as e:
        print(f" Error reading MAIN_DATA: {e}")
        print("Recreating MAIN_DATA file...")
        return create_main_data_file()

def change_version(df,from_df):
       
    #update the version of each subject (RIRM)
    df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce').astype(int)
    source_clean = from_df.copy()
    source_clean.columns = ['Part_id', 'Version'] 
    source_clean = source_clean.drop_duplicates(subset=['Part_id'], keep='first') #make sure to take the first
    source_clean["Part_id"] = source_clean["Part_id"].astype(str).str.strip()
    mapper= source_clean.set_index("Part_id")["Version"]#make part_id index
    lookup_key = df['subject_id'].astype(str).str.strip()
    
    if 'exp_version' in df.columns:
            df['exp_version'] = lookup_key.map(mapper).fillna(df['exp_version'])
    else:
        #if dosent exist create
        df['exp_version'] = lookup_key.map(mapper)
    df['exp_version'] = df['exp_version'].astype(str).str[-4:]

    return df 

def change_ADHD(df, from_df):
    """
    Update the subject group information safely.
    Ensures IDs are treated as strings and strips whitespace.
    """
    source_clean = from_df.copy()
    source_clean["Part_id"] = source_clean["Part_id"].astype(str).str.strip()
    lookup_key = df['subject_id'].astype(str).str.strip()

    mapper = source_clean.set_index("Part_id")["which"]
    if 'adhd' in df.columns:
        df['adhd'] = lookup_key.map(mapper).fillna(df['adhd'])
    else:
        df['adhd'] = lookup_key.map(mapper)
    
    df['adhd'] = lookup_key.map(mapper).fillna(df['adhd'])
    df['adhd']= np.where(df['adhd']=='X',0, 1) #changing X\V to 0\1
    
    
    return df    

def change_q_acc(df,from_df):
#calculate the mean of accuracy
    df.set_index(['subject_id', 'environment'], inplace=True)
    from_df.set_index(['Subject', 'Environment'],inplace=True)
    df['q_acc_all'].update(from_df['Overall_Accuracy'])
    df.reset_index( inplace=True)
    df['q_acc_all']= df['q_acc_all']/100

    return df

def change_interest_mean_all(df, from_df):
    # calculate the mean of interest_lvl
    
    # only Overall
    from_df_copy = change_envi(from_df.copy())
    overall_mean = from_df_copy[from_df_copy['Condition'] == 'Overall'].copy()
    
    # changing names of columns
    overall_mean = overall_mean.rename(columns={
        'Subject': 'subject_id',
        'Environment': 'environment',
        'Mean': 'interest_mean_all'
    })
    
    # normalization
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['environment'] = df['environment'].astype(str).str.strip().str.lower()
    overall_mean['subject_id'] = overall_mean['subject_id'].astype(str).str.strip()
    overall_mean['environment'] = overall_mean['environment'].astype(str).str.strip().str.lower()
    
    # deleting exist column
    if 'interest_mean_all' in df.columns:
        df.drop(columns=['interest_mean_all'], inplace=True)
    
    # merge
    df = df.merge(
        overall_mean[['subject_id', 'environment', 'interest_mean_all']],
        on=['subject_id', 'environment'],
        how='left'
    )
    
    print(f"interest_mean_all: {df['interest_mean_all'].notna().sum()} non-null values")
    
    return df    # calculate the mean of interest_lvl


def change_interest_lvl_per_condition(main_df,interest_lvl_df):
    
    # pivot table
    pivot_df = interest_lvl_df.pivot_table(
        index=['Subject', 'Environment'], 
        columns='Condition',
        values='Interest_Level' 
    ).reset_index()
    
    # changing names
    rename_map = {
        'Subject': 'subject_id',
        'Environment': 'environment',
        'R': 'interest_mean_r',
        'IR': 'interest_mean_ir',
        'M': 'interest_mean_m'
    }
    pivot_df.rename(columns=rename_map, inplace=True)
    
    
    main_df['subject_id'] = main_df['subject_id'].astype(str).str.strip()
    main_df['environment'] = main_df['environment'].astype(str).str.strip().str.lower()
    pivot_df['subject_id'] = pivot_df['subject_id'].astype(str).str.strip()
    pivot_df['environment'] = pivot_df['environment'].astype(str).str.strip().str.lower()
    
    print(f"Main DF rows: {len(main_df)}")
    print(f"Pivot DF rows: {len(pivot_df)}")
         
    cols_to_update = ['interest_mean_r', 'interest_mean_ir', 'interest_mean_m']
    for col in cols_to_update:
        if col in main_df.columns:
            main_df.drop(columns=[col], inplace=True)
    existing_cols = [col for col in cols_to_update if col in pivot_df.columns]

    
    # merge
    main_df = main_df.merge(
        pivot_df[['subject_id', 'environment'] + existing_cols],
        on=['subject_id', 'environment'],
        how='left'
    )
    
    print(f"After merge - rows: {len(main_df)}")
    for col in cols_to_update:
        if col in main_df.columns:
            print(f"{col}: {main_df[col].notna().sum()} non-null values")
    
    return main_df    
    
def change_SD_interest_lvl(df, from_df):
    """
calculte intrest lvl SD
    """
    from_df = change_envi(from_df.copy())
    overall_sd = from_df[from_df['Condition'] == 'Overall'].copy()
    overall_sd = overall_sd.rename(columns={
        'Subject': 'subject_id',
        'Environment': 'environment',
        'SD': 'interest_sd_all'
    })
    
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['environment'] = df['environment'].astype(str).str.strip().str.lower()
    overall_sd['subject_id'] = overall_sd['subject_id'].astype(str).str.strip()
    overall_sd['environment'] = overall_sd['environment'].astype(str).str.strip().str.lower()
    
    if 'interest_sd_all' in df.columns:
        df.drop(columns=['interest_sd_all'], inplace=True)
    
    df = df.merge(
        overall_sd[['subject_id', 'environment', 'interest_sd_all']],
        on=['subject_id', 'environment'],
        how='left'
    )
    
    print(f"interest_sd_all: {df['interest_sd_all'].notna().sum()} non-null values")
    
    return df

def change_median_interest_lvl(df, from_df):
    
    from_df = change_envi(from_df.copy())
    overall_sd = from_df[from_df['Condition'] == 'Overall'].copy()
    overall_sd = overall_sd.rename(columns={
        'Subject': 'subject_id',
        'Environment': 'environment',
        'Median': 'interest_median_all'
    })
    
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['environment'] = df['environment'].astype(str).str.strip().str.lower()
    overall_sd['subject_id'] = overall_sd['subject_id'].astype(str).str.strip()
    overall_sd['environment'] = overall_sd['environment'].astype(str).str.strip().str.lower()
    
    if 'interest_median_all' in df.columns:
        df.drop(columns=['interest_median_all'], inplace=True)
    
    df = df.merge(
        overall_sd[['subject_id', 'environment', 'interest_median_all']],
        on=['subject_id', 'environment'],
        how='left'
    )
    
    print(f"interest_median_all: {df['interest_median_all'].notna().sum()} non-null values")
    
    return df

def change_sd_lvl_per_condition(main_df,intrest_lvl_df):
    
    # pivot table
    pivot_df = intrest_lvl_df.pivot_table(
        index=['Subject', 'Environment'], 
        columns='Condition',
        values='SD' 
    ).reset_index()
    
    # changing names
    rename_map = {
        'Subject': 'subject_id',
        'Environment': 'environment',
        'R': 'interest_sd_r',
        'IR': 'interest_sd_ir',
        'M': 'interest_sd_m'
    }
    pivot_df.rename(columns=rename_map, inplace=True)
    
    
    main_df['subject_id'] = main_df['subject_id'].astype(str).str.strip()
    main_df['environment'] = main_df['environment'].astype(str).str.strip().str.lower()
    pivot_df['subject_id'] = pivot_df['subject_id'].astype(str).str.strip()
    pivot_df['environment'] = pivot_df['environment'].astype(str).str.strip().str.lower()
    
    print(f"Main DF rows: {len(main_df)}")
    print(f"Pivot DF rows: {len(pivot_df)}")
    
    cols_to_update = ['interest_sd_r', 'interest_sd_ir', 'interest_sd_m']
    for col in cols_to_update:
        if col in main_df.columns:
            main_df.drop(columns=[col], inplace=True)
    existing_cols = [col for col in cols_to_update if col in pivot_df.columns]

    
    # merge
    main_df = main_df.merge(
        pivot_df[['subject_id', 'environment'] + existing_cols],
        on=['subject_id', 'environment'],
        how='left'
    )
    
    print(f"After merge - rows: {len(main_df)}")
    for col in cols_to_update:
        if col in main_df.columns:
            print(f"{col}: {main_df[col].notna().sum()} non-null values")
    
    return main_df    

def change_acc_per_condition(main_df,cafe_df,classroom_df):
    acc_df= pd.concat([cafe_df,classroom_df],ignore_index= True)
    acc_df.columns = acc_df.columns.str.strip()
    
    # pivot table
    pivot_df = acc_df.pivot_table(
        index=['Subject', 'Environment'], 
        columns='Condition',
        values='Accuracy', 
        aggfunc='mean'
    ).reset_index()
    
    # changing names
    rename_map = {
        'Subject': 'subject_id',
        'Environment': 'environment',
        'R': 'q_acc_r',
        'IR': 'q_acc_ir',
        'M': 'q_acc_m'
    }
    pivot_df.rename(columns=rename_map, inplace=True)
    
    
    main_df['subject_id'] = main_df['subject_id'].astype(str).str.strip()
    main_df['environment'] = main_df['environment'].astype(str).str.strip().str.lower()
    pivot_df['subject_id'] = pivot_df['subject_id'].astype(str).str.strip()
    pivot_df['environment'] = pivot_df['environment'].astype(str).str.strip().str.lower()
    
    print(f"Main DF rows: {len(main_df)}")
    print(f"Pivot DF rows: {len(pivot_df)}")
    
    cols_to_update = ['q_acc_r', 'q_acc_ir', 'q_acc_m']
    for col in cols_to_update:
        if col in main_df.columns:
            main_df.drop(columns=[col], inplace=True)
    existing_cols = [col for col in cols_to_update if col in pivot_df.columns]

    
    # merge
    main_df = main_df.merge(
        pivot_df[['subject_id', 'environment'] + existing_cols],
        on=['subject_id', 'environment'],
        how='left'
    )
    
    print(f"After merge - rows: {len(main_df)}")
    for col in cols_to_update:
        if col in main_df.columns:
            print(f"{col}: {main_df[col].notna().sum()} non-null values")
    
    return main_df    


def main():
    
    #create new file if needed
    main_df = check_and_create_main_data()
    
    # reading another files
    print("\nLoading additional data files...")
    df_interest_lvl = pd.read_excel(INTEREST_ALL_OUTPUT)
    df_accuracy_cafe= pd.read_excel(ACCURACY_CAFE_OUTPUT)
    df_accuracy_classroom= pd.read_excel(ACCURACY_CLASSROOM_OUTPUT)
    df_information_of_subjects = pd.read_excel(PARTICIPANTS_INFO, sheet_name="Sheet1")
    df_information_of_subjects = df_information_of_subjects[['Part_id', 'Version']]

    #to read the data from the intrest_level_files
    cafe_int = pd.read_excel(INTEREST_CAFE_OUTPUT)  
    classroom_int = pd.read_excel(INTEREST_CLASSROOM_OUTPUT)  
    all_data_interest_level = pd.concat([cafe_int, classroom_int], ignore_index=True)
    df_adhd = pd.read_excel(DATA_PATH_ADHD)
    print("✓ All files loaded successfully")
    
    # update data
    print("\nUpdating MAIN_DATA with new calculations...")
    main_df['interest_mean_all'] = main_df[['interest_mean_r', 'interest_mean_ir', 'interest_mean_m']].mean(axis=1)
    main_df['interest_median_all'] = main_df[['interest_mean_r', 'interest_mean_ir', 'interest_mean_m']].median(axis=1)
    main_df = change_sd_lvl_per_condition(main_df, df_interest_lvl)
    main_df= change_interest_mean_all(main_df, df_interest_lvl)
    main_df= change_median_interest_lvl(main_df,df_interest_lvl)
    main_df= change_interest_lvl_per_condition(main_df,all_data_interest_level)   
    main_df= change_SD_interest_lvl(main_df,df_interest_lvl) 
    main_df= change_acc_per_condition(main_df,df_accuracy_cafe,df_accuracy_classroom)
    main_df= change_ADHD(main_df,df_adhd)
    main_df= change_version(main_df,df_information_of_subjects)
    #changing the order of the columns
    columns_order= ['subject_id','adhd', 'environment', 'session', 'exp_version',
 'q_acc_all', 'q_acc_r', 'q_acc_ir',
 'q_acc_m', 'interest_mean_all', 'interest_median_all',
 'interest_sd_all', 'interest_mean_r', 'interest_mean_ir',
 'interest_mean_m', 'interest_sd_r', 
 'interest_sd_ir', 'interest_sd_m']
    main_df=main_df[columns_order]
    # saving
    print("\nSaving updated MAIN_DATA...")
    main_df.to_excel(MAIN_DATA, index=False)
    
if __name__== "__main__":
    main()
