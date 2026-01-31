"""
Statistical Analysis - T-Tests
Performs paired t-tests comparing Cafe vs Classroom environments
for both Accuracy and Interest Level metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Get the base directory
BASE_DIR = Path(__file__).parent

# CONFIGURATION - Import from config file
try:
    from config import *
    print(" Configuration loaded successfully")
except ImportError:
    print(" Warning: config.py not found, using default paths")
    # Fallback paths
    ACCURACY_CAFE_OUTPUT = BASE_DIR / "Output_Results" / "accuracy" / "results_output_for_accuracy_Cafe.xlsx"
    ACCURACY_CLASSROOM_OUTPUT = BASE_DIR / "Output_Results" / "accuracy" / "results_output_for_accuracy_Classroom.xlsx"
    INTEREST_CAFE_OUTPUT = BASE_DIR / "Output_Results" / "interestlvl" / "results_output_for_intrestLVL_cafe.xlsx"
    INTEREST_CLASSROOM_OUTPUT = BASE_DIR / "Output_Results" / "interestlvl" / "results_output_for_intrestLVL_classroom.xlsx"
    OUTPUT_DIR = BASE_DIR / "Output_Results"


def perform_paired_ttest(cafe_data, classroom_data, metric_name="Metric"):
    #Perform paired t-test between cafe and classroom environments
   
    # Ensure equal length
    if len(cafe_data) != len(classroom_data):
        print(f" Warning: Unequal sample sizes - Cafe: {len(cafe_data)}, Classroom: {len(classroom_data)}")
        return None
    
    # Remove NaN values (paired)
    mask = ~(np.isnan(cafe_data) | np.isnan(classroom_data))
    cafe_clean = cafe_data[mask]
    classroom_clean = classroom_data[mask]
    
    if len(cafe_clean) < 2:
        print(f" Warning: Not enough data points for {metric_name}")
        return None
    
    # Calculate statistics
    cafe_mean = np.mean(cafe_clean)
    cafe_std = np.std(cafe_clean, ddof=1)
    classroom_mean = np.mean(classroom_clean)
    classroom_std = np.std(classroom_clean, ddof=1)
    
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(cafe_clean, classroom_clean)
    
    # Calculate effect size (Cohen's d for paired samples)
    diff = cafe_clean - classroom_clean
    cohen_d = np.mean(diff) / np.std(diff, ddof=1)
    
    # Determine significance level
    if p_value < 0.001:
        significance = "***"
        sig_text = "Highly Significant"
    elif p_value < 0.01:
        significance = "**"
        sig_text = "Very Significant"
    elif p_value < 0.05:
        significance = "*"
        sig_text = "Significant"
    else:
        significance = "ns"
        sig_text = "Not Significant"
    
    return {
        'metric': metric_name,
        'n_pairs': len(cafe_clean),
        'cafe_mean': cafe_mean,
        'cafe_std': cafe_std,
        'classroom_mean': classroom_mean,
        'classroom_std': classroom_std,
        'mean_difference': cafe_mean - classroom_mean,
        't_statistic': t_statistic,
        'p_value': p_value,
        'cohen_d': cohen_d,
        'significance': significance,
        'sig_text': sig_text
    }


def ttest_accuracy_by_environment():
    """
    Perform t-test for accuracy between Cafe and Classroom
    Tests overall accuracy and each condition (R, IR, M)
    """
    print("\n" + "="*80)
    print("T-TEST:accuracy")
    print("="*80)
    
    # Load data
    cafe_df = pd.read_excel(ACCURACY_CAFE_OUTPUT)
    classroom_df = pd.read_excel(ACCURACY_CLASSROOM_OUTPUT)
    
    # Ensure Subject column is consistent
    cafe_df['Subject'] = cafe_df['Subject'].astype(int)
    classroom_df['Subject'] = classroom_df['Subject'].astype(int)
    
    results = []
    
    # Test 1: Overall Accuracy (average across all conditions)
    print("\n1. Overall Accuracy:")
    cafe_overall = cafe_df.groupby('Subject')['Accuracy'].mean()
    classroom_overall = classroom_df.groupby('Subject')['Accuracy'].mean()
    
    # Align subjects
    common_subjects = cafe_overall.index.intersection(classroom_overall.index)
    cafe_overall = cafe_overall.loc[common_subjects].values
    classroom_overall = classroom_overall.loc[common_subjects].values
    
    result = perform_paired_ttest(cafe_overall, classroom_overall, "Overall Accuracy")
    if result:
        results.append(result)
        print_test_result(result)
    
    # Test 2-4: Each condition separately
    conditions = ['R', 'IR', 'M']
    for condition in conditions:
        print(f"\n2. Accuracy - Condition {condition}:")
        
        cafe_cond = cafe_df[cafe_df['Condition'] == condition].groupby('Subject')['Accuracy'].mean()
        classroom_cond = classroom_df[classroom_df['Condition'] == condition].groupby('Subject')['Accuracy'].mean()
        
        # Align subjects
        common_subjects = cafe_cond.index.intersection(classroom_cond.index)
        cafe_cond_values = cafe_cond.loc[common_subjects].values
        classroom_cond_values = classroom_cond.loc[common_subjects].values
        
        result = perform_paired_ttest(cafe_cond_values, classroom_cond_values, f"Accuracy - {condition}")
        if result:
            results.append(result)
            print_test_result(result)
    
    return pd.DataFrame(results)


def ttest_interest_by_environment():
    """
    Perform t-test for interest level between Cafe and Classroom
    Tests overall interest and each condition (R, IR, M)
    """
    print("\n" + "="*80)
    print("T-TEST:intrest level")
    print("="*80)
    
    # Load data
    cafe_df = pd.read_excel(INTEREST_CAFE_OUTPUT)
    classroom_df = pd.read_excel(INTEREST_CLASSROOM_OUTPUT)
    
    # Ensure Subject column is consistent
    cafe_df['Subject'] = cafe_df['Subject'].astype(int)
    classroom_df['Subject'] = classroom_df['Subject'].astype(int)
    
    results = []
    
    # Test 1: Overall Interest Level
    print("\n1. Overall Interest Level:")
    cafe_overall = cafe_df.groupby('Subject')['Interest_Level'].mean()
    classroom_overall = classroom_df.groupby('Subject')['Interest_Level'].mean()
    
    # Align subjects
    common_subjects = cafe_overall.index.intersection(classroom_overall.index)
    cafe_overall = cafe_overall.loc[common_subjects].values
    classroom_overall = classroom_overall.loc[common_subjects].values
    
    result = perform_paired_ttest(cafe_overall, classroom_overall, "Overall Interest Level")
    if result:
        results.append(result)
        print_test_result(result)
    
    # Test 2-4: Each condition separately
    conditions = ['R', 'IR', 'M']
    for condition in conditions:
        print(f"\n2. Interest Level - Condition {condition}:")
        
        cafe_cond = cafe_df[cafe_df['Condition'] == condition].groupby('Subject')['Interest_Level'].mean()
        classroom_cond = classroom_df[classroom_df['Condition'] == condition].groupby('Subject')['Interest_Level'].mean()
        
        # Align subjects
        common_subjects = cafe_cond.index.intersection(classroom_cond.index)
        cafe_cond_values = cafe_cond.loc[common_subjects].values
        classroom_cond_values = classroom_cond.loc[common_subjects].values
        
        result = perform_paired_ttest(cafe_cond_values, classroom_cond_values, f"Interest Level - {condition}")
        if result:
            results.append(result)
            print_test_result(result)
    
    return pd.DataFrame(results)


def print_test_result(result):
    #Print formatted t-test results
    
    
    print(f"   Metric: {result['metric']}")
    print(f"   N pairs: {result['n_pairs']}")
    print(f"   Cafe:      Mean={result['cafe_mean']:.2f}, SD={result['cafe_std']:.2f}")
    print(f"   Classroom: Mean={result['classroom_mean']:.2f}, SD={result['classroom_std']:.2f}")
    print(f"   Difference: {result['mean_difference']:.2f}")
    print(f"   t({result['n_pairs']-1}) = {result['t_statistic']:.3f}, p = {result['p_value']:.4f} {result['significance']}")
    print(f"   Cohen's d = {result['cohen_d']:.3f}")
    print(f"   → {result['sig_text']}")


def create_summary_table(accuracy_results, interest_results, output_path):
    #Create a combined summary table of all t-test results
        # Combine results
    accuracy_results['Category'] = 'Accuracy'
    interest_results['Category'] = 'Interest Level'
    
    combined = pd.concat([accuracy_results, interest_results], ignore_index=True)
    
    # Reorder columns for clarity
    columns_order = [
        'Category', 'metric', 'n_pairs',
        'cafe_mean', 'cafe_std',
        'classroom_mean', 'classroom_std',
        'mean_difference', 't_statistic', 'p_value',
        'cohen_d', 'significance', 'sig_text'
    ]
    
    combined = combined[columns_order]
    
    # Round numeric columns
    numeric_cols = ['cafe_mean', 'cafe_std', 'classroom_mean', 'classroom_std',
                    'mean_difference', 't_statistic', 'p_value', 'cohen_d']
    combined[numeric_cols] = combined[numeric_cols].round(4)
    
    # Save to Excel
    combined.to_excel(output_path, index=False)
    print(f"\n Summary table saved to: {output_path}")
    
    return combined


def main():
    #Main execution function - runs all t-tests
    
    # Create output directory
    stats_dir = OUTPUT_DIR / "statistical_tests"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    print("Comparing Cafe vs Classroom Environments")
    
    # Run accuracy t-tests
    accuracy_results = ttest_accuracy_by_environment()
    accuracy_output = stats_dir / "ttest_accuracy_results.xlsx"
    accuracy_results.to_excel(accuracy_output, index=False)
    print(f"\n Accuracy results saved to: {accuracy_output}")
    
    # Run interest level t-tests
    interest_results = ttest_interest_by_environment()
    interest_output = stats_dir / "ttest_interest_results.xlsx"
    interest_results.to_excel(interest_output, index=False)
    print(f"\n Interest level results saved to: {interest_output}")
    
    # Create combined summary
    summary_output = stats_dir / "ttest_summary_all.xlsx"
    summary_df = create_summary_table(accuracy_results, interest_results, summary_output)
    
    # Print final summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    significant_results = summary_df[summary_df['significance'] != 'ns']
    if len(significant_results) > 0:
        print(f"\n Found {len(significant_results)} significant difference(s):")
        for _, row in significant_results.iterrows():
            print(f"{row['Category']} - {row['metric']}: p={row['p_value']:.4f} {row['significance']}")
    else:
        print("\n No significant differences found between Cafe and Classroom")
    

#  Tests 
def test_perform_paired_ttest_significant():
    cafe =      np.array([90, 88, 85, 92, 87, 91, 89, 86, 93, 88])
    classroom = np.array([70, 68, 65, 72, 67, 71, 69, 66, 73, 68])
    result = perform_paired_ttest(cafe, classroom, "Test")
    assert result is not None
    assert result['p_value'] < 0.05
    assert result['mean_difference'] > 0  
    assert result['cohen_d'] > 0


def test_perform_paired_ttest_no_difference():
    data = np.array([80, 75, 82, 78, 81, 76, 83, 79, 77, 74])
    result = perform_paired_ttest(data, data.copy(), "Test")
    assert result is not None
    assert result['p_value'] == 1.0
    assert result['cohen_d'] == 0.0
    assert result['significance'] == 'ns'
if __name__ == "__main__":
    main()