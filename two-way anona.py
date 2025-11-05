"""
Statistical Analysis Runner - Complete Version
===============================================
This script runs t-test and Two-Way Mixed ANOVA on Cafe vs Classroom data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import pingouin as pg


CONFIG = {
    'base_path': r"C:\Users\peleg\Desktop\Lub",
    'cafe_assignment_log': r"C:\Users\peleg\Desktop\Lub\assignment_log_cafe.csv",
    'classroom_assignment_log': r"C:\Users\peleg\Desktop\Lub\assignment_log_classroom.csv",
    'answers_folder': r"C:\Users\peleg\Desktop\Lub\All_Answers",
    'output_folder': r"C:\Users\peleg\Desktop\Lub\statistical_analysis"
}



def parse_letter_sequence(input_str):
    letters = []
    i = 0
    while i < len(input_str):
        if i < len(input_str) - 1 and input_str[i:i + 2] == 'IR':
            letters.append('IR')
            i += 2
        else:
            letters.append(input_str[i])
            i += 1

    if len(letters) >= 3:
        return letters[0], letters[1], letters[2]
    else:
        return (letters + [None, None, None])[:3]

def counter_of_numbers(file_data: pd.DataFrame, range_start: int, range_end: int) -> int:
    """
    Count correct answers in range
    """
    filtered = file_data[
        (file_data['TrialID'] >= range_start) &
        (file_data['TrialID'] <= range_end) &
        (file_data['SingleChoiceAccurate'] == True)
    ]
    return len(filtered)

def total_counter_of_numbers(file_data: pd.DataFrame, range_start: int, range_end: int) -> int:
    """
    Count total attempts in range
    """
    filtered = file_data[
        (file_data['TrialID'] >= range_start) &
        (file_data['TrialID'] <= range_end)
    ]
    return len(filtered)


def update_counters_quiet(file_data, first_condition, second_condition, third_condition,
                          counter_R, counter_IR, counter_M,
                          total_R, total_IR, total_M):
    """
    Update counters without printing
    """
    counters = {'R': counter_R, 'IR': counter_IR, 'M': counter_M}
    totals = {'R': total_R, 'IR': total_IR, 'M': total_M}

    conditions = [
        (first_condition, 1, 10),
        (second_condition, 11, 20),
        (third_condition, 21, 30)
    ]

    for condition, range_start, range_end in conditions:
        if condition in counters:
            correct_count = counter_of_numbers(file_data, range_start, range_end)
            counters[condition] += correct_count
            total_count = total_counter_of_numbers(file_data, range_start, range_end)
            totals[condition] += total_count

    return counters['R'], counters['IR'], counters['M'], totals['R'], totals['IR'], totals['M']

def process_single_file(file_path, assignment_log_data, verbose=False):
    """Process one answer file"""
    file_name = os.path.basename(file_path)

    try:
        counter_R = counter_IR = counter_M = 0
        total_R = total_IR = total_M = 0

        file_data = pd.read_csv(file_path)

        filtered_data = file_data[
            (file_data['TrialID'] <= 30) &
            (file_data['QuestionID'] < 7777)
        ]

        if len(filtered_data) == 0:
            return None, "No data after filtering"

        number_of_subject = file_name[:2]
        if number_of_subject[1] == "_":
            number_of_subject = int(number_of_subject[0])
        else:
            number_of_subject = int(number_of_subject)

        file_name_lower = file_path.lower()
        if "_retest_" in file_name_lower:
            number_of_session = 2
        elif "_test_" in file_name_lower:
            number_of_session = 1
        else:
            return None, "Cannot determine session"

        mask = (assignment_log_data['Subject'] == number_of_subject) & \
               (assignment_log_data['Session'] == number_of_session)
        matching_rows = assignment_log_data[mask]

        if matching_rows.empty:
            return None, f"No matching condition"

        order_of_conditions = matching_rows['Condition'].iloc[0]
        environment = matching_rows['Environment'].iloc[0] if 'Environment' in matching_rows.columns else 'unknown'

        first_condition, second_condition, third_condition = parse_letter_sequence(order_of_conditions)

        counter_R, counter_IR, counter_M, total_R, total_IR, total_M = update_counters_quiet(
            filtered_data, first_condition, second_condition, third_condition,
            counter_R, counter_IR, counter_M, total_R, total_IR, total_M
        )

        percentage_R = (counter_R / total_R * 100) if total_R > 0 else 0
        percentage_IR = (counter_IR / total_IR * 100) if total_IR > 0 else 0
        percentage_M = (counter_M / total_M * 100) if total_M > 0 else 0
        total_correct = counter_R + counter_IR + counter_M
        total_attempts = total_R + total_IR + total_M
        overall_percentage = (total_correct / total_attempts * 100) if total_attempts > 0 else 0

        results = {
            'file_name': file_name,
            'subject': number_of_subject,
            'session': number_of_session,
            'environment': environment,
            'condition_order': order_of_conditions,
            'counter_R': counter_R,
            'counter_IR': counter_IR,
            'counter_M': counter_M,
            'total_R': total_R,
            'total_IR': total_IR,
            'total_M': total_M,
            'percentage_R': percentage_R,
            'percentage_IR': percentage_IR,
            'percentage_M': percentage_M,
            'total_correct': total_correct,
            'total_attempts': total_attempts,
            'overall_percentage': overall_percentage
        }

        return results, None

    except Exception as e:
        return None, str(e)

def load_environment_data(assignment_log_path, answers_folder):
    """
    Load data for one environment (Cafe or Classroom)
    """

    try:
        assignment_log = pd.read_csv(assignment_log_path)
        pattern = f"{answers_folder}/*.xlsx"
        answer_files = glob.glob(pattern)

        results = []
        for file_path in answer_files:
            result, error = process_single_file(file_path, assignment_log)
            if result is not None:
                results.append(result)

        if results:
            return pd.DataFrame(results)
        else:
            return None

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

#  T-TEST FUNCTIONS

def check_normality(data, name):
    """Check normality using Shapiro-Wilk test"""
    stat, p = stats.shapiro(data)
    is_normal = p > 0.05

    print(f"  {name}:")
    print(f"    W-statistic: {stat:.4f}")
    print(f"    p-value: {p:.4f}")
    print(f"    Result: {'✓ Normal' if is_normal else '✗ Not Normal'}")

    return is_normal, p

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

def interpret_cohens_d(d):
    """Interpret Cohen's d"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def run_independent_ttest(cafe_data, classroom_data, output_folder):
    """Independent t-test comparing Cafe vs Classroom"""

    print(f"\n{'=' * 80}")
    print(f"INDEPENDENT SAMPLES T-TEST")
    print(f"{'=' * 80}")
    print(f"Comparing: Cafe vs Classroom (Overall Success Rate)")
    print(f"-" * 80)

    cafe_overall = cafe_data['overall_percentage']
    classroom_overall = classroom_data['overall_percentage']

    # Descriptive Statistics
    print(f"\nDescriptive Statistics:")
    print(f"  Cafe:")
    print(f"    n = {len(cafe_overall)}")
    print(f"    Mean = {cafe_overall.mean():.2f}%")
    print(f"    SD = {cafe_overall.std():.2f}%")
    print(f"    Range: {cafe_overall.min():.2f}% - {cafe_overall.max():.2f}%")

    print(f"\n  Classroom:")
    print(f"    n = {len(classroom_overall)}")
    print(f"    Mean = {classroom_overall.mean():.2f}%")
    print(f"    SD = {classroom_overall.std():.2f}%")
    print(f"    Range: {classroom_overall.min():.2f}% - {classroom_overall.max():.2f}%")

    # Assumption Checking
    print(f"\n{'=' * 80}")
    print(f"ASSUMPTION CHECKING")
    print(f"{'=' * 80}")

    print(f"\n1. Normality Test (Shapiro-Wilk):")
    cafe_normal, cafe_p = check_normality(cafe_overall, "Cafe")
    classroom_normal, classroom_p = check_normality(classroom_overall, "Classroom")

    both_normal = cafe_normal and classroom_normal

    print(f"\n2. Levene's Test for Equality of Variances:")
    levene_stat, levene_p = stats.levene(cafe_overall, classroom_overall)
    equal_var = levene_p > 0.05

    print(f"  W-statistic: {levene_stat:.4f}")
    print(f"  p-value: {levene_p:.4f}")
    print(f"  Result: {'✓ Equal variances' if equal_var else '✗ Unequal variances'}")

    # T-Test
    print(f"\n{'=' * 80}")
    print(f"T-TEST RESULTS")
    print(f"{'=' * 80}")

    t_stat, p_value = stats.ttest_ind(cafe_overall, classroom_overall, equal_var=equal_var)
    df = len(cafe_overall) + len(classroom_overall) - 2

    if p_value < 0.001:
        sig_text = "*** p < 0.001"
    elif p_value < 0.01:
        sig_text = "** p < 0.01"
    elif p_value < 0.05:
        sig_text = "* p < 0.05"
    else:
        sig_text = "n.s."

    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f} {sig_text}")
    print(f"degrees of freedom: {df}")

    # Effect Size
    d = cohens_d(cafe_overall, classroom_overall)
    d_interpretation = interpret_cohens_d(d)

    print(f"\nEffect Size (Cohen's d): {d:.3f} ({d_interpretation})")

    # Save Results
    results_dict = {
        'Test': 't-test',
        'Cafe_n': len(cafe_overall),
        'Cafe_Mean': cafe_overall.mean(),
        'Cafe_SD': cafe_overall.std(),
        'Cafe_Min': cafe_overall.min(),
        'Cafe_Max': cafe_overall.max(),
        'Classroom_n': len(classroom_overall),
        'Classroom_Mean': classroom_overall.mean(),
        'Classroom_SD': classroom_overall.std(),
        'Classroom_Min': classroom_overall.min(),
        'Classroom_Max': classroom_overall.max(),
        't_statistic': t_stat,
        'p_value': p_value,
        'df': df,
        'Cohens_d': d,
        'Effect_Size': d_interpretation,
        'Levene_W': levene_stat,
        'Levene_p': levene_p,
        'Equal_Variances': 'Yes' if equal_var else 'No',
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    }

    results_df = pd.DataFrame([results_dict])
    csv_path = f"{output_folder}\\ttest_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    return results_df

#  TWO-WAY ANOVA FUNCTIONS

def prepare_data_for_anova(cafe_data, classroom_data):
    """Prepare data in long format for ANOVA"""

    # Cafe long format
    cafe_long = pd.DataFrame({
        'subject': np.repeat(cafe_data['subject'].values, 3),
        'environment': 'Cafe',
        'category': ['R', 'IR', 'M'] * len(cafe_data),
        'percentage': np.concatenate([
            cafe_data['percentage_R'].values,
            cafe_data['percentage_IR'].values,
            cafe_data['percentage_M'].values
        ])
    })

    # Classroom long format
    classroom_long = pd.DataFrame({
        'subject': np.repeat(classroom_data['subject'].values, 3),
        'environment': 'Classroom',
        'category': ['R', 'IR', 'M'] * len(classroom_data),
        'percentage': np.concatenate([
            classroom_data['percentage_R'].values,
            classroom_data['percentage_IR'].values,
            classroom_data['percentage_M'].values
        ])
    })

    # Offset to avoid subject ID collision
    classroom_long['subject'] = classroom_long['subject'] + 1000

    all_data = pd.concat([cafe_long, classroom_long], ignore_index=True)

    return all_data

def run_two_way_mixed_anova(data, output_folder):
    """Run Two-Way Mixed ANOVA"""

    print(f"\n{'=' * 80}")
    print(f"TWO-WAY MIXED ANOVA")
    print(f"{'=' * 80}")
    print(f"Between-subjects: Environment (Cafe vs Classroom)")
    print(f"Within-subjects: Category (R, IR, M)")
    print(f"-" * 80)

    # Run ANOVA
    aov = pg.mixed_anova(
        data=data,
        dv='percentage',
        within='category',
        between='environment',
        subject='subject'
    )

    print(f"\nANOVA Results:")
    print(aov.to_string())

    # Interpretation
    print(f"\n{'=' * 80}")
    print(f"INTERPRETATION:")
    print(f"{'=' * 80}")

    for idx, row in aov.iterrows():
        source = row['Source']
        p_val = row['p-unc']

        if p_val < 0.05:
            print(f"\n✓ {source}: SIGNIFICANT (p = {p_val:.4f})")
        else:
            print(f"\n✗ {source}: NOT significant (p = {p_val:.4f})")

    # Post-hoc
    posthoc_results = run_posthoc_tests(data, output_folder)

    # Save ANOVA results
    csv_path = f"{output_folder}\\anova_results.csv"
    aov.to_csv(csv_path, index=False)
    print(f"\n✓ ANOVA results saved to: {csv_path}")

    return aov

def run_posthoc_tests(data, output_folder):
    """Post-hoc pairwise comparisons"""

    print(f"\n{'=' * 80}")
    print(f"POST-HOC TESTS")
    print(f"{'=' * 80}")

    all_posthoc_results = []

    # Within each environment - compare categories
    for env in ['Cafe', 'Classroom']:
        print(f"\n{env} - Between Categories:")
        env_data = data[data['environment'] == env]

        posthoc = pg.pairwise_tests(
            data=env_data,
            dv='percentage',
            within='category',
            subject='subject',
            padjust='bonf'
        )

        for idx, row in posthoc.iterrows():
            comparison = f"{row['A']} vs {row['B']}"
            p = row['p-corr']
            sig = "*" if p < 0.05 else "n.s."
            print(f"  {comparison}: p = {p:.4f} {sig}")

            all_posthoc_results.append({
                'Environment': env,
                'Comparison_Type': 'Within_Categories',
                'Group_A': row['A'],
                'Group_B': row['B'],
                'p_uncorrected': row['p-unc'],
                'p_corrected': p,
                'Significant': 'Yes' if p < 0.05 else 'No'
            })

    # Between environments - for each category
    print(f"\n\nBetween Environments (for each Category):")
    for cat in ['R', 'IR', 'M']:
        cat_data = data[data['category'] == cat]
        cafe_vals = cat_data[cat_data['environment'] == 'Cafe']['percentage']
        classroom_vals = cat_data[cat_data['environment'] == 'Classroom']['percentage']

        t, p = stats.ttest_ind(cafe_vals, classroom_vals)
        p_corrected = min(p * 3, 1.0)  # Bonferroni correction

        sig = "*" if p_corrected < 0.05 else "n.s."
        mean_diff = cafe_vals.mean() - classroom_vals.mean()

        print(f"  Category {cat}: Diff = {mean_diff:+.2f}%, p = {p_corrected:.4f} {sig}")

        all_posthoc_results.append({
            'Environment': 'Cafe vs Classroom',
            'Comparison_Type': f'Category_{cat}',
            'Group_A': 'Cafe',
            'Group_B': 'Classroom',
            'Mean_Difference': mean_diff,
            'p_uncorrected': p,
            'p_corrected': p_corrected,
            'Significant': 'Yes' if p_corrected < 0.05 else 'No'
        })

    # Save post-hoc results
    posthoc_df = pd.DataFrame(all_posthoc_results)
    csv_path = f"{output_folder}\\posthoc_results.csv"
    posthoc_df.to_csv(csv_path, index=False)
    print(f"\n✓ Post-hoc results saved to: {csv_path}")

    return posthoc_df

def main():
    """Main execution function"""

    print(f"\n{'#' * 80}")
    print(f"# STATISTICAL ANALYSIS: CAFE VS CLASSROOM")
    print(f"{'#' * 80}\n")

    # Create output folder
    os.makedirs(CONFIG['output_folder'], exist_ok=True)

    # Load Cafe data
    print(f"Loading Cafe data...")
    cafe_data = load_environment_data(
        CONFIG['cafe_assignment_log'],
        CONFIG['answers_folder']
    )

    if cafe_data is None:
        print("✗ Failed to load Cafe data")
        return

    cafe_data = cafe_data[cafe_data['environment'] == 'Cafe']
    print(f"✓ Loaded {len(cafe_data)} Cafe subjects")

    # Load Classroom data
    print(f"\nLoading Classroom data...")
    classroom_data = load_environment_data(
        CONFIG['classroom_assignment_log'],
        CONFIG['answers_folder']
    )

    if classroom_data is None:
        print("✗ Failed to load Classroom data")
        return

    classroom_data = classroom_data[classroom_data['environment'] == 'Classroom']
    print(f"✓ Loaded {len(classroom_data)} Classroom subjects")

    # Run t-test
    print(f"\n{'#' * 80}")
    print(f"# RUNNING INDEPENDENT T-TEST")
    print(f"{'#' * 80}")

    run_independent_ttest(cafe_data, classroom_data, CONFIG['output_folder'])

    # Run ANOVA
    print(f"\n{'#' * 80}")
    print(f"# RUNNING TWO-WAY MIXED ANOVA")
    print(f"{'#' * 80}")

    anova_data = prepare_data_for_anova(cafe_data, classroom_data)
    run_two_way_mixed_anova(anova_data, CONFIG['output_folder'])

    print(f"\n{'#' * 80}")
    print(f"# ANALYSIS COMPLETE!")
    print(f"{'#' * 80}")
    print(f"\nAll results saved to: {CONFIG['output_folder']}")
    print(f"\nGenerated files:")
    print(f"  - ttest_results.csv")
    print(f"  - anova_results.csv")
    print(f"  - posthoc_results.csv")


if __name__ == "__main__":
    main()