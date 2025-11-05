import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def parse_letter_sequence(input_str):
    """
    seperated conditions R,IR,M
    return R,IR,M
    """
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

def calculate_cv(mean, std):
    """
    CV = (std/mean) × 100
    """
    if mean == 0 or pd.isna(mean):
        return 0.0
    return (std / mean) * 100

def update_counters_quiet(file_data, first_condition, second_condition, third_condition,
                          counter_R, counter_IR, counter_M,
                          total_R, total_IR, total_M,
                          counter_of_numbers, total_counter_of_numbers):
    """Same as update_counters but without printing"""
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

def update_counters(file_data, first_condition, second_condition, third_condition,
                    counter_R, counter_IR, counter_M,
                    total_R, total_IR, total_M,
                    counter_of_numbers, total_counter_of_numbers):
    # for each R,IR,M return number of correct answers and % of success
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

            percentage = (correct_count / total_count * 100) if total_count > 0 else 0

            print(
                f"    {condition} (trials {range_start}-{range_end}): {correct_count}/{total_count} ({percentage:.1f}%)")

    return counters['R'], counters['IR'], counters['M'], totals['R'], totals['IR'], totals['M']

def counter_of_numbers(file_data: pd.DataFrame, range_start: int, range_end: int) -> int:
    # calculate correct answers in range
    filtered = file_data[
        (file_data['TrialID'] >= range_start) &
        (file_data['TrialID'] <= range_end) &
        (file_data['SingleChoiceAccurate'] == True)
        ]
    return len(filtered)

def total_counter_of_numbers(file_data: pd.DataFrame, range_start: int, range_end: int) -> int:
    # counting how many attempts in range
    filtered = file_data[
        (file_data['TrialID'] >= range_start) &
        (file_data['TrialID'] <= range_end)
        ]
    return len(filtered)

def calculate_std_dev_for_file(file_data: pd.DataFrame, first_condition, second_condition, third_condition) -> dict:
    """
    calculate the standard deviation for each trial
    Args:
        file_data (DataFrame):  data of file
        first_condition, second_condition, third_condition
    Returns:
        dict:  standard deviation for each trial
    """
    conditions_ranges = {
        first_condition: (1, 10),
        second_condition: (11, 20),
        third_condition: (21, 30)
    }

    std_devs = {}

    for condition, (range_start, range_end) in conditions_ranges.items():
        if condition in ['R', 'IR', 'M']:
            filtered_data = file_data[
                (file_data['TrialID'] >= range_start) &
                (file_data['TrialID'] <= range_end)
                ]

            accuracy_values = filtered_data['SingleChoiceAccurate'].astype(int)

            if len(accuracy_values) > 1:
                std_dev = np.std(accuracy_values, ddof=1)
            else:
                std_dev = 0.0

            std_devs[condition] = std_dev

    return std_devs

def process_single_file(file_path, assignment_log_data, verbose=False):
    """
    processing one file in a time
    Args:
        file_path (str): path of file
        assignment_log_data (DataFrame): data of assignment log
        verbose (bool): if True, print detailed statistics for each file

    Returns:
        dict or tuple: results dict if successful, (None, error_msg) if failed
    """
    file_name = os.path.basename(file_path)

    if verbose:
        print(f"\n{'=' * 60}")

    try:
        # Initialize counters for this file
        counter_R = 0
        counter_IR = 0
        counter_M = 0

        # Initialize total counters for this file
        total_R = 0
        total_IR = 0
        total_M = 0

        # Reading the answers file
        file_data = pd.read_csv(file_path)

        # Filter data
        filtered_data_from_file = file_data[
            (file_data['TrialID'] <= 30) &
            (file_data['QuestionID'] < 7777)
            ]

        if len(filtered_data_from_file) == 0:
            error_msg = "No data after filtering (TrialID <= 30 and QuestionID < 7777)"
            return None, error_msg

        # Getting subject number
        number_of_subject = file_name[:2]
        if number_of_subject[1] == "_":
            number_of_subject = int(number_of_subject[0])
        else:
            number_of_subject = int(number_of_subject)

        # if _retest_ then Session 2
        # if  _test_ then Session 1
        file_name_lower = file_path.lower()
        if "_retest_" in file_name_lower:
            number_of_session = 2
        elif "_test_" in file_name_lower:
            number_of_session = 1
        else:
            error_msg = "Cannot determine session from filename (expected _test_ or _retest_)"
            return None, error_msg

        # Get condition order from assignment log
        mask = (assignment_log_data['Subject'] == number_of_subject) & (
                assignment_log_data['Session'] == number_of_session)
        matching_rows = assignment_log_data[mask]

        if matching_rows.empty:
            error_msg = f"No matching condition in assignment_log (Subject={number_of_subject}, Session={number_of_session})"
            return None, error_msg

        order_of_conditions = matching_rows['Condition'].iloc[0]
        environment = matching_rows['Environment'].iloc[0] if 'Environment' in matching_rows.columns else 'unknown'

        # Parse the condition sequence
        first_condition, second_condition, third_condition = parse_letter_sequence(order_of_conditions)

        # Update counters (only print if verbose)
        if verbose:
            print(f"  File: {file_name}")
            print(f"  Subject: {number_of_subject}, Session: {number_of_session}")
            print(f"  Condition order: {order_of_conditions}")
            print(f"  Parsed: 1st={first_condition}, 2nd={second_condition}, 3rd={third_condition}")

        counter_R, counter_IR, counter_M, total_R, total_IR, total_M = update_counters(
            filtered_data_from_file, first_condition, second_condition, third_condition,
            counter_R, counter_IR, counter_M, total_R, total_IR, total_M,
            counter_of_numbers, total_counter_of_numbers
        ) if verbose else update_counters_quiet(
            filtered_data_from_file, first_condition, second_condition, third_condition,
            counter_R, counter_IR, counter_M, total_R, total_IR, total_M,
            counter_of_numbers, total_counter_of_numbers
        )

        # calculate std for the file
        std_devs = calculate_std_dev_for_file(filtered_data_from_file, first_condition, second_condition,
                                              third_condition)

        # calculate percentage of success  for file
        percentage_R = (counter_R / total_R * 100) if total_R > 0 else 0
        percentage_IR = (counter_IR / total_IR * 100) if total_IR > 0 else 0
        percentage_M = (counter_M / total_M * 100) if total_M > 0 else 0
        total_correct = counter_R + counter_IR + counter_M
        total_attempts = total_R + total_IR + total_M
        overall_percentage = (total_correct / total_attempts * 100) if total_attempts > 0 else 0

        # Prepare results
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
            'std_dev_R': std_devs.get('R', 0.0),
            'std_dev_IR': std_devs.get('IR', 0.0),
            'std_dev_M': std_devs.get('M', 0.0),
            'total_correct': total_correct,
            'total_attempts': total_attempts,
            'overall_percentage': overall_percentage
        }

        if verbose:
            print(f"  RESULTS:")
            print(f"    R: {counter_R}/{total_R} ({percentage_R:.1f}%) [SD: {std_devs.get('R', 0.0):.3f}]")
            print(f"    IR: {counter_IR}/{total_IR} ({percentage_IR:.1f}%) [SD: {std_devs.get('IR', 0.0):.3f}]")
            print(f"    M: {counter_M}/{total_M} ({percentage_M:.1f}%) [SD: {std_devs.get('M', 0.0):.3f}]")
            print(f"    Overall: {total_correct}/{total_attempts} ({overall_percentage:.1f}%)")

        return results, None

    except Exception as e:
        error_msg = str(e)
        return None, error_msg

def create_scatter_plot(plot_data, category, color, output_path):
    """
    יצירת גרף נקודות עבור קטגוריה מסוימת
    """
    plt.figure(figsize=(14, 6))

    # הפרדת נקודות לפי סף 60%
    below_60 = plot_data[plot_data[f'percentage_{category}'] < 60]
    above_60 = plot_data[plot_data[f'percentage_{category}'] >= 60]

    # שרטוט נקודות מעל 60% בצבע הרגיל
    plt.scatter(above_60['subject'], above_60[f'percentage_{category}'],
                alpha=0.7, s=100, c=color, edgecolors='black', linewidth=1,
                label=f'{category} ≥ 60%')

    # שרטוט נקודות מתחת ל-60% בצבע אדום
    plt.scatter(below_60['subject'], below_60[f'percentage_{category}'],
                alpha=0.7, s=100, c='red', edgecolors='black', linewidth=1,
                label=f'{category} < 60%')

    # הוספת קו ממוצע
    mean_val = plot_data[f'percentage_{category}'].mean()
    plt.axhline(y=mean_val, color='darkred', linestyle='--', linewidth=2,
                label=f'Average: {mean_val:.1f}%')

    # הוספת קו ייחוס ב-60%
    plt.axhline(y=60, color='red', linestyle=':', linewidth=1.5, alpha=0.5,
                label='60% threshold')

    # סימון על ציר X עבור ערכים מתחת ל-60%
    for idx, row in below_60.iterrows():
        plt.axvline(x=row['subject'], color='red', alpha=0.2, linestyle='-', linewidth=2)
        plt.text(row['subject'], 2, f"{row['subject']}",
                 ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

    # עיצוב הגרף
    plt.xlabel('Subject Number', fontsize=12, fontweight='bold')
    plt.ylabel(f'Success Rate - {category} (%)', fontsize=12, fontweight='bold')
    plt.title(f'Category {category}: Success Rate by Subject', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 105)
    plt.legend(fontsize=10)

    # הוספת ערכים על הנקודות (אם יש לא יותר מדי נקודות)
    if len(plot_data) <= 40:
        for idx, row in plot_data.iterrows():
            plt.annotate(f"{row[f'percentage_{category}']:.0f}%",
                         (row['subject'], row[f'percentage_{category}']),
                         textcoords="offset points", xytext=(0, 8),
                         ha='center', fontsize=7, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graph saved to: {output_path}")
    plt.show()

def create_combined_plot(plot_data, output_path):
    """
    combines data from different categories
    """
    plt.figure(figsize=(14, 6))

    categories_stats = {}
    for cat in ['R', 'IR', 'M']:
        mean = plot_data[f'percentage_{cat}'].mean()
        std = plot_data[f'percentage_{cat}'].std()
        cv = calculate_cv(mean, std)
        categories_stats[cat] = {'mean': mean, 'std': std, 'cv': cv}

    subjects_below_60 = set()
    for category in ['R', 'IR', 'M']:
        below_60 = plot_data[plot_data[f'percentage_{category}'] < 60]
        subjects_below_60.update(below_60['subject'].tolist())

    #draw the dots for each category
    for category, color in [('R', 'blue'), ('IR', 'green'), ('M', 'orange')]:
        below_60 = plot_data[plot_data[f'percentage_{category}'] < 60]
        above_60 = plot_data[plot_data[f'percentage_{category}'] >= 60]

        plt.scatter(above_60['subject'], above_60[f'percentage_{category}'],
                    alpha=0.6, s=80, c=color, edgecolors='black', linewidth=0.5,
                    label=f'{category} ≥ 60%')

        # dots below 60% in red
        plt.scatter(below_60['subject'], below_60[f'percentage_{category}'],
                    alpha=0.6, s=80, c='red', edgecolors='black', linewidth=0.5)

    # avareg lines
    plt.axhline(y=categories_stats['R']['mean'], color='blue', linestyle='--', linewidth=2,
                label=f"Avg R: {categories_stats['R']['mean']:.1f}% (CV={categories_stats['R']['cv']:.1f}%)")
    plt.axhline(y=categories_stats['IR']['mean'], color='green', linestyle='--', linewidth=2,
                label=f"Avg IR: {categories_stats['IR']['mean']:.1f}% (CV={categories_stats['IR']['cv']:.1f}%)")
    plt.axhline(y=categories_stats['M']['mean'], color='orange', linestyle='--', linewidth=2,
                label=f"Avg M: {categories_stats['M']['mean']:.1f}% (CV={categories_stats['M']['cv']:.1f}%)")

    plt.axhline(y=60, color='red', linestyle=':', linewidth=1.5, alpha=0.5,
                label='60% threshold')

    for subject in subjects_below_60:
        plt.axvline(x=subject, color='red', alpha=0.2, linestyle='-', linewidth=2)
        plt.text(subject, 2, f"{subject}",
                 ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

    plt.scatter([], [], alpha=0.6, s=80, c='red', edgecolors='black', linewidth=0.5,
                label='< 60% (any category)')

    # design
    plt.xlabel('Subject Number', fontsize=12, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    plt.title('All Categories: Success Rate by Subject (Combined)\nCoefficient of Variation shown for each category',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 105)
    plt.legend(fontsize=8, loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined graph saved to: {output_path}")
    plt.show()

def create_bar_chart(plot_data, output_path):
    """
    create_bar_chart(plot_data, output_path)
    """
    # statics
    mean_R = plot_data['percentage_R'].mean()
    mean_IR = plot_data['percentage_IR'].mean()
    mean_M = plot_data['percentage_M'].mean()

    std_R = plot_data['percentage_R'].std()
    std_IR = plot_data['percentage_IR'].std()
    std_M = plot_data['percentage_M'].std()

    cv_R = calculate_cv(mean_R, std_R)
    cv_IR = calculate_cv(mean_IR, std_IR)
    cv_M = calculate_cv(mean_M, std_M)

    categories = ['R', 'IR', 'M']
    means = [mean_R, mean_IR, mean_M]
    stds = [std_R, std_IR, std_M]
    cvs = [cv_R, cv_IR, cv_M]
    colors = ['#3357FF',  # Blue for R
              '#33FF57',  # Green for IR
              '#FFFF00']  # Yellow for M

    plt.figure(figsize=(12, 8))

    bars = plt.bar(categories, means, color=colors, width=0.5,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # added error bars
    plt.errorbar(categories, means, yerr=stds, fmt='none',
                 ecolor='black', capsize=10, linewidth=2, alpha=0.7)

    # add text on bars
    for i, (bar, mean, std, cv) in enumerate(zip(bars, means, stds, cvs)):
        height = bar.get_height()

        # average above bar
        plt.text(bar.get_x() + bar.get_width() / 2, height + std + 2,
                 f'{mean:.1f}%',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

        # std in average
        plt.text(bar.get_x() + bar.get_width() / 2, height * 0.6,
                 f'SD: {std:.1f}',
                 ha='center', va='center', fontsize=10,
                 color='darkred', fontweight='bold')

        plt.text(bar.get_x() + bar.get_width() / 2, height * 0.4,
                 f'CV: {cv:.1f}%',
                 ha='center', va='center', fontsize=10,
                 color='purple', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    overall_mean = sum(means) / len(means)
    plt.axhline(y=overall_mean, color='purple', linestyle=':',
                linewidth=2, alpha=0.6, label=f'Overall avg: {overall_mean:.1f}%')

    # design
    plt.xlabel('Category', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Average Success Rate by Category\n Classroom',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 115)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.legend(fontsize=11, loc='lower right')


    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Bar chart saved to: {output_path}")
    plt.show()

def create_histogram(plot_data, output_path):
    """
    Histogram for succsess rate by precentage
    """
    success_rates = plot_data['overall_percentage']

    #min value
    min_val = int(success_rates.min() // 5) * 5
    max_val = 100

    bins = range(min_val, max_val + 6, 5)  # +6 כדי לוודא שמגיעים עד 100

    mean_val = success_rates.mean()

    plt.figure(figsize=(12, 7))

    n, bins_edges, patches = plt.hist(success_rates, bins=bins,
                                      edgecolor='black', linewidth=1.5,
                                      alpha=0.7, color='steelblue')

    plt.axvline(x=mean_val, color='darkred', linestyle='--',
                linewidth=3, label=f'Average: {mean_val:.1f}%')

    # add text on bars
    for i in range(len(patches)):
        height = patches[i].get_height()
        if height > 0:
            plt.text(patches[i].get_x() + patches[i].get_width() / 2., height,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel('Success Rate (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Students', fontsize=14, fontweight='bold')
    plt.title('Distribution of Overall Success Rate - Classroom\n(5% intervals)',
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.legend(fontsize=12, loc='upper left')

    plt.xlim(min_val, max_val)

    info_text = f'n = {len(plot_data)} students\nRange: {success_rates.min():.1f}% - {success_rates.max():.1f}%'
    plt.text(0.98, 0.02, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Histogram saved to: {output_path}")
    plt.show()

def load_classroom_data(classroom_assignment_log_path, answers_folder_path):
    """
loading classroom data
    """

    print(f"\n{'=' * 80}")
    print(f"Loading Classroom data...")
    print(f"{'=' * 80}")

    try:
        print(f"Loading assignment log from: {classroom_assignment_log_path}")
        classroom_assignment_log = pd.read_csv(classroom_assignment_log_path)

        pattern = f"{answers_folder_path}/*.xlsx"
        answer_files = glob.glob(pattern)
        print(f"Found {len(answer_files)} answer files")

        classroom_results = []

        for file_path in answer_files:
            result, error_msg = process_single_file(file_path, classroom_assignment_log, verbose=False)
            if result is not None:
                classroom_results.append(result)

        if classroom_results:
            classroom_df = pd.DataFrame(classroom_results)
            print(f"✓ Successfully loaded {len(classroom_df)} Classroom files")

            print(f"  Average R: {classroom_df['percentage_R'].mean():.1f}%")
            print(f"  Average IR: {classroom_df['percentage_IR'].mean():.1f}%")
            print(f"  Average M: {classroom_df['percentage_M'].mean():.1f}%")

            return classroom_df
        else:
            print("✗ No Classroom data found")
            return None

    except Exception as e:
        print(f"✗ Error loading Classroom data: {e}")
        return None

def create_comparison_bar_chart(cafe_data, classroom_data, output_path):
    """
    comparing between Cafe and Classroom
    """

    #statics for Cafe
    cafe_stats = {
        'R': {
            'mean': cafe_data['percentage_R'].mean(),
            'std': cafe_data['percentage_R'].std()
        },
        'IR': {
            'mean': cafe_data['percentage_IR'].mean(),
            'std': cafe_data['percentage_IR'].std()
        },
        'M': {
            'mean': cafe_data['percentage_M'].mean(),
            'std': cafe_data['percentage_M'].std()
        }
    }

    #statics for Classroom
    classroom_stats = {
        'R': {
            'mean': classroom_data['percentage_R'].mean(),
            'std': classroom_data['percentage_R'].std()
        },
        'IR': {
            'mean': classroom_data['percentage_IR'].mean(),
            'std': classroom_data['percentage_IR'].std()
        },
        'M': {
            'mean': classroom_data['percentage_M'].mean(),
            'std': classroom_data['percentage_M'].std()
        }
    }
    #stats for graph
    categories = ['R', 'IR', 'M']

    cafe_means = [cafe_stats[cat]['mean'] for cat in categories]
    classroom_means = [classroom_stats[cat]['mean'] for cat in categories]

    cafe_stds = [cafe_stats[cat]['std'] for cat in categories]
    classroom_stds = [classroom_stats[cat]['std'] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    #create graph
    fig, ax = plt.subplots(figsize=(14, 8))

    bars_cafe = ax.bar(x - width / 2, cafe_means, width,
                       label='Cafe', color='#FF6B6B',
                       edgecolor='black', linewidth=1.5, alpha=0.8)

    bars_classroom = ax.bar(x + width / 2, classroom_means, width,
                            label='Classroom', color='#4ECDC4',
                            edgecolor='black', linewidth=1.5, alpha=0.8)

    # add std
    ax.errorbar(x - width / 2, cafe_means, yerr=cafe_stds,
                fmt='none', ecolor='black', capsize=8, linewidth=2, alpha=0.7)

    ax.errorbar(x + width / 2, classroom_means, yerr=classroom_stds,
                fmt='none', ecolor='black', capsize=8, linewidth=2, alpha=0.7)

    # add values on top bars
    #  Cafe
    for i, (bar, mean, std) in enumerate(zip(bars_cafe, cafe_means, cafe_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + std + 2,
                f'{mean:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='#FF6B6B')
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2,
                f'SD: {std:.1f}',
                ha='center', va='center', fontsize=9,
                color='darkred', fontweight='bold')

    #  Classroom
    for i, (bar, mean, std) in enumerate(zip(bars_classroom, classroom_means, classroom_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + std + 2,
                f'{mean:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='#4ECDC4')
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2,
                f'SD: {std:.1f}',
                ha='center', va='center', fontsize=9,
                color='darkblue', fontweight='bold')

    #design the graph
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison: Cafe vs Classroom\n(Average Success Rate by Category)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # adding statistic information
    cafe_overall = sum(cafe_means) / len(cafe_means)
    classroom_overall = sum(classroom_means) / len(classroom_means)

    info_text = f'Cafe: n={len(cafe_data)} | Overall avg: {cafe_overall:.1f}%\n'
    info_text += f'Classroom: n={len(classroom_data)} | Overall avg: {classroom_overall:.1f}%'

    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison chart saved to: {output_path}")
    plt.show()

def create_overall_comparison_chart(cafe_data, classroom_data, output_path):
    """
    comaring the two enviroments total
    """

    # stats for Cafe
    cafe_overall_mean = cafe_data['overall_percentage'].mean()
    cafe_overall_std = cafe_data['overall_percentage'].std()
    cafe_cv = calculate_cv(cafe_overall_mean, cafe_overall_std)
    cafe_n = len(cafe_data)

    # stats for Classroom
    classroom_overall_mean = classroom_data['overall_percentage'].mean()
    classroom_overall_std = classroom_data['overall_percentage'].std()
    classroom_cv = calculate_cv(classroom_overall_mean, classroom_overall_std)
    classroom_n = len(classroom_data)

    # stats for graph
    environments = ['Cafe', 'Classroom']
    means = [cafe_overall_mean, classroom_overall_mean]
    stds = [cafe_overall_std, classroom_overall_std]
    cvs = [cafe_cv, classroom_cv]
    sample_sizes = [cafe_n, classroom_n]

    colors = ['#FF6B6B', '#4ECDC4']  # Red to Cafe, Turquoise-Classroom

    # create the graph
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(environments))
    bars = ax.bar(x, means, color=colors, width=0.6,
                  edgecolor='black', linewidth=2, alpha=0.8)

    #add std
    ax.errorbar(x, means, yerr=stds, fmt='none',
                ecolor='black', capsize=15, linewidth=2.5, alpha=0.8)

    for i, (bar, mean, std, cv, n) in enumerate(zip(bars, means, stds, cvs, sample_sizes)):
        height = bar.get_height()

        ax.text(bar.get_x() + bar.get_width() / 2, height + std + 2,
                f'{mean:.1f}%',
                ha='center', va='bottom', fontsize=16, fontweight='bold',
                color=colors[i])

        ax.text(bar.get_x() + bar.get_width() / 2, height * 0.7,
                f'SD: {std:.2f}%',
                ha='center', va='center', fontsize=11,
                color='darkred', fontweight='bold')

        ax.text(bar.get_x() + bar.get_width() / 2, height * 0.5,
                f'CV: {cv:.2f}%',
                ha='center', va='center', fontsize=11,
                color='purple', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='purple', linewidth=1.5, alpha=0.9))

        ax.text(bar.get_x() + bar.get_width() / 2, height * 0.25,
                f'n = {n}',
                ha='center', va='center', fontsize=10,
                color='darkblue', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          alpha=0.8))

    overall_avg = sum(means) / len(means)
    ax.axhline(y=overall_avg, color='purple', linestyle=':',
               linewidth=2.5, alpha=0.6,
               label=f'Combined average: {overall_avg:.1f}%')

    #design
    ax.set_ylabel('Overall Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Overall Success Rate Comparison\nCafe vs Classroom\n(with Standard Deviation and Coefficient of Variation)',
        fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(environments, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(means) + max(stds) + 15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    #add statistics
    difference = abs(cafe_overall_mean - classroom_overall_mean)
    better_env = "Cafe" if cafe_overall_mean > classroom_overall_mean else "Classroom"

    info_text = f'Difference: {difference:.2f}%\n'
    info_text += f'Better environment: {better_env}\n'
    info_text += f'Total participants: {cafe_n + classroom_n}'

    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat',
                      edgecolor='black', linewidth=1.5, alpha=0.8))

    #adding explanation
    legend_text = 'SD = Standard Deviation (variability)\n'
    legend_text += 'CV = Coefficient of Variation (SD/Mean×100)\n'
    legend_text += 'Lower CV = more consistent performance'

    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue',
                      alpha=0.7, edgecolor='navy', linewidth=1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Overall comparison chart saved to: {output_path}")
    plt.show()

    #printing
    print(f"\n{'=' * 80}")
    print(f"OVERALL COMPARISON SUMMARY:")
    print(f"{'=' * 80}")
    print(f"Cafe:")
    print(f"  Mean: {cafe_overall_mean:.2f}%")
    print(f"  SD: {cafe_overall_std:.2f}%")
    print(f"  CV: {cafe_cv:.2f}%")
    print(f"  n: {cafe_n}")
    print(f"\nClassroom:")
    print(f"  Mean: {classroom_overall_mean:.2f}%")
    print(f"  SD: {classroom_overall_std:.2f}%")
    print(f"  CV: {classroom_cv:.2f}%")
    print(f"  n: {classroom_n}")
    print(f"\nDifference: {difference:.2f}% (favoring {better_env})")
    print(f"{'=' * 80}\n")

def main():
    # reading assignment_log for all data
    assignment_log_path = r"C:\Users\peleg\Desktop\Lub\assignment_log_cafe.csv"
    print(f"Loading assignment log from: {assignment_log_path}")
    try:
        assignment_log_data = pd.read_csv(assignment_log_path)
        print(f"\n{'=' * 80}")
        print(f"Session distribution:")
        print(f"{'=' * 80}")
        print(assignment_log_data.groupby('Session').size())

    except Exception as e:
        print(f"Error loading assignment log: {e}")
        return

    folder_path = r"C:\Users\peleg\Desktop\Lub\All_Answers"
    pattern = f"{folder_path}/*.xlsx"
    answer_files = glob.glob(pattern)

    print(f"\nFound {len(answer_files)} files to process")
    print(f"Processing files... (verbose output disabled for cleaner logs)\n")

    all_results = []
    failed_files = []

    for i, file_path in enumerate(answer_files, 1):
        result, error_msg = process_single_file(file_path, assignment_log_data, verbose=False)
        if result is not None:
            all_results.append(result)
        else:
            failed_files.append({
                'file_name': os.path.basename(file_path),
                'error': error_msg
            })

    if all_results:
        results_df = pd.DataFrame(all_results)
        cafe_only = results_df[results_df['environment'] == 'Cafe']

        print(f"\n{'=' * 80}")
        print(f"Cafe ONLY - Found {len(cafe_only)} files")
        print(f"{'=' * 80}")

        if len(cafe_only) > 0:
            print(f"Average R: {cafe_only['percentage_R'].mean():.1f}%")
            print(f"Average IR: {cafe_only['percentage_IR'].mean():.1f}%")
            print(f"Average M: {cafe_only['percentage_M'].mean():.1f}%")
            print(f"Overall: {cafe_only['overall_percentage'].mean():.1f}%")

            plot_data = results_df.sort_values('subject')

            print(f"\n{'=' * 80}")
            print(f"Creating scatter plots for all categories...")
            print(f"{'=' * 80}")

            base_path = r"C:\Users\peleg\Desktop\Lub\cafe_graf"




            #comparing to classroom
            print(f"\n{'=' * 80}")
            print(f"COMPARISON: Loading Classroom data for comparison...")
            print(f"{'=' * 80}")

            classroom_assignment_log_path = r"C:\Users\peleg\Desktop\Lub\assignment_log_classroom.csv"
            classroom_answers_folder = r"C:\Users\peleg\Desktop\Lub\All_Answers"
            classroom_data = load_classroom_data(classroom_assignment_log_path, classroom_answers_folder)

            if classroom_data is not None and len(cafe_only) > 0:
                print(f"\n{'=' * 80}")
                print(f"Creating comparison chart...")
                print(f"{'=' * 80}")

                create_comparison_bar_chart( cafe_only, classroom_data,f"{base_path}\\comparison_cafe_vs_classroom.png")
            else:
                print("⚠ Skipping comparison chart - missing data")
                # גרף השוואה כללי חדש
                print(f"\n{'=' * 80}")
                print(f"Creating overall comparison chart...")
                print(f"{'=' * 80}")

            create_overall_comparison_chart(
                cafe_only,
                classroom_data,
                f"{base_path}\\overall_comparison_cafe_vs_classroom.png"
            )


if __name__ == "__main__":
    main()