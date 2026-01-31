#this file create all the graphs of the analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
# Get the base directory (where the script is located)
BASE_DIR = Path(__file__).parent
try:
    from config import *
except ImportError:
    # Default paths - relative to script location
    MAIN_DATA = BASE_DIR / "Output_Results" / "data_peleg.xlsx"
    ACCURACY_CAFE_OUTPUT = BASE_DIR / "Output_Results" / "accuracy" / "results_output_for_accuracy_Cafe.xlsx"
    ACCURACY_CLASSROOM_OUTPUT = BASE_DIR / "Output_Results" / "accuracy" / "results_output_for_accuracy_Classroom.xlsx"
    INTEREST_CAFE_OUTPUT = BASE_DIR / "Output_Results" / "interestlvl" / "results_output_for_intrestLVL_cafe.xlsx"
    INTEREST_CLASSROOM_OUTPUT = BASE_DIR / "Output_Results" / "interestlvl" / "results_output_for_intrestLVL_classroom.xlsx"
    OUTPUT_DIR_VISUALIZATION = BASE_DIR / "Output_Results" / "Visualization"

# Create output directory - works with Path objects
OUTPUT_DIR_VISUALIZATION = Path(OUTPUT_DIR_VISUALIZATION)
OUTPUT_DIR_VISUALIZATION.mkdir(parents=True, exist_ok=True)


def load_all_data():
    #Load all processed data files
    data = {
        'main': pd.read_excel(MAIN_DATA),
        'accuracy_cafe': pd.read_excel(ACCURACY_CAFE_OUTPUT),
        'accuracy_classroom': pd.read_excel(ACCURACY_CLASSROOM_OUTPUT),
        'interest_cafe': pd.read_excel(INTEREST_CAFE_OUTPUT),
        'interest_classroom': pd.read_excel(INTEREST_CLASSROOM_OUTPUT)
    }
# Combine cafe and classroom data
    data['accuracy_all'] = pd.concat([
        data['accuracy_cafe'], 
        data['accuracy_classroom']
    ], ignore_index=True)
    
    data['interest_all'] = pd.concat([
        data['interest_cafe'], 
        data['interest_classroom']
    ], ignore_index=True)
    
    return data

def slope_graph_accuracy(df, save_path):
    
    #Create slope graph showing accuracy change between cafe and classroom
    
    # Pivot data
    pivot_df = df.pivot_table(
        values="Accuracy",
        index="Subject",
        columns="Environment",
        aggfunc="mean"
    )
    
    # Get group information
    group_map = df.drop_duplicates('Subject').set_index('Subject')['Group']
    pivot_df = pivot_df.dropna()

    print(f"Creating slope graph for {len(pivot_df)} participants")

    fig, ax = plt.subplots(figsize=(12, 8))
    n_subjects = len(pivot_df)
    colors = plt.cm.tab20(np.linspace(0, 1, n_subjects))

    # Plot lines for each subject
    for idx, subject_id in enumerate(pivot_df.index):
        if subject_id not in group_map.index:
            continue

        cafe_value = pivot_df.loc[subject_id, 'CAFE']
        classroom_value = pivot_df.loc[subject_id, 'CLASSROOM']
        color = colors[idx]

        ax.plot([0, 1], [cafe_value, classroom_value],
                marker='o', markersize=8, linewidth=2.5,
                alpha=0.7, color=color)

        ax.text(-0.05, cafe_value, str(subject_id),
                fontsize=9, ha='right', va='center',
                color=color, fontweight='bold')

    # Configure axes
    ax.set_xlim(-0.3, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Cafe', 'Classroom'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Change by Environment', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(40, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def comparison_acc_by_condition_and_environment(df, save_path):
    
    #Create grouped bar chart comparing accuracy by condition (R, IR, M) and environment (Cafe vs Classroom)
    
    # Filter data by environment
    cafe_data = df[df['Environment'] == 'CAFE'].copy()
    classroom_data = df[df['Environment'] == 'CLASSROOM'].copy()
    
    # Calculate statistics for each condition in each environment
    conditions = ['R', 'IR', 'M']
    
    cafe_stats = []
    classroom_stats = []
    
    for condition in conditions:
        # Cafe statistics
        cafe_cond = cafe_data[cafe_data['Condition'] == condition]
        cafe_mean = cafe_cond['Accuracy'].mean()
        cafe_std = cafe_cond['Accuracy'].std()
        cafe_stats.append({'mean': cafe_mean, 'std': cafe_std})
        
        # Classroom statistics
        classroom_cond = classroom_data[classroom_data['Condition'] == condition]
        classroom_mean = classroom_cond['Accuracy'].mean()
        classroom_std = classroom_cond['Accuracy'].std()
        classroom_stats.append({'mean': classroom_mean, 'std': classroom_std})
    
    # Calculate overall averages
    cafe_overall_mean = cafe_data['Accuracy'].mean()
    classroom_overall_mean = classroom_data['Accuracy'].mean()
    
    # Count unique subjects
    n_cafe = cafe_data['Subject'].nunique()
    n_classroom = classroom_data['Subject'].nunique()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions
    x = np.arange(len(conditions))
    width = 0.35
    
    # *** EXACT COLORS FROM THE IMAGE ***
    cafe_color = '#E8A13B'  # Orange/Gold for Cafe
    classroom_color = '#4CAF7D'  # Green for Classroom
    
    # Create bars
    cafe_means = [s['mean'] for s in cafe_stats]
    cafe_stds = [s['std'] for s in cafe_stats]
    classroom_means = [s['mean'] for s in classroom_stats]
    classroom_stds = [s['std'] for s in classroom_stats]
    
    bars1 = ax.bar(x - width/2, cafe_means, width, 
                   label='Cafe', color=cafe_color, 
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    bars2 = ax.bar(x + width/2, classroom_means, width,
                   label='Classroom', color=classroom_color,
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add error bars
    ax.errorbar(x - width/2, cafe_means, yerr=cafe_stds,
                fmt='none', ecolor='black', capsize=8, 
                capthick=2, linewidth=2, alpha=0.7)
    
    ax.errorbar(x + width/2, classroom_means, yerr=classroom_stds,
                fmt='none', ecolor='black', capsize=8,
                capthick=2, linewidth=2, alpha=0.7)
    
    # Add mean values on top of bars
    for i, (cafe_mean, classroom_mean) in enumerate(zip(cafe_means, classroom_means)):
        # Cafe label
        ax.text(i - width/2, cafe_mean + cafe_stds[i] + 1.5,
                f'{cafe_mean:.1f}%',
                ha='center', va='bottom', fontsize=13,
                fontweight='bold', color='#D68910')  # Darker orange for text
        
        # Classroom label
        ax.text(i + width/2, classroom_mean + classroom_stds[i] + 1.5,
                f'{classroom_mean:.1f}%',
                ha='center', va='bottom', fontsize=13,
                fontweight='bold', color='#2D8659')  # Darker green for text
    
    # Add SD labels inside bars
    for i, (cafe_bar, classroom_bar) in enumerate(zip(bars1, bars2)):
        # Cafe SD
        cafe_height = cafe_bar.get_height()
        ax.text(cafe_bar.get_x() + cafe_bar.get_width()/2, cafe_height * 0.5,
                f'SD: {cafe_stds[i]:.1f}',
                ha='center', va='center', fontsize=10,
                color="#C27D0F", fontweight='bold')  #changing color by choice
        
        # Classroom SD
        classroom_height = classroom_bar.get_height()
        ax.text(classroom_bar.get_x() + classroom_bar.get_width()/2, classroom_height * 0.5,
                f'SD: {classroom_stds[i]:.1f}',
                ha='center', va='center', fontsize=10,
                color="#14A50C", fontweight='bold')  #changing color by choice
    
    # Configure axes
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison: Cafe vs Classroom\n(Average Success Rate by Category)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    
    # Add legend
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9,
              edgecolor='black', fancybox=True)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add info text box at bottom
    info_text = f'Cafe: n={n_cafe} | Overall avg: {cafe_overall_mean:.1f}%\n'
    info_text += f'Classroom: n={n_classroom} | Overall avg: {classroom_overall_mean:.1f}%'
    
    ax.text(0.02, 0.02, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='black', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")
    
def comparison_acc_by_condition(df, save_path):
    
    #Create bar chart comparing accuracy by condition (R, IR, M)
    
    # Calculate average per subject for each condition
    avg_per_subject = df.groupby(['Condition', 'Subject'])['Accuracy'].mean().reset_index()
    
    # Calculate statistics for each condition
    grouped_data = avg_per_subject.groupby('Condition').agg(
        mean=('Accuracy', 'mean'),
        std=('Accuracy', 'std'),
        count=('Subject', 'nunique')
    ).reset_index()
    
    # Order conditions as R, IR, M
    order = ['R', 'IR', 'M']
    grouped_data['Condition'] = pd.Categorical(grouped_data['Condition'],
                                               categories=order, ordered=True)
    grouped_data = grouped_data.sort_values('Condition')
    avg_per_subject['Condition'] = pd.Categorical(avg_per_subject['Condition'],
                                                   categories=order, ordered=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x_positions = np.arange(len(grouped_data))
    
    # Colors for each condition
    colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green
    
    # Create bars
    bars = ax.bar(x_positions, grouped_data['mean'],
                  color=colors, alpha=0.6,
                  edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax.errorbar(x_positions, grouped_data['mean'], yerr=grouped_data['std'],
                fmt='none', ecolor='black', capsize=8, capthick=2, alpha=0.7)
    
    # Add subject IDs with jitter
    for i, condition in enumerate(grouped_data['Condition']):
        subjects_in_condition = avg_per_subject[avg_per_subject['Condition'] == condition]
        x_jitter = np.random.normal(i + 0.15, 0.04, size=len(subjects_in_condition))
        
        for x_jittered, (_, row) in zip(x_jitter, subjects_in_condition.iterrows()):
            ax.text(x_jittered, row['Accuracy'], str(row['Subject']),
                    fontsize=8, ha='center', va='center',
                    color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.7, linewidth=1),
                    zorder=3)
    
    # Add mean and count labels on top of bars
    for i, (mean_val, count) in enumerate(zip(grouped_data['mean'], grouped_data['count'])):
        ax.text(i, mean_val + grouped_data['std'].iloc[i] + 2,
                f'{mean_val:.2f}%\n(n={int(count)})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Configure axes
    ax.set_xlabel('Condition Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Average Accuracy by Condition Type', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(grouped_data['Condition'], fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")


def histogram_acc_by_environment(df, save_path):
   #Create histograms showing distribution of accuracy scores
    cafe_avg = df[df['Environment'] == 'CAFE'].groupby('Subject')['Accuracy'].mean()
    classroom_avg = df[df['Environment'] == 'CLASSROOM'].groupby('Subject')['Accuracy'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bins = np.arange(0, 105, 5)

    # CAFE histogram
    counts_cafe, bins_cafe, patches_cafe = axes[0].hist(
        cafe_avg, bins=bins, color='#F39C12', alpha=0.7,
        edgecolor='black', linewidth=1.2
    )

    for count, patch in zip(counts_cafe, patches_cafe):
        height = patch.get_height()
        if height > 0:
            axes[0].text(patch.get_x() + patch.get_width() / 2, height + 0.3,
                         f'{int(height)}', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

    axes[0].set_xlabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Subjects', fontsize=13, fontweight='bold')
    axes[0].set_title(f'Accuracy Distribution - CAFE\n(n={len(cafe_avg)} subjects)',
                      fontsize=15, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_xlim(0, 100)

    # CLASSROOM histogram
    counts_classroom, bins_classroom, patches_classroom = axes[1].hist(
        classroom_avg, bins=bins, color='#27AE60', alpha=0.7,
        edgecolor='black', linewidth=1.2
    )

    for count, patch in zip(counts_classroom, patches_classroom):
        height = patch.get_height()
        if height > 0:
            axes[1].text(patch.get_x() + patch.get_width() / 2, height + 0.3,
                         f'{int(height)}', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

    axes[1].set_xlabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Subjects', fontsize=13, fontweight='bold')
    axes[1].set_title(f'Accuracy Distribution - CLASSROOM\n(n={len(classroom_avg)} subjects)',
                      fontsize=15, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")


#visulazation for interest level


def comparison_interest_by_condition_and_environment(df, save_path):
    #Create grouped bar chart comparing interest level by condition (R, IR, M) and environment (Cafe vs Classroom)
    
    # Filter data by environment
    cafe_data = df[df['Environment'] == 'CAFE'].copy()
    classroom_data = df[df['Environment'] == 'CLASSROOM'].copy()
    
    # Calculate statistics for each condition in each environment
    conditions = ['R', 'IR', 'M']
    
    cafe_stats = []
    classroom_stats = []
    
    for condition in conditions:
        # Cafe statistics
        cafe_cond = cafe_data[cafe_data['Condition'] == condition]
        cafe_mean = cafe_cond['Interest_Level'].mean()
        cafe_std = cafe_cond['Interest_Level'].std()
        cafe_stats.append({'mean': cafe_mean, 'std': cafe_std})
        
        # Classroom statistics
        classroom_cond = classroom_data[classroom_data['Condition'] == condition]
        classroom_mean = classroom_cond['Interest_Level'].mean()
        classroom_std = classroom_cond['Interest_Level'].std()
        classroom_stats.append({'mean': classroom_mean, 'std': classroom_std})
    
    # Calculate overall averages
    cafe_overall_mean = cafe_data['Interest_Level'].mean()
    classroom_overall_mean = classroom_data['Interest_Level'].mean()
    
    # Count unique subjects
    n_cafe = cafe_data['Subject'].nunique()
    n_classroom = classroom_data['Subject'].nunique()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions
    x = np.arange(len(conditions))
    width = 0.35
    
    cafe_color = '#E8A13B'  # Orange/Gold for Cafe
    classroom_color = '#4CAF7D'  # Green for Classroom
    
    # Create bars
    cafe_means = [s['mean'] for s in cafe_stats]
    cafe_stds = [s['std'] for s in cafe_stats]
    classroom_means = [s['mean'] for s in classroom_stats]
    classroom_stds = [s['std'] for s in classroom_stats]
    
    bars1 = ax.bar(x - width/2, cafe_means, width, 
                   label='Cafe', color=cafe_color, 
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    bars2 = ax.bar(x + width/2, classroom_means, width,
                   label='Classroom', color=classroom_color,
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add error bars
    ax.errorbar(x - width/2, cafe_means, yerr=cafe_stds,
                fmt='none', ecolor='black', capsize=8, 
                capthick=2, linewidth=2, alpha=0.7)
    
    ax.errorbar(x + width/2, classroom_means, yerr=classroom_stds,
                fmt='none', ecolor='black', capsize=8,
                capthick=2, linewidth=2, alpha=0.7)
    
    # Add mean values on top of bars
    for i, (cafe_mean, classroom_mean) in enumerate(zip(cafe_means, classroom_means)):
        # Cafe label
        ax.text(i - width/2, cafe_mean + cafe_stds[i] + 0.15,
                f'{cafe_mean:.2f}',
                ha='center', va='bottom', fontsize=13,
                fontweight='bold', color='#D68910')  # Darker orange for text
        
        # Classroom label
        ax.text(i + width/2, classroom_mean + classroom_stds[i] + 0.15,
                f'{classroom_mean:.2f}',
                ha='center', va='bottom', fontsize=13,
                fontweight='bold', color='#2D8659')  # Darker green for text
    
    # Add SD labels inside bars
    for i, (cafe_bar, classroom_bar) in enumerate(zip(bars1, bars2)):
        # Cafe SD
        cafe_height = cafe_bar.get_height()
        ax.text(cafe_bar.get_x() + cafe_bar.get_width()/2, cafe_height * 0.5,
                f'SD: {cafe_stds[i]:.2f}',
                ha='center', va='center', fontsize=10,
                color='#8B0000', fontweight='bold')  # Dark red
        
        # Classroom SD
        classroom_height = classroom_bar.get_height()
        ax.text(classroom_bar.get_x() + classroom_bar.get_width()/2, classroom_height * 0.5,
                f'SD: {classroom_stds[i]:.2f}',
                ha='center', va='center', fontsize=10,
                color='#00008B', fontweight='bold')  # Dark blue
    
    # Configure axes
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Interest Level (1-7)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison: Cafe vs Classroom\n(Average Interest Level by Category)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 8)  # Interest level is 1-7, so max at 8
    
    # Add legend
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9,
              edgecolor='black', fancybox=True)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add info text box at bottom
    info_text = f'Cafe: n={n_cafe} | Overall avg: {cafe_overall_mean:.2f}\n'
    info_text += f'Classroom: n={n_classroom} | Overall avg: {classroom_overall_mean:.2f}'
    
    ax.text(0.02, 0.02, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='black', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")
    
def comparison_bar_chart(df, comparison_type, save_path):
#Create bar chart comparing accuracy by group, environment, or session
    #group
    if comparison_type.lower() == 'group':
        avg_per_subject = df.groupby(['Group', 'Subject'])['Accuracy'].mean().reset_index()
        grouped_data = avg_per_subject.groupby('Group').agg(
            mean=('Accuracy', 'mean'),
            std=('Accuracy', 'std'),
            count=('Subject', 'nunique')
        ).reset_index()
        grouped_data['Group'] = grouped_data['Group'].map({0: 'ADHD', 1: 'Control'})
        avg_per_subject['Group'] = avg_per_subject['Group'].map({0: 'ADHD', 1: 'Control'})
        category_col = 'Group'
        title = 'Average Accuracy: ADHD vs Control'
        xlabel = 'Group'
        colors = ['#E74C3C', '#3498DB']
#environment
    elif comparison_type.lower() == 'environment':
        avg_per_subject = df.groupby(['Environment', 'Subject'])['Accuracy'].mean().reset_index()
        grouped_data = avg_per_subject.groupby('Environment').agg(
            mean=('Accuracy', 'mean'),
            std=('Accuracy', 'std'),
            count=('Subject', 'nunique')
        ).reset_index()
        category_col = 'Environment'
        title = 'Average Accuracy: Cafe vs Classroom'
        xlabel = 'Environment'
        colors = ['#F39C12', '#27AE60']
#session
    elif comparison_type.lower() == 'session':
        avg_per_subject = df.groupby(['Session', 'Subject'])['Accuracy'].mean().reset_index()
        grouped_data = avg_per_subject.groupby('Session').agg(
            mean=('Accuracy', 'mean'),
            std=('Accuracy', 'std'),
            count=('Subject', 'nunique')
        ).reset_index()
        grouped_data['Session'] = grouped_data['Session'].map({1: 'Test', 2: 'Retest'})
        avg_per_subject['Session'] = avg_per_subject['Session'].map({1: 'Test', 2: 'Retest'})
        category_col = 'Session'
        title = 'Average Accuracy: Test vs Retest'
        xlabel = 'Session'
        colors = ['#F39C12', '#27AE60']

    fig, ax = plt.subplots(figsize=(12, 8))
    x_positions = np.arange(len(grouped_data))

    bars = ax.bar(x_positions, grouped_data['mean'],
                  color=colors[:len(grouped_data)], alpha=0.6,
                  edgecolor='black', linewidth=1.5)

    ax.errorbar(x_positions, grouped_data['mean'], yerr=grouped_data['std'],
                fmt='none', ecolor='black', capsize=8, capthick=2, alpha=0.7)

    # Add subject IDs with jitter
    for i, category in enumerate(grouped_data.iloc[:, 0]):
        subjects_in_category = avg_per_subject[avg_per_subject[category_col] == category]
        x_jitter = np.random.normal(i + 0.15, 0.04, size=len(subjects_in_category))

        for x_jittered, (_, row) in zip(x_jitter, subjects_in_category.iterrows()):
            ax.text(x_jittered, row['Accuracy'], str(row['Subject']),
                    fontsize=8, ha='center', va='center',
                    color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.7, linewidth=1),
                    zorder=3)

    # Add labels
    for i, (mean_val, count) in enumerate(zip(grouped_data['mean'], grouped_data['count'])):
        ax.text(i, mean_val + grouped_data['std'].iloc[i] + 2,
                f'{mean_val:.2f}%\n(n={int(count)})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(grouped_data.iloc[:, 0], fontsize=13)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def comparison_interest_by_group(df, save_path):
    #Create bar chart comparing interest level between ADHD and Non-ADHD groups
    
    # Calculate statistics for each group
    adhd_data = df[df['Group'] == 0].copy()  # ADHD = 0
    control_data = df[df['Group'] == 1].copy()  # Control = 1
    
    # Calculate mean and SD for each group
    adhd_mean = adhd_data['Interest_Level'].mean()
    adhd_std = adhd_data['Interest_Level'].std()
    adhd_cv = (adhd_std / adhd_mean) * 100  # Coefficient of Variation
    
    control_mean = control_data['Interest_Level'].mean()
    control_std = control_data['Interest_Level'].std()
    control_cv = (control_std / control_mean) * 100
    
    # Count unique subjects
    n_adhd = adhd_data['Subject'].nunique()
    n_control = control_data['Subject'].nunique()
    
    # Calculate combined average
    combined_mean = df['Interest_Level'].mean()
    
    # Calculate difference
    difference = abs(control_mean - adhd_mean)
    better_performance = "Non-ADHD" if control_mean > adhd_mean else "ADHD"
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up bar positions
    x = np.arange(2)
    width = 0.6
    
    # Colors matching the accuracy chart
    adhd_color = "#D2940F"  
    control_color = "#55C718"  
    
    means = [adhd_mean, control_mean]
    stds = [adhd_std, control_std]
    cvs = [adhd_cv, control_cv]
    colors = [adhd_color, control_color]
    labels = ['ADHD', 'Control']
    counts = [n_adhd, n_control]
    
    # Create bars
    bars = ax.bar(x, means, width,
                  color=colors, alpha=0.85,
                  edgecolor='black', linewidth=2)
    
    # Add error bars
    ax.errorbar(x, means, yerr=stds,
                fmt='none', ecolor='black', capsize=10,
                capthick=2.5, linewidth=2.5, alpha=0.8)
    
    # Add mean values on top of bars
    for i, (mean_val, std_val, color) in enumerate(zip(means, stds, colors)):
        # Make text color darker version of bar color
        text_color = "#070707" if i == 0 else "#0C0C0D"
        ax.text(i, mean_val + std_val + 0.15,
                f'{mean_val:.1f}',
                ha='center', va='bottom', fontsize=18,
                fontweight='bold', color=text_color)
    
    # Add n labels at bottom of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, 0.3,
                f'n = {count}',
                ha='center', va='bottom', fontsize=12,
                color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                         edgecolor='black', linewidth=1.5, alpha=0.9))
    # Configure axes
    ax.set_ylabel('Overall Interest Level ', fontsize=16, fontweight='bold')
    ax.set_title('Overall Interest Level Comparison\nADHD vs Control\n(with Standard Deviation and Coefficient of Variation)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold')
    ax.set_ylim(0, 7.5)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Add info text box at bottom
    info_text = f'Difference: {difference:.2f}\n'
    info_text += f'Better performance: {better_performance}\n'
    info_text += f'Total participants: {n_adhd + n_control}'
    
    ax.text(0.4, 0.02, info_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                     edgecolor='black', linewidth=2, alpha=0.9),
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")
    
def histogram_interest_by_environment(df, save_path):
        #Create histograms showing distribution of interest level scores
    
    # Calculate average interest level per subject for each environment
    cafe_avg = df[df['Environment'] == 'CAFE'].groupby('Subject')['Interest_Level'].mean()
    classroom_avg = df[df['Environment'] == 'CLASSROOM'].groupby('Subject')['Interest_Level'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create bins from 1 to 7 with intervals of 0.5
    bins = np.arange(1, 7.6, 0.5)

    # CAFE histogram
    counts_cafe, bins_cafe, patches_cafe = axes[0].hist(
        cafe_avg, bins=bins, color='#F39C12', alpha=0.7,
        edgecolor='black', linewidth=1.2
    )

    # Add count labels on top of bars
    for count, patch in zip(counts_cafe, patches_cafe):
        height = patch.get_height()
        if height > 0:
            axes[0].text(patch.get_x() + patch.get_width() / 2, height + 0.15,
                         f'{int(height)}', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

    axes[0].set_xlabel('Average Interest Level (1-7)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Subjects', fontsize=13, fontweight='bold')
    axes[0].set_title(f'Interest Level Distribution - CAFE\n(n={len(cafe_avg)} subjects)',
                      fontsize=15, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_xlim(1, 7)
    
    # Add mean and SD text box
    cafe_mean = cafe_avg.mean()
    cafe_std = cafe_avg.std()
    cafe_text = f'Mean: {cafe_mean:.2f}\nSD: {cafe_std:.2f}'
    axes[0].text(0.98, 0.97, cafe_text,
                 transform=axes[0].transAxes,
                 fontsize=11, fontweight='bold',
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='black', linewidth=1.5, alpha=0.9))

    # CLASSROOM histogram
    counts_classroom, bins_classroom, patches_classroom = axes[1].hist(
        classroom_avg, bins=bins, color='#27AE60', alpha=0.7,
        edgecolor='black', linewidth=1.2
    )

    # Add count labels on top of bars
    for count, patch in zip(counts_classroom, patches_classroom):
        height = patch.get_height()
        if height > 0:
            axes[1].text(patch.get_x() + patch.get_width() / 2, height + 0.15,
                         f'{int(height)}', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

    axes[1].set_xlabel('Average Interest Level (1-7)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Subjects', fontsize=13, fontweight='bold')
    axes[1].set_title(f'Interest Level Distribution - CLASSROOM\n(n={len(classroom_avg)} subjects)',
                      fontsize=15, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_xlim(1, 7)
    
    # Add mean and SD text box
    classroom_mean = classroom_avg.mean()
    classroom_std = classroom_avg.std()
    classroom_text = f'Mean: {classroom_mean:.2f}\nSD: {classroom_std:.2f}'
    axes[1].text(0.98, 0.97, classroom_text,
                 transform=axes[1].transAxes,
                 fontsize=11, fontweight='bold',
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='black', linewidth=1.5, alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")
    
def slope_graph_interest_level(df, save_path):
    
    #Create slope graph showing accuracy change between cafe and classroom
    
    # Pivot data
    pivot_df = df.pivot_table(
        values="Interest_Level",
        index="Subject",
        columns="Environment",
        aggfunc="mean"
    )
    
    # Get group information
    group_map = df.drop_duplicates('Subject').set_index('Subject')['Group']
    pivot_df = pivot_df.dropna()

    print(f"Creating slope graph for {len(pivot_df)} participants")

    fig, ax = plt.subplots(figsize=(12, 8))
    n_subjects = len(pivot_df)
    colors = plt.cm.tab20(np.linspace(0, 1, n_subjects))

    # Plot lines for each subject
    for idx, subject_id in enumerate(pivot_df.index):
        if subject_id not in group_map.index:
            continue

        cafe_value = pivot_df.loc[subject_id, 'CAFE']
        classroom_value = pivot_df.loc[subject_id, 'CLASSROOM']
        color = colors[idx]

        ax.plot([0, 1], [cafe_value, classroom_value],
                marker='o', markersize=8, linewidth=2.5,
                alpha=0.7, color=color)

        ax.text(-0.05, cafe_value, str(subject_id),
                fontsize=9, ha='right', va='center',
                color=color, fontweight='bold')

    # Configure axes
    ax.set_xlim(-0.3, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Cafe', 'Classroom'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Average interest level (%)', fontsize=14, fontweight='bold')
    ax.set_title('interest level Change by Environment', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 7.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")
    
def comparison_interest_by_session(df, save_path):
    #Create bar chart comparing interest level between Test and Retest sessions
    
    # Calculate average per subject for each session
    avg_per_subject = df.groupby(['Session', 'Subject'])['Interest_Level'].mean().reset_index()
    
    # Calculate statistics for each session
    grouped_data = avg_per_subject.groupby('Session').agg(
        mean=('Interest_Level', 'mean'),
        std=('Interest_Level', 'std'),
        count=('Subject', 'nunique')
    ).reset_index()
    
    # Map session numbers to names
    grouped_data['Session'] = grouped_data['Session'].map({1: 'Test', 2: 'Retest'})
    avg_per_subject['Session'] = avg_per_subject['Session'].map({1: 'Test', 2: 'Retest'})
    
    category_col = 'Session'
    title = 'Average Interest Level: Test vs Retest'
    xlabel = 'Session'
    ylabel = 'Average Interest Level (1-7)'
    colors = ['#F39C12', '#27AE60']  # Matching the accuracy colors

    fig, ax = plt.subplots(figsize=(12, 8))
    x_positions = np.arange(len(grouped_data))

    # Create bars
    bars = ax.bar(x_positions, grouped_data['mean'],
                  color=colors[:len(grouped_data)], alpha=0.6,
                  edgecolor='black', linewidth=1.5)

    # Add error bars
    ax.errorbar(x_positions, grouped_data['mean'], yerr=grouped_data['std'],
                fmt='none', ecolor='black', capsize=8, capthick=2, alpha=0.7)

    # Add subject IDs with jitter
    for i, category in enumerate(grouped_data.iloc[:, 0]):
        subjects_in_category = avg_per_subject[avg_per_subject[category_col] == category]
        x_jitter = np.random.normal(i + 0.15, 0.04, size=len(subjects_in_category))

        for x_jittered, (_, row) in zip(x_jitter, subjects_in_category.iterrows()):
            ax.text(x_jittered, row['Interest_Level'], str(row['Subject']),
                    fontsize=8, ha='center', va='center',
                    color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.7, linewidth=1),
                    zorder=3)

    # Add labels on top of bars
    for i, (mean_val, count) in enumerate(zip(grouped_data['mean'], grouped_data['count'])):
        ax.text(i, mean_val + grouped_data['std'].iloc[i] + 0.2,
                f'{mean_val:.2f}\n(n={int(count)})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Configure axes
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(grouped_data.iloc[:, 0], fontsize=13)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 7.5)  # Scale for Interest Level
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")





def create_all_visualizations():
    
    """Create all visualization charts"""
    print("\n" + "="*80)
    print("CREATING ALL VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\n1. Loading all processed data...")
    data = load_all_data()
    
    # Create accuracy visualizations
    print("\n2. Creating accuracy visualizations...")
    
    slope_graph_accuracy(
        data['accuracy_all'],
        OUTPUT_DIR_VISUALIZATION / 'slope_graph_accuracy.png'
    )
    
    comparison_bar_chart(
        data['accuracy_all'],
        'environment',
        os.path.join(OUTPUT_DIR_VISUALIZATION, 'accuracy_by_environment.png')
    )
    
    comparison_bar_chart(
        data['accuracy_all'],
        'environment',
        OUTPUT_DIR_VISUALIZATION / 'accuracy_by_environment.png'
    )
    
    comparison_bar_chart(
        data['accuracy_all'],
        'session',
        OUTPUT_DIR_VISUALIZATION / 'accuracy_by_session.png'
    )
    
    comparison_acc_by_condition(
        data['accuracy_all'],
        OUTPUT_DIR_VISUALIZATION / 'accuracy_by_condition.png'
    )
    
    histogram_acc_by_environment(
        data['accuracy_all'],
        OUTPUT_DIR_VISUALIZATION / 'accuracy_histogram.png'
    )
    
    comparison_acc_by_condition_and_environment(data['accuracy_all'],
    OUTPUT_DIR_VISUALIZATION / 'condition_and_environment.png')
    
    
    
    # Create interest level visualizations
    print("\n3. Creating interest level visualizations...")
    

    comparison_interest_by_condition_and_environment(data['interest_all'],
        OUTPUT_DIR_VISUALIZATION / 'interest_by_condition_and_environment.png') 
        
        
    histogram_interest_by_environment(data['interest_all'],
        OUTPUT_DIR_VISUALIZATION / 'interest_histogram.png')

    slope_graph_interest_level(data['interest_all'],    
        OUTPUT_DIR_VISUALIZATION / 'interest_slope_bar.png')
    
    comparison_interest_by_group(data['interest_all'],
        OUTPUT_DIR_VISUALIZATION / 'interest_by_group.png')
   
    comparison_interest_by_session(data['interest_all'],
        OUTPUT_DIR_VISUALIZATION / 'interest_by_session.png')
   
    
    print(f" Saved to: {OUTPUT_DIR_VISUALIZATION}")


#  Tests 
def test_slope_graph_accuracy_creates_file(tmp_path):
    df = pd.DataFrame({
        'Subject':     [1, 1, 2, 2],
        'Environment': ['CAFE', 'CLASSROOM', 'CAFE', 'CLASSROOM'],
        'Accuracy':    [80, 85, 70, 75],
        'Group':       [0, 0, 1, 1],
    })
    out = tmp_path / "test_slope.png"
    slope_graph_accuracy(df, out)
    assert out.exists()


def test_comparison_acc_by_condition_and_environment_creates_file(tmp_path):
    df = pd.DataFrame({
        'Subject':     [1, 1, 1, 1, 1, 1],
        'Environment': ['CAFE', 'CAFE', 'CAFE', 'CLASSROOM', 'CLASSROOM', 'CLASSROOM'],
        'Condition':   ['R', 'IR', 'M', 'R', 'IR', 'M'],
        'Accuracy':    [80, 75, 70, 85, 78, 72],
    })
    out = tmp_path / "test_bar.png"
    comparison_acc_by_condition_and_environment(df, out)
    assert out.exists()


if __name__ == "__main__":
    create_all_visualizations()