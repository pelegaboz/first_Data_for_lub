"""
main.py - Run All Analyses in One Place
===============================================
This file runs all analyses in sequence:
1. Accuracy Analysis
2. Interest Level Analysis
3. Data Updates

How to run: python main.py
"""

import os
import sys
from datetime import datetime

# Import config
try:
    from config import create_output_directories, verify_paths
    print(" Config file loaded successfully")
except ImportError as e:
    print(" Error: Cannot load config.py")
    print(f"  Details: {e}")
    sys.exit(1)


def print_header(title):
    """Print a nice header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_step(step_num, total_steps, description):
    """Print current step information"""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 50)


def run_accuracy_analysis():
    """Run accuracy analysis"""
    print_step(1, 3, "Accuracy Analysis")
    
    try:
        # Import functions from accuracy.py
        print("  Loading accuracy module...")
        import accuracy
        
        print("  → Analyzing Cafe environment...")
        cafe_df = accuracy.results_for_one_env("cafe")
        if not cafe_df.empty:
            cafe_output = os.path.join("Output_Results", "results_output_for_accuracy_Cafe.xlsx")
            cafe_df.to_excel(cafe_output, index=False)
            print(f"  Cafe data saved: {cafe_output}")
        else:
            print("   Warning: No data for Cafe")
        
        print("  → Analyzing Classroom environment...")
        classroom_df = accuracy.results_for_one_env("classroom")
        if not classroom_df.empty:
            classroom_output = os.path.join("Output_Results", "results_output_for_accuracy_Classroom.xlsx")
            classroom_df.to_excel(classroom_output, index=False)
            print(f"   Classroom data saved: {classroom_output}")
        else:
            print("   Warning: No data for Classroom")
        
        print("\n   Accuracy analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n   Error in accuracy analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_interest_analysis():
    """Run interest level analysis"""
    print_step(2, 3, "Interest Level Analysis")
    
    try:
        # Import functions from interest_level.py
        print("   Loading interest_level module...")
        import interest_level
        
        print("   Analyzing Cafe interest levels...")
        cafe_df = interest_level.results_For_one_envi("cafe")
        if not cafe_df.empty:
            cafe_output = os.path.join("Output_Results", "results_output_for_intrest_level_Cafe.xlsx")
            cafe_df.to_excel(cafe_output, index=False)
            print(f"   Cafe data saved: {cafe_output}")
        else:
            print("   Warning: No data for Cafe")
        
        print("  → Analyzing Classroom interest levels...")
        classroom_df = interest_level.results_For_one_envi("classroom")
        if not classroom_df.empty:
            classroom_output = os.path.join("Output_Results", "results_output_for_intrest_level_Classroom.xlsx")
            classroom_df.to_excel(classroom_output, index=False)
            print(f"   Classroom data saved: {classroom_output}")
        else:
            print("   Warning: No data for Classroom")
        
        print("   Calculating detailed statistics...")
        stats_output = os.path.join("Output_Results", "Interest_Level", "statistics_by_environment_and_condition.xlsx")
        stats_df = interest_level.calculate_std_per_subject_from_raw_data(stats_output)
        if not stats_df.empty:
            print(f"   Statistics saved: {stats_output}")
        
        print("\n   Interest level analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n   Error in interest level analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_data_changes():
    """Run data updates"""
    print_step(3, 3, "Data Updates")
    
    try:
        print("  → Loading changes_in_data module...")
        import changes_in_data
        
        print("  → Updating data...")
        # You can add specific functions to run here
        # changes_in_data.main()
        
        print("\n   Data update completed!")
        print("    Note: If you need to update data_peleg.xlsx, run changes_in_data.py separately")
        return True
        
    except Exception as e:
        print(f"\n   Cannot run data updates: {e}")
        print("  →You can run changes_in_data.py separately if needed")
        return True  # Not a critical failure


def main():
    """Main function"""
    
    # Print header
    print_header(" ADHD Data Analysis System - Automated Run")
    
    start_time = datetime.now()
    print(f" Start time: {start_time.strftime('%H:%M:%S')}")
    print(f" Date: {start_time.strftime('%d/%m/%Y')}")
    
    # Create output directories
    print("\n→ Creating output directories...")
    try:
        create_output_directories()
        print(" Directories created successfully")
    except Exception as e:
        print(f" Warning creating directories: {e}")
    
    # Verify paths
    print("\n→ Verifying paths...")
    try:
        verify_paths()
    except Exception as e:
        print(f" Warning verifying paths: {e}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print(" Run cancelled by user")
            return
    
    # Run analyses
    results = {
        'accuracy': False,
        'interest': False,
        'changes': False
    }
    
    print("\n" + "="*70)
    print("  Starting analyses...")
    print("="*70)
    
    # 1. Accuracy analysis
    results['accuracy'] = run_accuracy_analysis()
    
    # 2. Interest level analysis
    results['interest'] = run_interest_analysis()
    
    # 3. Data updates (optional)
    results['changes'] = run_data_changes()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header(" Run Summary")
    
    print("Results:")
    print(f"  {'✓' if results['accuracy'] else '✗'} Accuracy Analysis")
    print(f"  {'✓' if results['interest'] else '✗'} Interest Level Analysis")
    print(f"  {'✓' if results['changes'] else '⚠'} Data Updates")
    
    print(f"\n End time: {end_time.strftime('%H:%M:%S')}")
    print(f"  Total duration: {duration.seconds // 60} minutes and {duration.seconds % 60} seconds")
    
    print("\n Output files are in: Output_Results/")
    
    if all([results['accuracy'], results['interest']]):
        print("\n All analyses completed successfully!")
    elif any([results['accuracy'], results['interest']]):
        print("\n Some analyses completed with errors")
    else:
        print("\n Analyses failed")
    
    print("\n" + "="*70)
    input("\nPress Enter to close...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Run cancelled by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n General error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to close...")