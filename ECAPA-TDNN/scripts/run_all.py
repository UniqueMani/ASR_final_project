"""
Main Execution Script
Complete workflow for language identification project
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run command and print output"""
    print("\n" + "=" * 60)
    print(f"{description}")
    print("=" * 60)

    result = subprocess.run(
        command,
        shell=True,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"Error: {description} failed!")
        return False

    return True


def main():
    """Main function"""
    print("=" * 60)
    print("Language Identification Project - Complete Workflow")
    print("=" * 60)
    print("\nIncludes the following steps:")
    print("1. Data preprocessing")
    print("2. Model training")
    print("3. Result visualization")

    # Switch to scripts directory
    scripts_dir = Path(__file__).parent
    os.chdir(scripts_dir)

    # Ask user which steps to execute
    print("\nPlease select which steps to execute:")
    print("1. Data preprocessing only")
    print("2. Model training only")
    print("3. Visualization only")
    print("4. Complete workflow (preprocessing + training + visualization)")
    print("5. Inference testing")

    choice = input("\nEnter option (1-5): ").strip()

    if choice == '1' or choice == '4':
        # Step 1: Data preprocessing
        if not run_command(
            f"{sys.executable} preprocess_data.py",
            "Step 1: Data Preprocessing"
        ):
            return

    if choice == '2' or choice == '4':
        # Step 2: Model training
        if not run_command(
            f"{sys.executable} train_ecapa.py",
            "Step 2: Model Training"
        ):
            return

    if choice == '3' or choice == '4':
        # Step 3: Visualization
        if not run_command(
            f"{sys.executable} visualize_tsne.py",
            "Step 3: t-SNE Visualization"
        ):
            return

    if choice == '5':
        # Inference testing
        print("\n" + "=" * 60)
        print("Inference Testing")
        print("=" * 60)
        run_command(
            f"{sys.executable} inference.py",
            "Inference Testing"
        )

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

    # Display result locations
    print("\nResult file locations:")
    print(f"  - Processed data: {Path('../data/processed').resolve()}")
    print(f"  - Trained model: {Path('../models/ecapa_lang_id').resolve()}")
    print(f"  - Visualization results: {Path('../results').resolve()}")


if __name__ == '__main__':
    main()
