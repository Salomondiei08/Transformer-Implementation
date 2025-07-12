#!/usr/bin/env python3
"""
Master script to run any of the four sequence tasks.
Usage: python run_all_tasks.py [task_name]
Task names: reverse, copy, sort, shift
"""

import sys
import os
import subprocess


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_all_tasks.py [task_name]")
        print("Available tasks: reverse, copy, sort, shift")
        print("Example: python run_all_tasks.py reverse")
        return

    task = sys.argv[1].lower()
    valid_tasks = ['reverse', 'copy', 'sort', 'shift']

    if task not in valid_tasks:
        print(f"Invalid task: {task}")
        print(f"Available tasks: {', '.join(valid_tasks)}")
        return

    # Check if task folder exists
    task_dir = f"tasks/{task}"
    if not os.path.exists(task_dir):
        print(f"Task directory {task_dir} not found!")
        return

    # Change to task directory and run the training script
    script_path = os.path.join(task_dir, f"train_{task}.py")
    if not os.path.exists(script_path):
        print(f"Training script {script_path} not found!")
        return

    print(f"🚀 Running {task} task...")
    print(f"📁 Working directory: {task_dir}")
    print("=" * 50)

    # Change to task directory and run the script
    original_dir = os.getcwd()
    os.chdir(task_dir)

    try:
        # Run the training script
        result = subprocess.run([sys.executable, f"train_{task}.py"],
                                capture_output=False, text=True)

        if result.returncode == 0:
            print(f"\n✅ {task} task completed successfully!")
        else:
            print(f"\n❌ {task} task failed with return code {result.returncode}")

    except Exception as e:
        print(f"\n❌ Error running {task} task: {e}")
    finally:
        # Change back to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
