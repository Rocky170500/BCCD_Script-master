import subprocess
import sys

def run(step_name, command):
    print(f"\n{'='*60}")
    print(f"Running: {step_name}")
    print(f"{'='*60}\n")

    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n {step_name} failed. Stopping pipeline.")
        sys.exit(1)

    print(f"\n {step_name} completed successfully.")

if __name__ == "__main__":
    
    run(
        "Model Training",
        "python scripts/finetune_detr.py"
    )
    
    run(
        "Precision-Recall Curve",
        "python scripts/plot_pr_curve.py"
    )

    run(
        "Confusion Matrix",
        "python scripts/confusion_matrix.py"
    )

    run(
        "Visualization",
        "python scripts/visual_prediction.py"
    )

    print("\n FULL PIPELINE EXECUTED SUCCESSFULLY!")
