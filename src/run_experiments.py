import os
import subprocess
import glob
import argparse

def find_teacher_model(teacher_dir):
    """Find teacher model file"""
    # Find all version directories
    version_dirs = glob.glob(os.path.join(teacher_dir, "lightning_logs/version_*"))
    if not version_dirs:
        raise Exception(f"No version directories found in {teacher_dir}")
    
    # Get the latest version directory
    latest_version = max(version_dirs, key=os.path.getctime)
    
    # Find .pt files in distill_models directory
    distill_dir = os.path.join(latest_version, "distill_models")
    pt_files = glob.glob(os.path.join(distill_dir, "*.pt"))
    
    if not pt_files:
        raise Exception(f"No .pt files found in {distill_dir}")
    
    return pt_files[0]

def run_command(command):
    """Run command and print output"""
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed with return code {process.returncode}")

def main():
    parser = argparse.ArgumentParser(description='Run FSCIL experiments')
    parser.add_argument('--datasets', nargs='+', default=['cic2018', 'edge_iiot', 'iot_nid'],
                      help='Datasets to run experiments on')
    parser.add_argument('--shots', nargs='+', type=int, default=[10, 5],
                      help='Number of shots to try')
    parser.add_argument('--pre-modes', nargs='+', default=['none', 'recon', 'contrastive', 'hybrid'],
                      help='Pre-training modes to try')
    args = parser.parse_args()

    # Dataset configuration
    dataset_configs = {
        'cic2018': '9 3',
        'edge_iiot': '12 3',
        'iot_nid': '7 3'
    }

    for dataset in args.datasets:
        for shot in args.shots:
            for pre_mode in args.pre_modes:
                # Build teacher training command
                teacher_cmd = (
                    f"python3 main.py --is-fscil --dataset {dataset} "
                    f"--fields PL IAT DIR WIN FLG TTL --num-pkts 20 --shots {shot} "
                    f"--queries 40 --gpus 1 --num-tasks 100 --max_epochs 200 --seed 0 "
                    f"--approach rfs --patience 20 --monitor valid_accuracy --min_delta 0.001 "
                    f"--mode max --double-monitor --lr 0.0001 --lr-strat none "
                    f"--classes-per-set {dataset_configs[dataset]} "
                    f"--default_root_dir ../results_rfs_teacher_{dataset}_{shot}shot_{pre_mode} "
                    f"--network UNet1D2D --base-learner nn --pre-mode {pre_mode}"
                )

                # Run teacher training
                print(f"\n{'='*50}")
                print(f"Running teacher training for {dataset} with {shot} shots and {pre_mode} pre-mode")
                print(f"{'='*50}\n")
                run_command(teacher_cmd)

                # Find teacher model file
                teacher_dir = f"../results_rfs_teacher_{dataset}_{shot}shot_{pre_mode}"
                teacher_model = find_teacher_model(teacher_dir)

                # Build student training command
                student_cmd = (
                    f"python3 main.py --is-fscil --dataset {dataset} "
                    f"--fields PL IAT DIR WIN FLG TTL --num-pkts 20 --shots {shot} "
                    f"--queries 40 --gpus 1 --num-tasks 100 --max_epochs 200 --seed 0 "
                    f"--approach rfs --patience 20 --monitor valid_accuracy --min_delta 0.001 "
                    f"--mode max --double-monitor --lr 0.0001 --lr-strat none "
                    f"--classes-per-set {dataset_configs[dataset]} "
                    f"--default_root_dir ../results_rfs_student_{dataset}_{shot}shot_{pre_mode} "
                    f"--network UNet1D2D --base-learner nn --kd-t 1 "
                    f"--teacher-path {teacher_model} --is-distill"
                )

                # Run student training
                print(f"\n{'='*50}")
                print(f"Running student training for {dataset} with {shot} shots and {pre_mode} pre-mode")
                print(f"{'='*50}\n")
                run_command(student_cmd)

if __name__ == "__main__":
    main()