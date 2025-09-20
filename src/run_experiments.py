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
    parser.add_argument('--shots', nargs='+', type=int, default=[9, 8, 7, 6, 5],
                      help='Number of shots to try')
    parser.add_argument('--pre-modes', nargs='+', default=['none', 'recon', 'contrastive', 'hybrid'],
                      help='Pre-training modes to try')
    parser.add_argument('--classifier', nargs='+', default=['nn', 'lr'],
                      help='Pre-training modes to try') 
    parser.add_argument('--memory-selectors', nargs='+', default=['herding', 'uncertainty', 'random'],
                      help='Memory selectors to try')
    parser.add_argument('--noise-ratios', nargs='+', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help='Noise ratios to try')
    args = parser.parse_args()

    # Dataset configuration
    dataset_configs = {
        'cic2018': {
            'classes-per-set': '9 3',
            'nn': {
                'pre-mode':'hybrid', 
                'memory_selection':'random'
            },
            'lr': {
                'pre-mode':'contrastive', 
                'memory_selection':'uncertainty'
            }
        },
        'edge_iiot': {
            'classes-per-set': '12 3',
            'nn': {
                'pre-mode':'none', 
                'memory_selection':'herding'
            },
            'lr': {
                'pre-mode':'none', 
                'memory_selection':'herding'
            }
        },
        'iot_nid': {
            'classes-per-set': '7 3',
            'nn': {
                'pre-mode':'contrastive', 
                'memory_selection':'uncertainty'
            },
            'lr': {
                'pre-mode':'contrastive', 
                'memory_selection':'random'
            }
        }
    }
    mix_datasets = ['cic2018_mix_iot_nid',
                   'cic2018_mix_edge_iiot',
                   'edge_iiot_mix_cic2018',
                   'edge_iiot_mix_iot_nid',
                   'iot_nid_mix_cic2018',
                   'iot_nid_mix_edge_iiot']
    
    # for mix_dataset in mix_datasets:
    for dataset in args.datasets:
        for shot in args.shots:
            # for pre_mode in args.pre_modes:
            for classifier in args.classifier:
                for noise_ratio in args.noise_ratios:
                # for memory_selector in args.memory_selectors:
                    # # Build teacher training command
                    # teacher_cmd = (
                    #     f"python3 main.py --is-fscil --dataset {dataset} "
                    #     f"--fields PL IAT DIR WIN FLG TTL --num-pkts 20 --shots {shot} "
                    #     f"--queries 40 --gpus 1 --num-tasks 100 --max_epochs {dataset_configs[dataset][2]} --seed 0 "
                    #     f"--approach rfs --patience 20 --monitor valid_accuracy --min_delta 0.001 "
                    #     f"--mode max --double-monitor --lr 0.0001 --lr-strat none "
                    #     f"--classes-per-set {dataset_configs[dataset][0]} "
                    #     f"--default_root_dir ../results_rfs_teacher_allpre/results_rfs_teacher_{dataset}_{shot}shot_{pre_mode}_{classifier}_{memory_selector} "
                    #     f"--network UNet1D2D --base-learner {classifier} --pre-mode {pre_mode} "
                    #     f"--memory-selector {memory_selector} "
                    #     # f"--noise-label --noise-ratio {noise_ratio} --denoising LOF"
                    # )

                    # # Run teacher training
                    # print(f"\n{'='*50}")
                    # print(f"Running teacher training for {dataset} with {shot} shots, {pre_mode} pre-mode, {memory_selector} selector")
                    # print(f"{'='*50}\n")
                    # run_command(teacher_cmd)
                # dataset = mix_dataset.split('_mix_')[0]
                
                    # Find teacher model file
                    pre_mode = dataset_configs[dataset][classifier]['pre-mode']
                    memory_selector = dataset_configs[dataset][classifier]['memory_selection']
                    classes_per_set = dataset_configs[dataset]['classes-per-set']
                    print(f"Running student training for {dataset} with {shot} shots, {pre_mode} pre-mode, {memory_selector} selector, {classifier} classifier, {noise_ratio} noise ratio, denoising DCML")


                    teacher_dir = f"../save_files/results_rfs_teacher_allpre/results_rfs_teacher_{dataset}_10shot_{pre_mode}_{classifier}_{memory_selector}"
                    teacher_model = find_teacher_model(teacher_dir)

                    # Build student training command
                    student_cmd = (
                        f"python3 main.py --is-fscil --dataset {dataset} "
                        f"--fields PL IAT DIR WIN FLG TTL --num-pkts 20 --shots {shot} "
                        f"--queries 40 --gpus 1 --num-tasks 100 --max_epochs 100 --seed 0 "
                        f"--approach rfs --patience 20 --monitor valid_accuracy --min_delta 0.001 "
                        f"--mode max --double-monitor --lr 0.0001 --lr-strat none "
                        f"--classes-per-set {classes_per_set} "
                        f"--default_root_dir ../save_files/results_rfs_student_bestcombo_DCML_denoise_new/results_rfs_student_{dataset}_{shot}shot_{pre_mode}_{classifier}_{memory_selector}_noise_{noise_ratio}_denoising_DCML "
                        f"--network UNet1D2D --base-learner {classifier} --kd-t 1 "
                        f"--teacher-path {teacher_model} --is-distill --memory-selector {memory_selector} "
                        f"--noise-label --noise-ratio {noise_ratio}  --denoising DCML"  
                    )
                    print(student_cmd)

                    # Run student training
                    print(f"\n{'='*50}")
                    print(f"Running student training for {dataset} with {shot} shots, {pre_mode} pre-mode, {memory_selector} selector, {classifier} classifier, {noise_ratio} noise ratio, denoising DCML")
                    print(f"{'='*50}\n")
                    
                    run_command(student_cmd)

if __name__ == "__main__":
    main()