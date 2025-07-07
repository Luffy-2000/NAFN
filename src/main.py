from copy import deepcopy
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import torch
import util.rng as rng
import util.callbacks as callbacks
from data.data_loader import get_loaders, MemoryTaskDataset, EpisodeLoader
from networks.network import LLL_Net
from trainers.trainer import TLTrainer
# from data.memory_selection import HerdingExemplarsSelector, UncertaintyExemplarsSelector
from data.memory_selection import ExemplarsSelector
from approach import (
    LightningRFS,
    LightningTLModule,
    LightningUnsupervised
)


def main():
    ####
    ## 0 - PARSING INPUT
    ####    
    parser = ArgumentParser(conflict_handler='resolve', add_help=True) 
    parser = LightningRFS.add_model_specific_args(parser)
    parser = callbacks.EarlyStoppingDoubleMetric.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--approach', type=str, default='finetuning')
    parser.add_argument('--pt-only', action='store_true', default=False)
    parser.add_argument('--ft-only', action='store_true', default=False)
    # Dataset args
    parser.add_argument('--dataset', type=str, default='iot_nid')
    parser.add_argument('--num-pkts', type=int, default=None)
    parser.add_argument('--num-tasks', type=int, default=100)
    parser.add_argument('--classes-per-set', type=int, default=[], nargs='+')
    parser.add_argument('--shuffle-classes', action='store_true', default=False)
    parser.add_argument('--memory-selector', type=str, default='herding',
                      choices=['herding', 'uncertainty'],
                      help='Memory selector type for FSCIL (herding or uncertainty)')
    # Exp args
    parser.add_argument('--is-fscil', action='store_true', default=False)
    parser.add_argument(
        '--fields', type=str, default=[], 
        choices=['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL'],
        help='Field or fields used (default=%(default)s)',
        nargs='+', metavar='FIELD'
    )
    # Model args
    parser.add_argument('--network', type=str, default='Lopez17CNN')
    parser.add_argument('--weights-path', type=str, default=None)
    parser.add_argument('--out-features-size', type=int, default=320)
    parser.add_argument(
        '--scale', type=float, default=1,
        help='Scaling factor to modify the number of trainable '
        'parameters used by model (default=%(default)s)'
    )
    # Teacher model learning args
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--pre-mode', type=str, default='none', choices=['recon', 'contrastive', 'hybrid', 'none'],
                        help='Teacher model learning mode: reconstruction, contrastive learning, hybrid, or none')
    parser.add_argument('--temperature', type=float, default=0.5, 
                        help='Temperature parameter for contrastive loss')
    parser.add_argument('--transform-strength', type=float, default=0.8,
                        help='Strength of data augmentation transformations (0-1)')
    parser.add_argument('--recon-weight', type=float, default=0.3,
                        help='Weight for reconstruction loss')
    parser.add_argument('--ce-weight', type=float, default=0.7,
                        help='Weight for cross-entropy loss')
    parser.add_argument('--contrastive-weight', type=float, default=0.3,
                        help='Weight for contrastive loss')
    parser.add_argument('--noise-label', action='store_true', default=False,
                        help='If set, add label noise to support set in each episode')
    parser.add_argument('--noise-ratio', type=float, default=0.2,
                        help='Ratio of support samples to corrupt with label noise (default=0.2)')
    args = parser.parse_args()
    dict_args = vars(args)
    
    assert (
        not (args.pt_only and args.ft_only) 
    ), '--pt-only and --ft-only cannot be both True'
    
    dict_args_copy = deepcopy(dict_args)
    dict_args_copy.pop('tpu_cores')

    try:
        dict_args['device'] = 'cuda' if args.gpus > 0 else 'cpu'
    except:
        dict_args['device'] = 'cpu'
    
    ####
    ## 1 - GET LOADERS
    ####  
    rng.seed_everything(args.seed)
    ways, pretrain_datamodule, finetune_taskset = get_loaders(
        dataset=args.dataset,
        num_pkts=args.num_pkts, 
        fields=args.fields, 
        seed=args.seed, 
        classes_per_set=args.classes_per_set,
        queries=args.queries, 
        shots=args.shots, 
        shuffle_classes=args.shuffle_classes,
        is_fscil=args.is_fscil,
        num_tasks=args.num_tasks
    )
    
    # Add label noise to finetune_taskset support samples if required
    def add_label_noise(dataset, noise_ratio, num_classes):
        """
        Randomly corrupt a portion of the labels in the dataset (in-place).
        Only for support (train) samples, not query.
        """
        n = len(dataset)
        num_noisy = int(n * noise_ratio)
        if num_noisy == 0:
            return []
        import numpy as np
        noisy_indices = np.random.choice(n, num_noisy, replace=False)
        for idx in noisy_indices:
            true_label = int(dataset.labels[idx])
            # Choose a wrong label
            candidates = [l for l in range(num_classes) if l != true_label]
            noisy_label = np.random.choice(candidates)
            dataset.labels[idx] = noisy_label
        return noisy_indices

    if args.noise_label:
        # 假设finetune_taskset是NetworkingDataset类型
        # 只对support部分加噪，query不加噪
        # 这里假设support和query是分开的，如果不是，需要在采样时区分
        # 这里默认全部加噪（实际可根据采样逻辑调整）
        noise_ratio = 0.2  # 可调整
        num_classes = ways[1]  # 新类数量
        print(f"Adding label noise to finetune_taskset: ratio={noise_ratio}, num_classes={num_classes}")
        add_label_noise(finetune_taskset, noise_ratio, num_classes)
    
    # ways [7, 3]
    # pretrain_datamodule <data.datamodules.PLDataModule object at 0x7ff376c780d0>
    # finetune_taskset <learn2learn.data.task_dataset.TaskDataset object at 0x7ff376ad0e80>


    ####
    ## 2 - GET MODEL AND APPROACH
    ####
    args.num_outputs = ways[0]
    net = LLL_Net.factory_network(**vars(args))
    print(net)
    
    if args.patience == -1:
        args.patience = float('inf')
        
    approach = LightningTLModule.factory_approach(
        args.approach, net, **dict_args)
    
    
    ####
    ## 3 - TRAIN AND TEST
    ####   
    args.callbacks = []
    tl_trainer = TLTrainer(args, dict_args_copy)

    # Pre-training
    eval_res = {}
    if args.pt_only or not args.ft_only:
        # Set finetuning dataset in the trainer
        tl_trainer.set_finetune_taskset(finetune_taskset)
        # Pre-Training fit
        tl_trainer.fit(approach=approach, datamodule=pretrain_datamodule) 
        # Pre-Training test
        eval_res = tl_trainer.test()[0]


    # initialize memory selection
    memory_task_dataset = MemoryTaskDataset(
        dataset=finetune_taskset,
        memory_dataset = pretrain_datamodule.train_set, 
        memory_selector=args.memory_selector, 
        ways=ways[1],  # 新类数量
        shots=args.shots, 
        queries=args.queries,
        old_class_ids=list(range(ways[0])),
        new_class_ids=list(range(ways[0], ways[0] + ways[1])),
        noise_label=args.noise_label,
        noise_ratio=args.noise_ratio,
    )
    memory_task_dataset.initialize_memory(net, pretrain_datamodule.train_set)
    
    episode_loader = torch.utils.data.DataLoader(
        EpisodeLoader(memory_task_dataset, num_episodes=args.num_tasks),
        batch_size=None
    )

    # Adaptation
    if not args.pt_only or args.ft_only:
        # if not args.is_unsupervised:
            ft_res = tl_trainer.adaptation(approach=approach, dataloader=episode_loader)
            # print(f"ft_res: {ft_res}")
            eval_res = {**eval_res, **ft_res}
    tl_trainer.save_results(eval_res)


if __name__ == '__main__':
    main()
