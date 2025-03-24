from copy import deepcopy
from argparse import ArgumentParser
from pytorch_lightning import Trainer

import util.rng as rng
import util.callbacks as callbacks
from data.data_loader import get_loaders 
from networks.network import LLL_Net
from trainers.trainer import TLTrainer
from approach import (
    LightningRFS,
<<<<<<< HEAD
    LightningTLModule
=======
    LightningTLModule,
    LightningUnsupervised
>>>>>>> 13490ca (Fix: Unsupervised Learning)
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
    # Exp args
    parser.add_argument('--is-fscil', action='store_true', default=False)
<<<<<<< HEAD
=======
    parser.add_argument('--is-unsupervised', action='store_true', default=False)
>>>>>>> 13490ca (Fix: Unsupervised Learning)
    parser.add_argument(
        '--fields', type=str, default=[], 
        choices=['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL'],
        help='Field or fields used (default=%(default)s)',
        nargs='+', metavar='FIELD'
    )
    # Model args
    parser.add_argument('--network', type=str, default='Lopez17CNN')
    parser.add_argument('--weights-path', type=str, default=None)
    parser.add_argument('--out-features-size', type=int, default=None)
    parser.add_argument(
        '--scale', type=float, default=1,
        help='Scaling factor to modify the number of trainable '
        'parameters used by model (default=%(default)s)'
    )
<<<<<<< HEAD
=======
    # Unsupervised learning args
    parser.add_argument('--learning-rate', type=float, default=1e-3)
>>>>>>> 13490ca (Fix: Unsupervised Learning)
    args = parser.parse_args()
    dict_args = vars(args)
    
    assert (
        not (args.pt_only and args.ft_only) 
    ), '--pt-only and --ft-only cannot be both True'
    
<<<<<<< HEAD
    dict_args_copy = deepcopy(dict_args)  # Used to store the initial args before the training procedure
    dict_args_copy.pop('tpu_cores')  # Removing the tpu_cores entry that should not be saved
=======
    dict_args_copy = deepcopy(dict_args)
    dict_args_copy.pop('tpu_cores')
>>>>>>> 13490ca (Fix: Unsupervised Learning)

    try:
        dict_args['device'] = 'cuda' if args.gpus > 0 else 'cpu'
    except:
        dict_args['device'] = 'cpu'
    
    ####
    ## 1 - GET LOADERS
    ####  
    rng.seed_everything(args.seed)
<<<<<<< HEAD
    
=======

>>>>>>> 13490ca (Fix: Unsupervised Learning)
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
        num_tasks=args.num_tasks,
<<<<<<< HEAD
    )
=======
        is_unsupervised=args.is_unsupervised,
    )
    # ways [7, 3]
    # pretrain_datamodule <data.datamodules.PLDataModule object at 0x7ff376c780d0>
    # finetune_taskset <learn2learn.data.task_dataset.TaskDataset object at 0x7ff376ad0e80>
>>>>>>> 13490ca (Fix: Unsupervised Learning)

    ####
    ## 2 - GET MODEL AND APPROACH
    ####
<<<<<<< HEAD
    args.num_outputs = ways[0]
    net = LLL_Net.factory_network(**vars(args))
    print(net)
    
    if args.patience == -1:
        args.patience = float('inf')
        
    approach = LightningTLModule.factory_approach(
        args.approach, net, **dict_args)
=======
    if args.is_unsupervised:
        # 使用自编码器进行无监督学习
        args.num_outputs =  ways[0]
        net = LLL_Net.factory_network(**vars(args))
        approach = LightningUnsupervised(net, **dict_args)
        
    else:
        # 有监督学习
        args.num_outputs = ways[0] # 7
        
        net = LLL_Net.factory_network(**vars(args))
        approach = LightningTLModule.factory_approach(
            args.approach, net, **dict_args)
>>>>>>> 13490ca (Fix: Unsupervised Learning)

    ####
    ## 3 - TRAIN AND TEST
    ####   
<<<<<<< HEAD
    args.callbacks = [] # Add custom callbacks here 
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
    
    # Adaptation
    if not args.pt_only or args.ft_only:
        ft_res = tl_trainer.adaptation(approach=approach, dataloader=finetune_taskset) 
        eval_res = {**eval_res, **ft_res}
=======
    args.callbacks = []
    tl_trainer = TLTrainer(args, dict_args_copy)

    # Pre-training
    eval_res = {}
    if args.pt_only or not args.ft_only:
        if args.is_unsupervised:
            tl_trainer.fit(approach=approach, datamodule=pretrain_datamodule)
        else:
            # Set finetuning dataset in the trainer
            tl_trainer.set_finetune_taskset(finetune_taskset)
            # Pre-Training fit
            tl_trainer.fit(approach=approach, datamodule=pretrain_datamodule) 
            # Pre-Training test
            eval_res = tl_trainer.test()[0]

    # Adaptation
    if not args.pt_only or args.ft_only:
        if not args.is_unsupervised:
            ft_res = tl_trainer.adaptation(approach=approach, dataloader=finetune_taskset)
            eval_res = {**eval_res, **ft_res}
>>>>>>> 13490ca (Fix: Unsupervised Learning)
    tl_trainer.save_results(eval_res)


if __name__ == '__main__':
    main()
