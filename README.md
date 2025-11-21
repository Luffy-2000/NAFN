# fscil-nids

## Installation


1. Place the dataset according to the the path defined in [`dataset_config.py`](src/data/dataset_config.py).

2. It is recommended to use `virtualenv` to create an isolated Python environment:
    ```bash
    virtualenv venv
    source venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## How To Use It

Navigate to the `src` directory and execute the main script:
```bash
cd src
python3 -u main.py
```
followed by general options:

- `--gpus`: Number of GPUs to use for the experiment (default: 0)
- `--max_epochs`: Maximum number of epochs
- `--default_root_dir`: Default path for logs and weights if no logger or `lightning.pytorch.callbacks.ModelCheckpoint` callback is passed
- `--seed`: Random seed
- `--is-fscil`: Enable Few-Shot Class Incremental (FSCIL) procedure if present; otherwise, use Few-Shot Learning (FSL) procedure
- `--pt-only`: Run only the pre-training phase if present
- `--ft-only`: Run only the adaptation phase if present
- `--num-tasks`: Number of episodes used for the adaptation task (default: 100)
- `--shuffle-classes`: shuffles FSL/FSCIL partition classes based on the seed if present
- `--classes-per-set`: Number of classes for pre-training and adaptation (e.g., '7 3' means the first 7 classes are used for pre-training and the next 3 for adaptation, order defined in `dataset_config.py`). If not set, the 'train_classes' for pre-training and 'test_classes' for adaptation are used, as defined in `dataset_config.py`.
- Additional options are those defined by the PyTorch Lightnining `Trainer` class. They can be viewed via:
  ```bash
  python3 -u main.py --help
  ```

and specific options for `--approach`,`--datasets` and `--network`.

### Approach Options

`LightningTLModule` in [`tl_module.py`](src/approach/tl_module.py) defines methods for pre-training and fine-tuning along with utility functions. `LightningRFS` inherits from this class. To run this approach, add the name of the `*.py` file to `--approach`.

Generic options in `LightningTLModule` include:

- `--shots`: Number of shots used during fine-tuning (default: 5)
- `--queries`: Number of queries used during fine-tuning (default: 5)
- `--lr`: Starting learning rate (default: 0.001)
- `--lr-strat`: Learning rate scheduler strategy (default: 'none') (choices: ['lrop', 'cawr', 'none'])
- `--scheduler_patience`: Reduce LR on plateau (lrop) scheduler patience (default: 10)
- `--scheduler_decay`: lrop decay rate (default: 0.1)
- `--t0`: Cosine annealing warm restarts (cawr) period (default: 10)
- `--eta-min`: cawr minimum LR (default: 1e-5)
- `--ckpt-path`: Path to resume a saved PyTorch Lightning module with the .ckpt extension (default: None)

`LightningRFS` ([rfs.py](src/approach/rfs.py)) implements the RFS algorithm for Few-Shot Learning, as described in ["Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?"](https://arxiv.org/abs/2003.11539). Options include:

- `--alpha`: Weight for CE loss (default: 0.5)
- `--gamma`: Weight for KL loss (default: 0.5)
- `--is-distill`: Enable knowledge distillation if present
- `--kd-t`: Temperature for knowledge distillation loss (default: 1)
- `--teacher-path`: Path to the teacher model (default: None)
- `--base-learner`: Type of base learner ('lr' or 'nn') (default: 'nn')

### Network Options 

The [network](src/networks/network.py) class defines the architecture model to be used. This class includes methods to manage the model.

Options include:

- `--network`: Embedding function to use (default: `Lopez17CNN`)
- `--out-features-size`: Feature vector size (default: -1)
- `--weights-path`: Path to a `*.pt` file with weights to initialize the embedding function (default: None)
- `--scale`: Scaling factor for the neural network backbone (default: 1)

### Dateset Options

Datasets are defined in [`dataset_config.py`](src/data/dataset_config.py). Each key is considered a dataset name.

Options include:

- `--dataset`: Dataset to use (default: 'iot_nid')
- `--num-pkts`: Number of packets to select from the dataset (default: None)
- `--fields`: Packet field(s) used (default: [], choices: ['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'PL_DIR'])


### Early-Stopping Options

Early-stopping parameters are handled by `EarlyStoppingDoubleMetric` in [`callbacks.py`](src/util/callbacks.py), wrapping PyTorch Lightning's `EarlyStopping` to add features like logging and double monitored metrics.

Options include:

- `--monitor`: Metric to monitor for early stopping (default: 'valid_accuracy')
- `--min_delta`: Early-stopping minimum delta (default: 0.01)
- `--patience`: Early-stopping patience (default: 17)
- `--mode`: Early-stopping mode (default: 'auto')
- `--double-monitor`: Monitor both `valid_accuracy` and `valid_loss` if present


## Project Structure

### Approaches

In [src/approach](./src/approach/), you can find the PyTorch Lightning implementation of RFS.

### Modules

In [src/modules](./src/modules/), you can find implementations of various `nn.Module` components (e.g., losses, teacher).

### Data

In [src/data](./src/data/), several classes act as wrappers for the PyTorch Lightning dataloader modules.

### Trainers

In [src/trainers](./src/trainers/), there is a trainer class that acts as a wrapper for the PyTorch Lightning trainer, extending its functionality to manage custom logic (e.g., FSCIL).

### Utils

In [src/util](./src/util/), you can find various utility functions for managing the seed and RNG state, the logger, and callbacks.

### Networks

In [src/networks](./src/networks/), you can find implementations of embedding functions and the logic needed to manage them.

## Execution of Experiments

### Step 1: Train the Teacher

```bash
python3 main.py --is-fscil --dataset iot_nid --fields PL IAT DIR WIN FLG TTL --num-pkts 20 --shots 10 --queries 40 --gpus 1 --num-tasks 100 --max_epochs 200 --seed 0 --approach rfs --patience 20 --monitor valid_accuracy --min_delta 0.001 --mode max --double-monitor --lr 0.0001 --lr-strat none --classes-per-set 7 3 --default_root_dir ../results_rfs_teacher --network Lopez17CNN --base-learner nn

```

### Step 2: Train the Student

After completing the training, select the best model weights from the results folder (e.g., `../results_rfs_teacher/lightning_logs/version_0/distill_models/teacher_ep100.p`). Use these weights to initialize the teacher and train the student:

```bash
python3 main.py --is-fscil --dataset iot_nid --fields PL IAT DIR WIN FLG TTL --num-pkts 20 --shots 10 --queries 40 --gpus 1 --num-tasks 100 --max_epochs 200 --seed 0 --approach rfs --patience 20 --monitor valid_accuracy --min_delta 0.001 --mode max --double-monitor --lr 0.0001 --lr-strat none --classes-per-set 7 3 --default_root_dir ../results_rfs_student --network Lopez17CNN --base-learner nn --kd-t 1 --teacher-path ../results_rfs_teacher/lightning_logs/version_0/distill_models/teacher_ep100.pt --is-distill
```

Note that you can train a new student using the previous student as a teacher.

## Acknowledgement

We thank the following open-source implementations that were used in this work:

- [LibFewShot](https://github.com/RL-VIG/LibFewShot)
- [learn2learn](https://github.com/learnables/learn2learn/)

## Citation

```
@article{xx,
  title = 
}
```