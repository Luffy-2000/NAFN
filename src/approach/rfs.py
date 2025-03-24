import os
import torch
import learn2learn as l2l
from torch import nn
from learn2learn.utils import accuracy

from approach.tl_module import LightningTLModule
from modules.losses import DistillKLLoss
from modules.teachers import Teacher
from modules.classifiers import NN, LR
<<<<<<< HEAD

=======
>>>>>>> 13490ca (Fix: Unsupervised Learning)
EPSILON = 1e-8


class LightningRFS(LightningTLModule):
    """
    [[Link to Source Code]](https://github.com/RL-VIG/LibFewShot)
    LightningRFS is a class that implements the RFS (Rethinking Few-Shot) algorithm for few-shot learning,
    as described in "Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?".

    Args:
        - net (``nn.Module``): The base neural network architecture.
        - loss (``nn.Module``, optional): The loss function for the algorithm. 
         Defaults to ``nn.CrossEntropyLoss(reduction="mean")``.
        - kd_T (float, optional): Temperature for knowledge distillation loss. Defaults to 1.
        - alpha (float, optional): Weight for CE loss. Defaults to 0.5.
        - gamma (float, optional): Weight for KL loss. Defaults to 0.5.
        - is_distill (bool, optional): Whether to use knowledge distillation. Defaults to False.
        - teacher_path (str, optional): Path to the teacher model. Defaults to None.
<<<<<<< HEAD
        - base_learner (str, optional): Type of base learner ('lr' or 'nn'). Defaults to 'lr'.
=======
        - base_learner (str, optional): Type of base learner ('lr' or 'nn'). Defaults to 'nn'.
        - pretrained_autoencoder (str, optional): Path to pretrained autoencoder. Defaults to None.
        - skip_training (bool, optional): Whether to skip training when using nn classifier. Defaults to False.
>>>>>>> 13490ca (Fix: Unsupervised Learning)
        - ``**kwargs`` (dict): Additional keyword arguments.

    Attributes:
        - kd_T (float): Temperature for knowledge distillation loss.
        - ce_loss (``nn.Module``): The cross-entropy loss function.
        - kl_loss (``DistillKLLoss``): The knowledge distillation KL loss function.
        - nl_loss (``nn.Module``): The negative log-likelihood loss function.
        - net (``nn.Module``): The base neural network architecture.
        - base_learner (str): Type of base learner ('lr' or 'nn').
        - is_distill (bool): Whether to use knowledge distillation.
        - teacher_path (str): Path to the teacher model.
        - alpha (float): Weight for CE loss.
        - gamma (float): Weight for KL loss.
        - teacher (``src.modules.teacher.Teacher``): Teacher model instance.
<<<<<<< HEAD
=======
        - is_unsupervised (bool): Whether to run in unsupervised mode.
>>>>>>> 13490ca (Fix: Unsupervised Learning)
    """
    
    alpha = 0.5
    gamma = 0.5
    is_distill = False
    kd_T = 1
    teacher_path = None
    base_learner = 'nn'
<<<<<<< HEAD
=======
    pretrained_autoencoder = None
    is_unsupervised = False
>>>>>>> 13490ca (Fix: Unsupervised Learning)
    
    def __init__(self, net, loss=None, **kwargs):
        super().__init__(**kwargs)
        
        # Transfer-Learning specific parameters 
        self.kd_T = kwargs.get('kd_t', LightningRFS.kd_T)
        self.ce_loss = loss or nn.CrossEntropyLoss(reduction="mean")
        self.kl_loss = DistillKLLoss(T=self.kd_T)
        self.nl_loss = nn.NLLLoss()
        self.net = net
        self.base_learner = kwargs.get('base_learner', LightningRFS.base_learner)
        self.is_distill = kwargs.get('is_distill', LightningRFS.is_distill)
        self.teacher_path = kwargs.get('teacher_path', LightningRFS.teacher_path)
        self.alpha = kwargs.get('alpha', LightningRFS.alpha) if self.is_distill else 0
        self.gamma = kwargs.get('gamma', LightningRFS.gamma) if self.is_distill else 1
<<<<<<< HEAD
        assert (
            self.alpha + self.gamma == 1.0
        ), 'alpha + gamma should be equal to 1'
=======
        self.pretrained_autoencoder = kwargs.get('pretrained_autoencoder', LightningRFS.pretrained_autoencoder)
        self.is_unsupervised = kwargs.get('is_unsupervised', LightningRFS.is_unsupervised)
        
        assert (
            self.alpha + self.gamma == 1.0
        ), 'alpha + gamma should be equal to 1'


        # 如果指定了预训练的自编码器，加载其权重并冻结
        if self.pretrained_autoencoder:
            print(f'Loading pretrained autoencoder from {self.pretrained_autoencoder}')
            checkpoint = torch.load(self.pretrained_autoencoder, weights_only=True)
            # 加载 encoder 参数
            # print(checkpoint['encoder'].keys())
            # exit()
            self.net.model.load_state_dict(checkpoint['encoder'])
            self.net.head.load_state_dict(checkpoint['head'])

            # **只冻结 `model` 部分，不冻结 `head`**
            for param in self.net.model.parameters():
                param.requires_grad = False

            # for param in self.net.model.bottleneck.parameters():
            #     param.requires_grad = True

            # print("Trainable parameters:")
            # for name, param in self.net.named_parameters():
            #     if param.requires_grad:
            #         print(name)



>>>>>>> 13490ca (Fix: Unsupervised Learning)
        self.teacher = Teacher(
            net=self.net, 
            is_distill=self.is_distill, 
            teacher_path=self.teacher_path
        )
        
        self.save_hyperparameters({
            "shots": self.shots,
            "queries": self.queries,
            "kd_T": self.kd_T,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "is_distill": self.is_distill,
            "teacher_path": self.teacher_path,
<<<<<<< HEAD
            "base_learner": self.base_learner
=======
            "base_learner": self.base_learner,
            "pretrained_autoencoder": self.pretrained_autoencoder,
            "is_unsupervised": self.is_unsupervised
>>>>>>> 13490ca (Fix: Unsupervised Learning)
        })


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningTLModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--alpha",
            type=float,
            default=LightningRFS.alpha
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=LightningRFS.gamma
        )
        parser.add_argument(
            "--is-distill", 
            action='store_true', 
            default=LightningRFS.is_distill
        )
        parser.add_argument(
            "--kd-t", 
            type=float, 
            default=LightningRFS.kd_T
        )
        parser.add_argument(
            "--teacher-path", 
            type=str, 
            default=LightningRFS.teacher_path
        )
        parser.add_argument(
            "--base-learner", 
            nargs='?', 
            choices=['lr', 'nn'],
            default=LightningRFS.base_learner
        )
<<<<<<< HEAD
=======
        parser.add_argument(
            "--pretrained-autoencoder",
            type=str,
            default=LightningRFS.pretrained_autoencoder,
            help="Path to pretrained autoencoder model"
        )
>>>>>>> 13490ca (Fix: Unsupervised Learning)
        return parser
    
    def on_adaptation_start(self):
        self.net.freeze_backbone()
        
    def training_epoch_end(self, _):
        # Save the network state dict since it can be used to inizialize the Teacher
        saved_weights_path = f'{self.logger.log_dir}/distill_models'
        os.makedirs(saved_weights_path, exist_ok=True)
        torch.save(self.net.state_dict(), 
                   f'{saved_weights_path}/teacher_ep{self.trainer.current_epoch}.pt')
<<<<<<< HEAD
           
    def pt_step(self, batch, batch_idx, **kwargs):
        data, labels = batch
        
        student_logits = self.net(data)
        teacher_logits = self.teacher(data)
        
=======
    

    def pt_step(self, batch, batch_idx, **kwargs):
        data, labels = batch
        student_logits = self.net(data)
        teacher_logits = self.teacher(data)    # None

>>>>>>> 13490ca (Fix: Unsupervised Learning)
        # CE on actual label and student logits
        gamma_loss = self.ce_loss(student_logits, labels) 
        # KL on teacher and student logits
        alpha_loss = self.kl_loss(student_logits, teacher_logits)
<<<<<<< HEAD
        
        eval_accuracy = accuracy(student_logits, labels)
        loss = gamma_loss * self.gamma + alpha_loss * self.alpha
        
=======
        eval_accuracy = accuracy(student_logits, labels)
        loss = gamma_loss * self.gamma + alpha_loss * self.alpha  # tensor(5.3006, device='cuda:0')

>>>>>>> 13490ca (Fix: Unsupervised Learning)
        return {
            'loss': loss,
            'accuracy': eval_accuracy,
            'labels': labels,
            'logits': student_logits
        }
        
<<<<<<< HEAD
=======

>>>>>>> 13490ca (Fix: Unsupervised Learning)
    @torch.no_grad()
    def ft_step(self, batch):
        # Grad disabled since the backbone is frozen and a fully-connected head is not used
        self.net.eval()
<<<<<<< HEAD
        
=======
>>>>>>> 13490ca (Fix: Unsupervised Learning)
        data, labels = batch
        labels, le = self.label_encoding(labels)

        way = labels.unique().size(0)

        # Embed the input
        _, feats = self.net(data, return_features=True)
        
        # Split the embedded input
        batch_support, batch_query = l2l.data.utils.partition_task(
            feats, labels, shots=self.shots)
        s_embeddings, support_labels = batch_support
        q_embeddings, query_labels = batch_query
        
        if self.base_learner == 'lr':
            # Logistic Regressor used to classify the query feats 
            y_pred = LR(
                x_support=s_embeddings, 
                y_support=support_labels, 
                x_query=q_embeddings,
            )
            y_pred = torch.tensor(y_pred)
           
        elif self.base_learner == 'nn':
            # Nearest-neighbour used to classify the query feats
            soft_values, y_pred = NN(
                x_support=s_embeddings, 
                y_support=support_labels, 
                x_query=q_embeddings, 
            )   
            soft_values = soft_values.clamp(EPSILON, 1 - EPSILON).to(self._device)
            eval_loss = self.nl_loss(soft_values.log(), query_labels)
            
        else:
            ValueError('Bad base learner')
           
        if self.base_learner == 'lr': 
            # Since 'lr' and 'h' do not produce soft values, 
            # one-hot encoding of predictions is performed.
            soft_values = torch.nn.functional.one_hot(y_pred, num_classes=way)  
            assert torch.equal(y_pred, torch.argmax(soft_values, dim=1))
            eval_loss = torch.zeros(1)
        
        y_pred = y_pred.to(self._device)
        query_accuracy = (y_pred == query_labels).sum().float() / y_pred.size(0)  
            
        return {
            'loss': eval_loss,
            'accuracy': query_accuracy,
            'query_labels': query_labels,
            'support_labels': support_labels,
            'preds': y_pred,
            'logits': soft_values,
            'le_map': le,
            'support': s_embeddings,
            'query': q_embeddings
        }