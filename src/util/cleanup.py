import os
from glob import glob 


def cleanup_distill_models(path):
    ckpt_e = [v.split('=')[-1].split('.')[0] for v in glob(f'{path}/checkpoints/*')][0]
    if not os.path.isdir(f'{path}/distill_models/'):
        return
    for f in glob(f'{path}/distill_models/*'):
        f_norm = os.path.normpath(f)
        if os.path.exists(f_norm) and f_norm != os.path.normpath(f'{path}/distill_models/teacher_ep{ckpt_e}.pt'):
            os.remove(f_norm)
            print(f_norm)
            
def cleanup_autoencoder_models(path):
    # 获取checkpoints目录下最好的模型对应的epoch
    ckpt_e = [v.split('=')[-1].split('.')[0] for v in glob(f'{path}/checkpoints/*')][0]
    
    # 如果pretrain_models目录不存在，直接返回
    if not os.path.isdir(f'{path}/pretrain_models/'):
        return
        
    # 遍历pretrain_models目录下的所有文件
    for f in glob(f'{path}/pretrain_models/*'):
        f_norm = os.path.normpath(f)
        # 如果文件不是最好的模型对应的epoch，就删除它
        if os.path.exists(f_norm) and f_norm != os.path.normpath(f'{path}/pretrain_models/autoencoder_ep{ckpt_e}.pt'):
            os.remove(f_norm)
            print(f_norm)
            
def cleanup_embeddings(path):
    if not os.path.isdir(f'{path}/adaptation_data/'):
        return
    files = glob(f'{path}/adaptation_data/queries*') + glob(f'{path}/adaptation_data/supports*')
    for f in files:
        if os.path.exists(f):
            os.remove(f)