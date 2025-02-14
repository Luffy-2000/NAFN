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
            
def cleanup_embeddings(path):
    if not os.path.isdir(f'{path}/adaptation_data/'):
        return
    files = glob(f'{path}/adaptation_data/queries*') + glob(f'{path}/adaptation_data/supports*')
    for f in files:
        if os.path.exists(f):
            os.remove(f)